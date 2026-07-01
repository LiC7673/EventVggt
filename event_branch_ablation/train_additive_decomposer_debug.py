"""Standalone additive-event decomposition debug training.

This script intentionally does not train VGGT. It only checks whether a small
event/RGB decomposition head can learn:

    full event voxel -> geometry_motion / material_reflection / noise voxels

If pred_geometry stays black here, the downstream VGGT experiment has no chance
to use reliable geometry events.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import finetune_event as fe
from event_branch_ablation.data import FixedWindowAdditiveDataset
from event_branch_ablation.visualization import save_event_bin_visuals
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from eventvggt.models.streamvggt_additive_decomposition_detail import AdditiveEventTokenDecomposer


TARGET_KEYS = {
    "geometry": "event_geometry_voxel",
    "material": "event_material_voxel",
    "noise": "event_noise_voxel",
}


def _make_base_dataset(args, split: str):
    return get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=None,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.active_scene_count,
        split=split,
        test_frame_count=args.test_frame_count,
        ldr_event_id=args.ldr_event_id,
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_bins,
        return_normal_gt=False,
    )


def _make_loader(args, split: str) -> DataLoader:
    dataset = _make_base_dataset(args, split)
    dataset = FixedWindowAdditiveDataset(
        dataset,
        primary_branch="full",
        attach_targets=True,
        root_name=args.additive_event_root,
        mask_dilate_kernel=args.mask_dilate_kernel,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )


def _stack_views(views: List[Dict], key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([view[key] for view in views], dim=1).to(device)


def _dilate_support(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask
    pad = kernel // 2
    shape = mask.shape
    flat = mask.reshape(-1, 1, shape[-2], shape[-1]).float()
    flat = F.max_pool2d(flat, kernel_size=kernel, stride=1, padding=pad)
    return flat.reshape(shape)


def _branch_targets(
    views: List[Dict],
    device: torch.device,
    *,
    dilate_kernel: int,
    full_voxel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.stack(
        [_stack_views(views, TARGET_KEYS[name], device).clamp_min(0.0) for name in ("geometry", "material", "noise")],
        dim=2,
    )
    if dilate_kernel > 1:
        geometry = targets[:, :, 0]
        geo_energy = geometry.sum(dim=2, keepdim=True)
        geo_support = _dilate_support((geo_energy > 0).float(), dilate_kernel)
        # Dilate the supervision support without inventing event counts outside
        # the observed full event stream.
        full_share = full_voxel * geo_support
        targets[:, :, 0] = torch.maximum(geometry, full_share)

    target_sum = targets.sum(dim=2)
    full = full_voxel.clamp_min(0.0)
    # Keep branch targets on the same total mass as full events. This avoids
    # punishing small additive-rendering mismatches after resize/masking.
    scale = full / (target_sum + 1e-6)
    targets = targets * scale.unsqueeze(2)
    branch_share = targets / (full.unsqueeze(2) + 1e-6)
    return targets, branch_share, full


def _make_loss(
    branch_voxels: torch.Tensor,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    branch_share: torch.Tensor,
    full_voxel: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = torch.tensor(
        [args.geometry_weight, args.material_weight, args.noise_weight],
        dtype=branch_voxels.dtype,
        device=branch_voxels.device,
    ).view(1, 1, 3, 1, 1, 1)

    pred_log = torch.log1p(branch_voxels.clamp_min(0.0))
    target_log = torch.log1p(targets.clamp_min(0.0))
    recon_l1 = (weights * (pred_log - target_log).abs()).mean()

    # Directly supervise the per-channel branch partition. This is the key
    # diagnostic loss: pred_geometry should not be allowed to collapse to black.
    partition_l1 = (weights * (probabilities - branch_share.detach()).abs()).mean()

    full_energy = full_voxel.sum(dim=2, keepdim=True)
    geo_energy = targets[:, :, 0].sum(dim=2, keepdim=True)
    pred_geo_energy = branch_voxels[:, :, 0].sum(dim=2, keepdim=True)
    target_presence = _dilate_support((geo_energy > args.event_threshold).float(), args.presence_dilate_kernel)
    pred_presence = (pred_geo_energy / (full_energy + 1e-6)).clamp(1e-5, 1.0 - 1e-5)
    pos_weight = 1.0 + args.presence_pos_weight * target_presence
    presence_loss = (
        F.binary_cross_entropy(pred_presence, target_presence, reduction="none") * pos_weight
    ).mean()

    # Penalize assigning geometry probability where the full event stream itself
    # has no evidence. This keeps dilation from spreading to empty regions.
    empty = (full_energy <= args.event_threshold).float()
    empty_geo_loss = (probabilities[:, :, 0].mean(dim=2, keepdim=True) * empty).mean()

    total = (
        args.recon_weight * recon_l1
        + args.partition_weight * partition_l1
        + args.presence_weight * presence_loss
        + args.empty_weight * empty_geo_loss
    )

    with torch.no_grad():
        pred_mask = pred_presence > 0.5
        target_mask = target_presence > 0.5
        inter = (pred_mask & target_mask.bool()).float().sum()
        union = (pred_mask | target_mask.bool()).float().sum().clamp_min(1.0)
        geo_reliability = pred_presence[target_mask.bool()].mean() if target_mask.any() else pred_presence.mean()
        background_reliability = pred_presence[~target_mask.bool()].mean() if (~target_mask.bool()).any() else pred_presence.mean()
        metrics = {
            "loss": float(total.detach().cpu()),
            "recon_l1": float(recon_l1.detach().cpu()),
            "partition_l1": float(partition_l1.detach().cpu()),
            "presence_loss": float(presence_loss.detach().cpu()),
            "empty_geo_loss": float(empty_geo_loss.detach().cpu()),
            "geo_iou": float((inter / union).detach().cpu()),
            "geo_pred_mean": float(pred_presence.mean().detach().cpu()),
            "geo_target_mean": float(target_presence.mean().detach().cpu()),
            "geo_pos_pred_mean": float(geo_reliability.detach().cpu()),
            "geo_neg_pred_mean": float(background_reliability.detach().cpu()),
        }
    return total, metrics


def _update_average(accum: Dict[str, float], metrics: Dict[str, float], count: int) -> None:
    for key, value in metrics.items():
        accum[key] = accum.get(key, 0.0) + float(value) * count


def _finish_average(accum: Dict[str, float], total: int) -> Dict[str, float]:
    total = max(int(total), 1)
    return {key: value / total for key, value in sorted(accum.items())}


def _visual_cfg(output_dir: Path, args) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(output_dir),
        vis=SimpleNamespace(
            event_bins_enabled=True,
            save_every_steps=1,
            sample_index=0,
            event_bins_count=args.event_bins,
            event_bins_num_views=min(args.num_views, args.vis_num_views),
            event_bin_panel_width=args.vis_panel_width,
        ),
    )


def _save_visualization(
    model: AdditiveEventTokenDecomposer,
    views: List[Dict],
    device: torch.device,
    output_dir: Path,
    args,
    *,
    global_step: int,
    prefix: str,
) -> None:
    model.eval()
    with torch.no_grad():
        images = _stack_views(views, "img", device)
        full_voxel = _stack_views(views, "event_voxel", device).clamp_min(0.0)
        branch_voxels, _ = model(full_voxel, images)
    aux = {
        "pred_event_geometry_token": branch_voxels[:, :, 0].detach().cpu(),
        "pred_event_material_token": branch_voxels[:, :, 1].detach().cpu(),
        "pred_event_noise_token": branch_voxels[:, :, 2].detach().cpu(),
    }
    cpu_views = []
    for view in views:
        copied = dict(view)
        for key in ("event_voxel", "event_geometry_voxel", "event_material_voxel", "event_noise_voxel"):
            if key in copied and torch.is_tensor(copied[key]):
                copied[key] = copied[key].detach().cpu()
        cpu_views.append(copied)
    save_event_bin_visuals(
        fe,
        _visual_cfg(output_dir, args),
        cpu_views,
        aux,
        global_step,
        vis_subdir="debug_vis",
        force=True,
        filename_prefix=prefix,
    )


def run_epoch(
    model: AdditiveEventTokenDecomposer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args,
    *,
    max_batches: int,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    accum: Dict[str, float] = {}
    seen = 0
    for batch_idx, views in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = _stack_views(views, "img", device)
        full_voxel = _stack_views(views, "event_voxel", device).clamp_min(0.0)
        targets, branch_share, full_voxel = _branch_targets(
            views,
            device,
            dilate_kernel=args.geometry_dilate_kernel,
            full_voxel=full_voxel,
        )
        branch_voxels, probabilities = model(full_voxel, images)
        loss, metrics = _make_loss(branch_voxels, probabilities, targets, branch_share, full_voxel, args)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        batch_size = int(images.shape[0])
        _update_average(accum, metrics, batch_size)
        seen += batch_size
    return _finish_average(accum, seen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="abl_event_exp/additive_decomposer_debug")
    parser.add_argument("--additive-event-root", default="events_additive")
    parser.add_argument("--initial-scene-idx", type=int, default=12)
    parser.add_argument("--active-scene-count", type=int, default=3)
    parser.add_argument("--test-frame-count", type=int, default=12)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-bins", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--count-cmax", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-dilate-kernel", type=int, default=5)
    parser.add_argument("--geometry-dilate-kernel", type=int, default=9)
    parser.add_argument("--presence-dilate-kernel", type=int, default=9)
    parser.add_argument("--event-threshold", type=float, default=1.0e-5)
    parser.add_argument("--geometry-weight", type=float, default=4.0)
    parser.add_argument("--material-weight", type=float, default=1.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--partition-weight", type=float, default=2.0)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--presence-pos-weight", type=float, default=4.0)
    parser.add_argument("--empty-weight", type=float, default=0.25)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--vis-every-epochs", type=int, default=1)
    parser.add_argument("--vis-num-views", type=int, default=4)
    parser.add_argument("--vis-panel-width", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader = _make_loader(args, "train")
    val_loader = _make_loader(args, "test")

    model = AdditiveEventTokenDecomposer(
        num_bins=args.event_bins,
        hidden_dim=args.hidden_dim,
        count_cmax=args.count_cmax,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_iou = -math.inf
    history = []
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            args,
            max_batches=args.max_val_batches,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": time.time() - start,
        }
        history.append(record)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        print(
            "epoch {epoch:03d} "
            "train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            "val_iou={val_iou:.4f} val_geo_pos={val_pos:.4f} val_geo_neg={val_neg:.4f} "
            "time={time:.1f}s".format(
                epoch=epoch,
                train_loss=train_metrics.get("loss", float("nan")),
                val_loss=val_metrics.get("loss", float("nan")),
                val_iou=val_metrics.get("geo_iou", float("nan")),
                val_pos=val_metrics.get("geo_pos_pred_mean", float("nan")),
                val_neg=val_metrics.get("geo_neg_pred_mean", float("nan")),
                time=record["time"],
            ),
            flush=True,
        )

        if val_metrics.get("geo_iou", -math.inf) > best_iou:
            best_iou = val_metrics["geo_iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val": val_metrics,
                },
                output_dir / "decomposer-best.pth",
            )

        if args.vis_every_epochs > 0 and epoch % args.vis_every_epochs == 0:
            try:
                sample_views = next(iter(val_loader))
                _save_visualization(
                    model,
                    sample_views,
                    device,
                    output_dir,
                    args,
                    global_step=epoch,
                    prefix=f"epoch_{epoch:03d}_",
                )
            except Exception as exc:  # noqa: BLE001 - visualization must not kill training
                print(f"[warn] visualization failed at epoch {epoch}: {exc}", flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, output_dir / "decomposer-last.pth")
    print(f"done. output={output_dir} best_geo_iou={best_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
