"""Pretrain an explicit geometry/material/noise event-source decomposer."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_branch_ablation.data import (  # noqa: E402
    FixedWindowAdditiveDataset,
    switch_event_source,
)
from eventvggt.datasets.my_event_dataset import (  # noqa: E402
    event_multiview_collate,
    get_combined_dataset,
)
from eventvggt.models.streamvggt_additive_decomposition_detail import (  # noqa: E402
    AdditiveEventTokenDecomposer,
)
from paper_scale_training.scene_split_loader import load_scene_split  # noqa: E402


TARGET_KEYS = (
    "event_geometry_voxel",
    "event_material_voxel",
    "event_noise_voxel",
)


def _format_ldr(value):
    value = str(value)
    return value if value.startswith("ev_") else f"ev_{value}"


def _make_dataset(args, scenes, ldr_id):
    requested = list(scenes)
    base = get_combined_dataset(
        root=args.root,
        num_views=1,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=requested,
        initial_scene_idx=0,
        active_scene_count=len(requested),
        split="all",
        test_frame_count=0,
        ldr_event_id=_format_ldr(ldr_id),
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_bins,
        return_normal_gt=False,
    )
    missing = sorted(set(requested) - set(base.scenes))
    if missing:
        raise ValueError(f"Source pretraining scenes unavailable at LDR={ldr_id}: {missing}")
    base.set_active_scenes(requested)
    switch_event_source(base, branch="full", root_name=args.additive_event_root)
    return FixedWindowAdditiveDataset(
        base,
        primary_branch="full",
        attach_targets=True,
        root_name=args.additive_event_root,
        mask_dilate_kernel=args.mask_dilate_kernel,
    )


def _make_loaders(args):
    manifest = load_scene_split(args.scene_manifest)
    train_scenes = list(manifest["splits"]["train"])
    ldr_ids = [_format_ldr(value) for value in args.train_ldr_ids]
    train_sets = []
    for index, ldr_id in enumerate(ldr_ids):
        assigned = train_scenes[index :: len(ldr_ids)]
        train_sets.append(_make_dataset(args, assigned, ldr_id))
        print(f"[source data] train LDR={ldr_id} scenes={len(assigned)} frames={len(train_sets[-1])}")
    val_set = _make_dataset(args, manifest["splits"]["val"], args.val_ldr_id)
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        collate_fn=event_multiview_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    return train_loader, val_loader


def _normalize(value, cmax):
    return torch.log1p(value.clamp_min(0.0).clamp_max(cmax)) / math.log1p(cmax)


def _batch_tensors(views, device):
    view = views[0]
    rgb = view["img"].to(device, non_blocking=True).unsqueeze(1)
    full = view["event_voxel"].to(device, non_blocking=True).clamp_min(0.0).unsqueeze(1)
    targets = torch.stack(
        [view[key].to(device, non_blocking=True).clamp_min(0.0) for key in TARGET_KEYS],
        dim=1,
    )
    return rgb, full, targets


def _loss_and_metrics(branches, probabilities, targets, full, args):
    branches = branches.squeeze(1)
    probabilities = probabilities.squeeze(1)
    full = full.squeeze(1)
    target_sum = targets.sum(dim=1, keepdim=True)
    target_probability = targets / target_sum.clamp_min(1.0e-6)
    presence = (target_sum > args.event_threshold).to(targets.dtype)
    partition_error = (probabilities - target_probability).abs()
    partition_loss = (partition_error * presence).sum() / (
        presence.sum().clamp_min(1.0) * 3.0
    )

    reconstruction_weight = 0.05 + 0.95 * presence
    reconstruction = (
        (_normalize(branches, args.count_cmax) - _normalize(targets, args.count_cmax)).abs()
        * reconstruction_weight
    ).sum() / (reconstruction_weight.sum().clamp_min(1.0) * 3.0)

    full_energy = full.sum(dim=1, keepdim=True)
    target_geo = targets[:, 0].sum(dim=1, keepdim=True) / full_energy.clamp_min(1.0e-6)
    pred_geo = branches[:, 0].sum(dim=1, keepdim=True) / full_energy.clamp_min(1.0e-6)
    pixel_presence = (full_energy > args.event_threshold).to(full.dtype)
    geometry_loss = (
        F.smooth_l1_loss(pred_geo.clamp(0, 1), target_geo.clamp(0, 1), reduction="none")
        * pixel_presence
    ).sum() / pixel_presence.sum().clamp_min(1.0)

    loss = (
        args.partition_weight * partition_loss
        + args.reconstruction_weight * reconstruction
        + args.geometry_weight * geometry_loss
    )
    with torch.no_grad():
        active = presence.bool().expand_as(probabilities)
        source_accuracy = (
            (probabilities.argmax(dim=1) == target_probability.argmax(dim=1))
            * presence.squeeze(1).bool()
        ).sum() / presence.squeeze(1).sum().clamp_min(1.0)
        target_geo_mask = (target_geo >= 0.5) & pixel_presence.bool()
        pred_geo_mask = (pred_geo >= 0.5) & pixel_presence.bool()
        geometry_intersection = (target_geo_mask & pred_geo_mask).sum()
        geometry_union = (target_geo_mask | pred_geo_mask).sum()
        additive_error = (full - targets.sum(dim=1)).abs().mean()
        metrics = {
            "loss": loss.detach(),
            "partition_loss": partition_loss.detach(),
            "reconstruction_loss": reconstruction.detach(),
            "geometry_loss": geometry_loss.detach(),
            "geometry_mae": (
                (pred_geo - target_geo).abs() * pixel_presence
            ).sum().detach() / pixel_presence.sum().clamp_min(1.0),
            "source_accuracy": source_accuracy.detach(),
            "geometry_iou": (geometry_intersection / geometry_union.clamp_min(1)).detach(),
            "geometry_target_mean": (
                target_geo * pixel_presence
            ).sum().detach() / pixel_presence.sum().clamp_min(1.0),
            "geometry_pred_mean": (
                pred_geo * pixel_presence
            ).sum().detach() / pixel_presence.sum().clamp_min(1.0),
            "additive_error": additive_error.detach(),
        }
        del active
    return loss, metrics


def run_epoch(model, loader, optimizer, device, args):
    train = optimizer is not None
    model.train(train)
    totals = {}
    count = 0
    max_batches = args.max_train_batches if train else args.max_val_batches
    for batch_index, views in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        rgb, full, targets = _batch_tensors(views, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            branches, probabilities = model(full, rgb)
            loss, metrics = _loss_and_metrics(branches, probabilities, targets, full, args)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        batch_size = int(rgb.shape[0])
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.float().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def _save(path, model, optimizer, epoch, args, metrics):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ldr-ids", nargs="+", default=["ev_2", "ev_5", "ev_10"])
    parser.add_argument("--val-ldr-id", default="ev_5")
    parser.add_argument("--additive-event-root", default="events_additive")
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-bins", type=int, default=10)
    parser.add_argument("--mask-dilate-kernel", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--count-cmax", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--partition-weight", type=float, default=2.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--event-threshold", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader, val_loader = _make_loaders(args)
    model = AdditiveEventTokenDecomposer(
        num_bins=args.event_bins,
        hidden_dim=args.hidden_dim,
        count_cmax=args.count_cmax,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best = float("inf")
    history = []
    bad_epochs = 0
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, args)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": time.time() - start,
        }
        history.append(record)
        (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _save(output / "checkpoint-last.pth", model, optimizer, epoch, args, record)
        if val_metrics["loss"] < best - 1.0e-4:
            best = val_metrics["loss"]
            bad_epochs = 0
            _save(output / "checkpoint-best.pth", model, optimizer, epoch, args, record)
        else:
            bad_epochs += 1
        print(
            f"epoch {epoch:03d} train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} geo_mae={val_metrics['geometry_mae']:.4f} "
            f"geo_iou={val_metrics['geometry_iou']:.4f} "
            f"source_acc={val_metrics['source_accuracy']:.4f} "
            f"additive={val_metrics['additive_error']:.6f} bad={bad_epochs}/5 "
            f"time={record['time']:.1f}s",
            flush=True,
        )
        if bad_epochs >= 5:
            break
    print(f"done. best_val_loss={best:.6f}, checkpoint={output / 'checkpoint-best.pth'}")


if __name__ == "__main__":
    main()
