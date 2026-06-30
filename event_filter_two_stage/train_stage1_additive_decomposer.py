"""Stage 1: pretrain full+RGB to additive event-stream decomposition."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from omegaconf import OmegaConf

from event_filter_two_stage.data import build_full_event_dataset, make_loader
from eventvggt.models.streamvggt_additive_decomposition_detail import (
    AdditiveEventTokenDecomposer,
)


TARGET_KEYS = ("event_geometry_voxel", "event_material_voxel", "event_noise_voxel")


def normalize_token(token: torch.Tensor, cmax: float) -> torch.Tensor:
    return torch.log1p(token.clamp_min(0.0).clamp_max(cmax)) / torch.log1p(
        token.new_tensor(cmax)
    )


def stack_views(views, key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([view[key] for view in views], dim=1).to(device, non_blocking=True)


def run_epoch(model, loader, optimizer, device, args, *, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "geometry": 0.0, "material": 0.0, "noise": 0.0, "consistency": 0.0}
    batches = 0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for views in loader:
            rgb = stack_views(views, "img", device)
            full = stack_views(views, "event_voxel", device)
            targets = torch.stack(
                [stack_views(views, key, device) for key in TARGET_KEYS], dim=2
            )
            presence = (targets.sum(dim=2) > 0).to(dtype=full.dtype)
            weight = 0.02 + 0.98 * presence
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ):
                predictions, _ = model(full, rgb)
                branch_losses = []
                for branch_idx in range(3):
                    error = (
                        normalize_token(predictions[:, :, branch_idx], args.event_count_cmax)
                        - normalize_token(targets[:, :, branch_idx], args.event_count_cmax)
                    ).abs()
                    branch_losses.append((error * weight).sum() / weight.sum().clamp_min(1.0))
                consistency = (
                    normalize_token(predictions.sum(dim=2), args.event_count_cmax)
                    - normalize_token(targets.sum(dim=2), args.event_count_cmax)
                ).abs().mean()
                loss = (
                    args.geometry_weight * branch_losses[0]
                    + args.material_weight * branch_losses[1]
                    + args.noise_weight * branch_losses[2]
                    + args.consistency_weight * consistency
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            values = [loss, *branch_losses, consistency]
            for key, value in zip(totals, values):
                totals[key] += float(value.detach())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--out-dir", default="abl_event_exp/additive_decomposer_stage1_scene12")
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=12)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--event-count-cmax", type=float, default=3.0)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--material-weight", type=float, default=0.75)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_cfg(args):
    return OmegaConf.create(
        {
            "seed": 0,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_mem": False,
            "data": {
                "root": args.root,
                "num_views": args.num_views,
                "resolution": list(args.resolution),
                "fps": 120,
                "scene_names": None,
                "initial_scene_idx": args.initial_scene_idx,
                "active_scene_count": args.active_scene_count,
                "test_frame_count": args.test_frame_count,
                "event_resize_method": "voxel_antialias",
                "event_resize_bins": args.num_bins,
                "random_train_ldr": True,
                "eval_ldr_event_id": args.ldr_event_id,
                "additive_event_root": "events_additive",
            },
        }
    )


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    cfg = make_cfg(args)
    train_dataset = build_full_event_dataset(cfg, split="train", attach_targets=True)
    val_dataset = build_full_event_dataset(cfg, split="test", attach_targets=True)
    train_loader = make_loader(cfg, train_dataset, split="train")
    val_loader = make_loader(cfg, val_dataset, split="test")
    print(f"stage1 train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"scenes={train_dataset.get_active_scenes()}")

    model = AdditiveEventTokenDecomposer(
        num_bins=args.num_bins,
        hidden_dim=args.hidden_dim,
        count_cmax=args.event_count_cmax,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False)
        metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(out_dir / "checkpoint-last.pth", model, optimizer, epoch, args, metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(out_dir / "checkpoint-best.pth", model, optimizer, epoch, args, metrics)
        print(
            f"epoch {epoch:03d} train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} geo={val_metrics['geometry']:.5f} "
            f"mat={val_metrics['material']:.5f} noise={val_metrics['noise']:.5f} "
            f"time={time.time() - start:.1f}s"
        )


if __name__ == "__main__":
    main()

