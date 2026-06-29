"""Stage-1 training for geometry-event reliability.

This is deliberately independent from VGGT. It learns:

    ReliabilityNet(full event voxel, RGB) -> R_geo

where R_geo is supervised by additive event branches:

    R_geo_gt = abs(V_geometry) / (abs(V_full) + eps)
"""

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
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from reliability_pretrain.dataset import build_reliability_dataloader
from reliability_pretrain.model import ReliabilityUNet


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def weighted_bce_logits(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def smoothness_loss(pred: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dx = pred[..., :, 1:] - pred[..., :, :-1]
    wx = weight[..., :, 1:] * weight[..., :, :-1]
    dy = pred[..., 1:, :] - pred[..., :-1, :]
    wy = weight[..., 1:, :] * weight[..., :-1, :]
    return (dx.abs() * wx).sum() / wx.sum().clamp_min(1.0) + (dy.abs() * wy).sum() / wy.sum().clamp_min(1.0)


def event_normalize(voxel: torch.Tensor, cmax: float) -> torch.Tensor:
    return torch.log1p(voxel.clamp_min(0.0).clamp_max(cmax)) / torch.log1p(
        voxel.new_tensor(float(cmax))
    )


def run_epoch(model, loader, optimizer, scaler, device, args, train: bool):
    model.train(train)
    total = {}
    count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        event_full = batch["event_full"].to(device, non_blocking=True)
        target = batch["target_reliability"].to(device, non_blocking=True)
        presence = batch["event_presence"].to(device, non_blocking=True)
        event_full = event_normalize(event_full, args.event_count_cmax)
        weight = args.empty_weight + (1.0 - args.empty_weight) * presence

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits = model.forward_logits(event_full, rgb)
            pred = torch.sigmoid(logits)
            l1 = weighted_l1(pred, target, weight)
            bce = weighted_bce_logits(logits, target, weight)
            smooth = smoothness_loss(pred, weight)
            loss = args.l1_weight * l1 + args.bce_weight * bce + args.smooth_weight * smooth

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            pred_binary = pred > args.binary_threshold
            target_binary = target > args.binary_threshold
            valid = weight > 0
            tp = (pred_binary & target_binary & valid.bool()).sum().float()
            fp = (pred_binary & ~target_binary & valid.bool()).sum().float()
            fn = (~pred_binary & target_binary & valid.bool()).sum().float()
            iou = tp / (tp + fp + fn).clamp_min(1.0)
            mae = weighted_l1(pred, target, weight)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "l1": float(l1.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "smooth": float(smooth.detach().cpu()),
            "mae": float(mae.detach().cpu()),
            "iou": float(iou.detach().cpu()),
            "target_mean": float((target * weight).sum().detach().cpu() / weight.sum().detach().cpu().clamp_min(1.0)),
            "pred_mean": float((pred * weight).sum().detach().cpu() / weight.sum().detach().cpu().clamp_min(1.0)),
            "additive_error": float(batch["additive_error"].float().mean().cpu()),
        }
        for key, value in metrics.items():
            total[key] = total.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in total.items()}


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--out-dir", default="checkpoints/reliability_net_stage1")
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=12)
    parser.add_argument("--test-scene-count", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--event-count-cmax", type=float, default=3.0)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--empty-weight", type=float, default=0.05)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    common = dict(
        ldr_event_id=args.ldr_event_id,
        resolution=tuple(args.resolution),
        num_bins=args.num_bins,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.active_scene_count,
        test_scene_count=args.test_scene_count,
    )
    train_dataset, train_loader = build_reliability_dataloader(
        root=args.root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        **common,
    )
    val_dataset, val_loader = build_reliability_dataloader(
        root=args.root,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        **common,
    )
    print(f"train samples={len(train_dataset)}, val samples={len(val_dataset)}")
    print(f"train scenes={train_dataset.get_sampled_scene_names()}")
    print(f"val scenes={val_dataset.get_sampled_scene_names()}")

    model = ReliabilityUNet(event_channels=2 * args.num_bins, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp and device.type == "cuda",
    )
    writer = SummaryWriter(str(out_dir / "tb")) if SummaryWriter is not None else None
    best_val = float("inf")
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, args, train=False)
        for key, value in train_metrics.items():
            if writer is not None:
                writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            if writer is not None:
                writer.add_scalar(f"val/{key}", value, epoch)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"time={time.time() - start:.1f}s"
        )
        metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(out_dir / "checkpoint-last.pth", model, optimizer, epoch, args, metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(out_dir / "checkpoint-best.pth", model, optimizer, epoch, args, metrics)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
