"""Train ReliabilityNet on rendered real-event weak labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from reliability_pretrain.model import ReliabilityUNet
from real_reliability_stage.dataset import RenderedReliabilityDataset


def event_normalize(voxel: torch.Tensor, cmax: float) -> torch.Tensor:
    return torch.log1p(voxel.clamp_min(0.0).clamp_max(cmax)) / torch.log1p(
        voxel.new_tensor(float(cmax))
    )


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


def run_epoch(model, loader, optimizer, scaler, device, args, *, train: bool):
    model.train(train)
    totals = {}
    count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        event_full = batch["event_full"].to(device, non_blocking=True)
        target = batch["target_reliability"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        event_full = event_normalize(event_full, args.event_count_cmax)

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
            pred_bin = pred > args.binary_threshold
            target_bin = target > args.binary_threshold
            valid = weight > args.valid_weight_threshold
            tp = (pred_bin & target_bin & valid).float().sum()
            fp = (pred_bin & ~target_bin & valid).float().sum()
            fn = (~pred_bin & target_bin & valid).float().sum()
            iou = tp / (tp + fp + fn).clamp_min(1.0)
            pos_pred = pred[target_bin & valid].mean() if (target_bin & valid).any() else pred.mean()
            neg_pred = pred[(~target_bin) & valid].mean() if ((~target_bin) & valid).any() else pred.mean()
            metrics = {
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "smooth": float(smooth.detach().cpu()),
                "mae": float(weighted_l1(pred, target, weight).detach().cpu()),
                "iou": float(iou.detach().cpu()),
                "pred_mean": float(((pred * weight).sum() / weight.sum().clamp_min(1.0)).detach().cpu()),
                "target_mean": float(((target * weight).sum() / weight.sum().clamp_min(1.0)).detach().cpu()),
                "pos_pred_mean": float(pos_pred.detach().cpu()),
                "neg_pred_mean": float(neg_pred.detach().cpu()),
            }
        batch_count = int(rgb.shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_count
        count += batch_count
    return {key: value / max(count, 1) for key, value in sorted(totals.items())}


def _to_rgb01(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.min() < -0.05:
        rgb = (rgb + 1.0) * 0.5
    return np.clip(rgb, 0.0, 1.0)


def _gray(value: np.ndarray) -> np.ndarray:
    value = np.squeeze(value).astype(np.float32)
    finite = value[np.isfinite(value)]
    scale = max(float(np.percentile(finite, 99.5)) if finite.size else 1.0, 1e-6)
    panel = np.clip(value / scale, 0.0, 1.0)
    return np.repeat((panel * 255.0).round().astype(np.uint8)[..., None], 3, axis=-1)


def _event_panel(voxel: np.ndarray) -> np.ndarray:
    bins = voxel.shape[0] // 2
    pos = np.log1p(np.clip(voxel[:bins], 0.0, None).sum(axis=0))
    neg = np.log1p(np.clip(voxel[bins : 2 * bins], 0.0, None).sum(axis=0))
    scale = max(float(np.percentile(np.concatenate([pos.reshape(-1), neg.reshape(-1)]), 99.5)), 1e-6)
    out = np.zeros((*pos.shape, 3), dtype=np.float32)
    out[..., 0] = np.clip(pos / scale, 0.0, 1.0)
    out[..., 2] = np.clip(neg / scale, 0.0, 1.0)
    out[..., 1] = 0.25 * np.minimum(out[..., 0], out[..., 2])
    return (out * 255.0).round().astype(np.uint8)


def save_preview(model, dataset, device, args, out_dir: Path, epoch: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for idx in range(min(args.preview_count, len(dataset))):
        sample = dataset[idx]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        event = sample["event_full"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(event_normalize(event, args.event_count_cmax), rgb).squeeze(0).cpu().numpy()
        panels = [
            (_to_rgb01(sample["rgb"].numpy()) * 255.0).round().astype(np.uint8),
            _event_panel(sample["event_full"].numpy()),
            _gray(sample["geometry_support"].numpy()),
            _gray(sample["target_reliability"].numpy()),
            _gray(pred),
            _gray(sample["weight"].numpy()),
        ]
        labels = ["rgb", "event", "geo", "target", "pred", "weight"]
        h, w = panels[0].shape[:2]
        title_h = 20
        canvas = np.zeros((h + title_h, w * len(panels), 3), dtype=np.uint8)
        for col, (label, panel) in enumerate(zip(labels, panels)):
            x = col * w
            canvas[title_h:, x : x + w] = panel
            cv2.putText(canvas, label, (x + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        Image.fromarray(canvas).save(out_dir / f"epoch_{epoch:03d}_{idx:03d}.png")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="abl_event_exp/real_reliability_stage/labels")
    parser.add_argument("--out-dir", default="abl_event_exp/real_reliability_stage/reliability_net")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--event-count-cmax", type=float, default=3.0)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--valid-weight-threshold", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preview-count", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_dataset = RenderedReliabilityDataset(args.data_dir, split="train")
    val_dataset = RenderedReliabilityDataset(args.data_dir, split="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    print(f"train samples={len(train_dataset)}, val samples={len(val_dataset)}")

    model = ReliabilityUNet(event_channels=2 * args.num_bins, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    history = []
    best_iou = -1.0
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, args, train=False)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": time.time() - start,
        }
        history.append(record)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} pos={val_metrics['pos_pred_mean']:.4f} "
            f"neg={val_metrics['neg_pred_mean']:.4f} time={record['time']:.1f}s",
            flush=True,
        )
        metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(out_dir / "checkpoint-last.pth", model, optimizer, epoch, args, metrics)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(out_dir / "checkpoint-best.pth", model, optimizer, epoch, args, metrics)
        if args.preview_count > 0:
            save_preview(model, val_dataset, device, args, out_dir / "preview", epoch)
    print(f"done. best_iou={best_iou:.4f}, output={out_dir}")


if __name__ == "__main__":
    main()
