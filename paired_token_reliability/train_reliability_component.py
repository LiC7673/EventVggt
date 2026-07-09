from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from paired_token_reliability.common import normalize_event_voxel
from paired_token_reliability.dataset import PairedReliabilityDataset
from reliability_pretrain.model import ReliabilityUNet


TARGET_MODES = (
    "full",
    "event",
    "geometry",
    "token",
    "event_geometry",
    "event_token",
    "geometry_token",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReliabilityUNet with separated target components.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--event-count-cmax", type=float, default=3.0)
    parser.add_argument("--pair-weight", type=float, default=0.20)
    parser.add_argument("--rank-weight", type=float, default=0.10)
    parser.add_argument("--rank-margin", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


class ComponentTargetDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, target_mode: str) -> None:
        self.base = PairedReliabilityDataset(manifest, split)
        self.target_mode = str(target_mode)

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _load_component(npz, key: str) -> torch.Tensor:
        if key not in npz:
            raise KeyError(
                f"Target npz does not contain '{key}'. Re-run paired_token_reliability.export_targets "
                "after this patch so component maps are saved."
            )
        return torch.from_numpy(npz[key].astype("float32") / 255.0).unsqueeze(1)

    def _target_from_npz(self, npz) -> torch.Tensor:
        mode = self.target_mode
        if mode == "full":
            return torch.from_numpy(npz["target"].astype("float32") / 255.0).unsqueeze(1)
        event = self._load_component(npz, "event_support")
        geometry = self._load_component(npz, "geometry")
        token = self._load_component(npz, "token_agreement")
        if mode == "event":
            return event
        if mode == "geometry":
            return geometry
        if mode == "token":
            return token
        if mode == "event_geometry":
            return event * geometry
        if mode == "event_token":
            return event * token
        if mode == "geometry_token":
            return geometry * token
        raise ValueError(mode)

    def __getitem__(self, index):
        item = self.base[index]
        record = self.base.records[index]
        with torch.no_grad():
            npz = __import__("numpy").load(self.base.root / record["target"])
            item["target"] = self._target_from_npz(npz)
            item["weight"] = torch.from_numpy(npz["weight"].astype("float32") / 255.0).unsqueeze(1)
        item["target_mode"] = self.target_mode
        return item


def weighted_bce(logits, target, weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def ranking_loss(prediction, target, weight, margin, threshold):
    flat_pred = prediction.flatten(1)
    flat_target = target.flatten(1)
    flat_weight = weight.flatten(1) > 0
    losses = []
    for pred, tgt, valid in zip(flat_pred, flat_target, flat_weight):
        positive = pred[(tgt >= threshold) & valid]
        negative = pred[(tgt <= 0.05) & valid]
        if positive.numel() and negative.numel():
            losses.append(F.relu(float(margin) - positive.mean() + negative.mean()))
    return torch.stack(losses).mean() if losses else prediction.new_tensor(0.0)


def run_epoch(model, loader, optimizer, scaler, device, args, train):
    model.train(train)
    totals = {
        "loss": 0.0,
        "mae": 0.0,
        "iou": 0.0,
        "pair": 0.0,
        "rank": 0.0,
        "pred_pos": 0.0,
        "target_pos": 0.0,
    }
    count = 0
    for batch in loader:
        event = batch["event"].to(device, non_blocking=True)
        rgb_a = batch["rgb_a"].to(device, non_blocking=True)
        rgb_b = batch["rgb_b"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        batch_size, seq_len, channels, height, width = event.shape
        event = normalize_event_voxel(
            event.reshape(batch_size * seq_len, channels, height, width),
            args.event_count_cmax,
        )
        rgb_a = rgb_a.reshape(batch_size * seq_len, 3, height, width)
        rgb_b = rgb_b.reshape(batch_size * seq_len, 3, height, width)
        target = target.reshape(batch_size * seq_len, 1, height, width)
        weight = weight.reshape(batch_size * seq_len, 1, height, width)
        context = torch.enable_grad() if train else torch.no_grad()
        with context, torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits_a = model.forward_logits(event, rgb_a)
            logits_b = model.forward_logits(event, rgb_b)
            pred_a = torch.sigmoid(logits_a)
            pred_b = torch.sigmoid(logits_b)
            bce = weighted_bce(logits_a, target, weight) + weighted_bce(logits_b, target, weight)
            pair = ((pred_a - pred_b).abs() * weight).sum() / weight.sum().clamp_min(1.0)
            rank = 0.5 * (
                ranking_loss(pred_a, target, weight, args.rank_margin, args.threshold)
                + ranking_loss(pred_b, target, weight, args.rank_margin, args.threshold)
            )
            loss = 0.5 * bce + args.pair_weight * pair + args.rank_weight * rank
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        prediction = 0.5 * (pred_a.detach() + pred_b.detach())
        valid = weight > 0
        pred_pos = (prediction >= args.threshold) & valid
        target_pos = (target >= args.threshold) & valid
        intersection = (pred_pos & target_pos).sum()
        union = (pred_pos | target_pos).sum().clamp_min(1)
        totals["loss"] += float(loss.detach())
        totals["mae"] += float((prediction[valid] - target[valid]).abs().mean())
        totals["iou"] += float(intersection / union)
        totals["pair"] += float(pair.detach())
        totals["rank"] += float(rank.detach())
        totals["pred_pos"] += float(pred_pos.float().mean())
        totals["target_pos"] += float(target_pos.float().mean())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_set = ComponentTargetDataset(args.manifest, "train", args.target_mode)
    val_set = ComponentTargetDataset(args.manifest, "val", args.target_mode)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    event_channels = 2 * int(train_set.base.manifest["event_resize_bins"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReliabilityUNet(
        event_channels=event_channels,
        image_channels=3,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    history = []
    best_iou = -1.0
    print(
        f"mode={args.target_mode} train={len(train_set)} val={len(val_set)} "
        f"event_channels={event_channels}",
        flush=True,
    )
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args, True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, args, False)
        record = {"epoch": epoch, "target_mode": args.target_mode, "train": train_metrics, "val": val_metrics}
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "event_channels": event_channels,
            "epoch": epoch,
            "target_mode": args.target_mode,
        }
        torch.save(checkpoint, output / "checkpoint-last.pth")
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(checkpoint, output / "checkpoint-best.pth")
        with (output / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(
            f"epoch {epoch:03d} mode={args.target_mode} train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} mae={val_metrics['mae']:.4f} "
            f"iou={val_metrics['iou']:.4f} pred_pos={val_metrics['pred_pos']:.4f} "
            f"target_pos={val_metrics['target_pos']:.4f} pair={val_metrics['pair']:.4f} "
            f"rank={val_metrics['rank']:.4f} time={time.time()-start:.1f}s",
            flush=True,
        )
    print(f"done: mode={args.target_mode} best_iou={best_iou:.4f} output={output}")


if __name__ == "__main__":
    main()
