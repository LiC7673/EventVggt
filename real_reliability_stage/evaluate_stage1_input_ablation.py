"""Evaluate which inputs a pretrained Stage-1 ReliabilityNet actually uses.

The evaluation is performed on the rendered Stage-1 validation split. Besides
the standard RGB/event ablations, ``stage2_rgb_domain`` reproduces the RGB
conversion currently used when ReliabilityNet is embedded in Stage 2. This
exposes train/inference RGB-range mismatches directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

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
from real_reliability_stage.train_reliability_net import event_normalize


CONDITIONS = (
    "full_train_domain",
    "event_only",
    "rgb_only",
    "both_zero",
    "stage2_rgb_domain",
)


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_module_prefix(state):
    return {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.weight = 0.0
        self.abs_error = 0.0
        self.bce = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.pos_sum = 0.0
        self.pos_count = 0.0
        self.neg_sum = 0.0
        self.neg_count = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.full_difference = 0.0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        full_pred: torch.Tensor,
        *,
        binary_threshold: float,
        valid_weight_threshold: float,
    ) -> None:
        pred = pred.float()
        target = target.float()
        weight = weight.float()
        valid = weight > valid_weight_threshold
        effective_weight = weight * valid.to(weight.dtype)
        weight_sum = effective_weight.sum()

        self.weight += float(weight_sum)
        self.abs_error += float(((pred - target).abs() * effective_weight).sum())
        bce = F.binary_cross_entropy(
            pred.clamp(1.0e-6, 1.0 - 1.0e-6), target, reduction="none"
        )
        self.bce += float((bce * effective_weight).sum())
        self.pred_sum += float((pred * effective_weight).sum())
        self.target_sum += float((target * effective_weight).sum())
        self.full_difference += float(((pred - full_pred).abs() * effective_weight).sum())

        pred_binary = pred >= binary_threshold
        target_binary = target >= binary_threshold
        self.tp += float((pred_binary & target_binary & valid).sum())
        self.fp += float((pred_binary & ~target_binary & valid).sum())
        self.fn += float((~pred_binary & target_binary & valid).sum())

        positive = target_binary & valid
        negative = ~target_binary & valid
        self.pos_sum += float(pred[positive].sum())
        self.pos_count += float(positive.sum())
        self.neg_sum += float(pred[negative].sum())
        self.neg_count += float(negative.sum())

        self.sum_x += float((effective_weight * pred).sum())
        self.sum_y += float((effective_weight * target).sum())
        self.sum_x2 += float((effective_weight * pred.square()).sum())
        self.sum_y2 += float((effective_weight * target.square()).sum())
        self.sum_xy += float((effective_weight * pred * target).sum())

    def compute(self) -> Dict[str, float]:
        eps = 1.0e-12
        precision = self.tp / max(self.tp + self.fp, eps)
        recall = self.tp / max(self.tp + self.fn, eps)
        iou = self.tp / max(self.tp + self.fp + self.fn, eps)
        f1 = 2.0 * precision * recall / max(precision + recall, eps)
        covariance = self.sum_xy - self.sum_x * self.sum_y / max(self.weight, eps)
        variance_x = self.sum_x2 - self.sum_x * self.sum_x / max(self.weight, eps)
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / max(self.weight, eps)
        pearson = covariance / max(np.sqrt(max(variance_x * variance_y, 0.0)), eps)
        return {
            "weighted_mae": self.abs_error / max(self.weight, eps),
            "weighted_bce": self.bce / max(self.weight, eps),
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "weighted_pearson": pearson,
            "pred_mean": self.pred_sum / max(self.weight, eps),
            "target_mean": self.target_sum / max(self.weight, eps),
            "positive_pred_mean": self.pos_sum / max(self.pos_count, eps),
            "negative_pred_mean": self.neg_sum / max(self.neg_count, eps),
            "positive_negative_margin": (
                self.pos_sum / max(self.pos_count, eps)
                - self.neg_sum / max(self.neg_count, eps)
            ),
            "mean_abs_change_from_full": self.full_difference / max(self.weight, eps),
            "valid_weight_sum": self.weight,
        }


def _condition_inputs(
    event: torch.Tensor, rgb: torch.Tensor, condition: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition == "full_train_domain":
        return event, rgb
    if condition == "event_only":
        # ImgNorm maps middle gray to zero, making this a neutral missing-RGB input.
        return event, torch.zeros_like(rgb)
    if condition == "rgb_only":
        return torch.zeros_like(event), rgb
    if condition == "both_zero":
        return torch.zeros_like(event), torch.zeros_like(rgb)
    if condition == "stage2_rgb_domain":
        # Matches FrozenReliabilityEventFilter._prepare_rgb for minus_one_one.
        return event, (rgb + 1.0) * 0.5
    raise ValueError(f"Unknown condition: {condition}")


def _to_rgb_panel(rgb: torch.Tensor) -> np.ndarray:
    value = rgb.detach().float().cpu().numpy()
    if value.shape[0] == 3:
        value = np.transpose(value, (1, 2, 0))
    if float(value.min()) < -0.05:
        value = (value + 1.0) * 0.5
    return (np.clip(value, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _gray_panel(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().squeeze().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return np.repeat((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)[..., None], 3, axis=-1)


def _event_panel(event: torch.Tensor) -> np.ndarray:
    voxel = event.detach().float().cpu().numpy()
    bins = voxel.shape[0] // 2
    pos = np.log1p(np.clip(voxel[:bins], 0.0, None).sum(axis=0))
    neg = np.log1p(np.clip(voxel[bins : 2 * bins], 0.0, None).sum(axis=0))
    scale = max(float(np.percentile(np.concatenate([pos.ravel(), neg.ravel()]), 99.5)), 1.0e-6)
    panel = np.zeros((*pos.shape, 3), dtype=np.float32)
    panel[..., 0] = np.clip(pos / scale, 0.0, 1.0)
    panel[..., 2] = np.clip(neg / scale, 0.0, 1.0)
    return (panel * 255.0).round().astype(np.uint8)


def _save_preview(
    path: Path,
    rgb: torch.Tensor,
    event: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
) -> None:
    panels = [
        ("rgb", _to_rgb_panel(rgb)),
        ("event", _event_panel(event)),
        ("target", _gray_panel(target)),
        ("weight", _gray_panel(weight)),
    ]
    panels.extend((name, _gray_panel(pred)) for name, pred in predictions.items())
    height, width = panels[0][1].shape[:2]
    title_height = 22
    canvas = np.zeros((height + title_height, width * len(panels), 3), dtype=np.uint8)
    for index, (label, panel) in enumerate(panels):
        x = index * width
        canvas[title_height:, x : x + width] = panel
        cv2.putText(canvas, label, (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


@torch.inference_mode()
def evaluate(model, loader, device, args):
    accumulators = {name: MetricAccumulator() for name in CONDITIONS}
    preview_count = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        rgb = batch["rgb"].to(device, non_blocking=True)
        raw_event = batch["event_full"].to(device, non_blocking=True)
        target = batch["target_reliability"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        event = event_normalize(raw_event, args.event_count_cmax)
        predictions = {}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.amp == "bf16" and device.type == "cuda",
        ):
            for condition in CONDITIONS:
                condition_event, condition_rgb = _condition_inputs(event, rgb, condition)
                predictions[condition] = model(condition_event, condition_rgb).float()
        full_pred = predictions["full_train_domain"]
        for condition, pred in predictions.items():
            accumulators[condition].update(
                pred,
                target,
                weight,
                full_pred,
                binary_threshold=args.binary_threshold,
                valid_weight_threshold=args.valid_weight_threshold,
            )

        while preview_count < args.preview_count and preview_count < (batch_index + 1) * rgb.shape[0]:
            local_index = preview_count - batch_index * rgb.shape[0]
            if local_index < 0 or local_index >= rgb.shape[0]:
                break
            _save_preview(
                Path(args.output_dir) / "preview" / f"sample_{preview_count:04d}.png",
                rgb[local_index],
                raw_event[local_index],
                target[local_index],
                weight[local_index],
                {name: value[local_index] for name, value in predictions.items()},
            )
            preview_count += 1
        batches += 1
    return {name: accumulator.compute() for name, accumulator in accumulators.items()}, batches


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="abl_event_exp/real_reliability_stage/reliability_net/checkpoint-best.pth",
    )
    parser.add_argument("--data-dir", default="abl_event_exp/real_reliability_stage/labels")
    parser.add_argument(
        "--output-dir",
        default="abl_event_exp/real_reliability_stage/stage1_input_ablation",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--event-count-cmax", type=float, default=None)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--valid-weight-threshold", type=float, default=0.05)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    args.data_dir = str(data_dir)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    args.num_bins = int(args.num_bins or checkpoint_args.get("num_bins", 10))
    args.base_channels = int(args.base_channels or checkpoint_args.get("base_channels", 32))
    args.event_count_cmax = float(
        args.event_count_cmax or checkpoint_args.get("event_count_cmax", 3.0)
    )
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = ReliabilityUNet(
        event_channels=2 * args.num_bins,
        base_channels=args.base_channels,
    )
    message = model.load_state_dict(_strip_module_prefix(state), strict=True)
    print(f"[checkpoint] {checkpoint_path}; load={message}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dataset = RenderedReliabilityDataset(args.data_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    metrics, evaluated_batches = evaluate(model, loader, device, args)
    summary = {
        "checkpoint": str(checkpoint_path),
        "data_dir": args.data_dir,
        "split": args.split,
        "num_samples": len(dataset),
        "evaluated_batches": evaluated_batches,
        "conditions": metrics,
        "interpretation": {
            "event_dependence": "Compare full_train_domain with rgb_only.",
            "rgb_dependence": "Compare full_train_domain with event_only.",
            "integration_domain_shift": "Compare full_train_domain with stage2_rgb_domain.",
            "constant_output_bias": "Inspect both_zero and its positive/negative margin.",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    rows = [{"condition": name, **values} for name, values in metrics.items()]
    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nStage-1 ReliabilityNet input ablation")
    print("condition              MAE      IoU       F1  Pearson  Pos-Neg  |change|")
    for name, values in metrics.items():
        print(
            f"{name:22s} {values['weighted_mae']:7.4f} {values['iou']:8.4f} "
            f"{values['f1']:8.4f} {values['weighted_pearson']:8.4f} "
            f"{values['positive_negative_margin']:8.4f} "
            f"{values['mean_abs_change_from_full']:8.4f}"
        )
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
