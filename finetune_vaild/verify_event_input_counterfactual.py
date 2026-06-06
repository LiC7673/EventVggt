"""Verify whether event input changes depth predictions under controlled perturbations.

For the same RGB views and checkpoint, this script runs four event conditions:
real events, zero events, reversed temporal bins, and swapped polarities. A
model whose event branch is disconnected should produce virtually identical
outputs for all conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from eventvggt.datasets.my_event_dataset import event_multiview_collate  # noqa: E402
from exp_test import visualize_normal_error_event_corr as corr_vis  # noqa: E402


CONDITIONS = ("real", "zero", "reverse_time", "swap_polarity")
EPS = 1e-8


def nanmean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def perturb_event_views(views: List[Dict], condition: str) -> List[Dict]:
    perturbed = [dict(view) for view in views]
    if condition == "real":
        return perturbed

    for out, original in zip(perturbed, views):
        voxel = original.get("event_voxel")
        if torch.is_tensor(voxel) and voxel.ndim == 4 and voxel.shape[1] >= 2:
            num_bins = voxel.shape[1] // 2
            pos = voxel[:, :num_bins]
            neg = voxel[:, num_bins : 2 * num_bins]
            remainder = voxel[:, 2 * num_bins :]
            if condition == "zero":
                out["event_voxel"] = torch.zeros_like(voxel)
            elif condition == "reverse_time":
                out["event_voxel"] = torch.cat([pos.flip(1), neg.flip(1), remainder], dim=1)
            elif condition == "swap_polarity":
                out["event_voxel"] = torch.cat([neg, pos, remainder], dim=1)

        if condition == "zero":
            for key in ("event_xy", "event_t", "event_p", "events"):
                values = original.get(key)
                if isinstance(values, list):
                    out[key] = [value[:0] if torch.is_tensor(value) else value for value in values]
                elif torch.is_tensor(values):
                    out[key] = values[:0]
        elif condition == "swap_polarity":
            values = original.get("event_p")
            if isinstance(values, list):
                out["event_p"] = [-value if torch.is_tensor(value) else value for value in values]
            elif torch.is_tensor(values):
                out["event_p"] = -values
    return perturbed


@torch.inference_mode()
def predict_depth(model, views: List[Dict], condition: str, device: torch.device) -> torch.Tensor:
    condition_views = perturb_event_views(views, condition)
    condition_views = corr_vis.move_views_to_device(condition_views, device)
    condition_views = fe.maybe_denormalize_views(condition_views)
    output = model(condition_views)
    return torch.stack([res["depth"] for res in output.ress], dim=1).squeeze(-1).detach().cpu()


def masked_mean_tensor(value: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool() & torch.isfinite(value)
    return float(value[valid].mean()) if valid.any() else float("nan")


def normal_valid_mask(views: List[Dict], depth_gt: torch.Tensor) -> torch.Tensor:
    mask = fe.build_valid_mask(views, depth_gt, depth_min=1e-6).cpu()
    mask[..., 0, :] = False
    mask[..., -1, :] = False
    mask[..., :, 0] = False
    mask[..., :, -1] = False
    return mask


def event_regions(view: Dict, mask: torch.Tensor, args) -> tuple[np.ndarray, np.ndarray]:
    sample_view = corr_vis.select_sample(view, 0)
    pos, neg = corr_vis.event_voxel_parts(sample_view, args.event_resize_bins)
    support = corr_vis.event_support_from_parts(pos, neg, mode=args.event_support_mode)
    return corr_vis.ranked_event_regions(
        support,
        mask.cpu().numpy().astype(bool),
        high_fraction=args.event_high_fraction,
        low_fraction=args.event_low_fraction,
    )


def resized_panel(label: str, image: np.ndarray, width: int) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if width > 0 and panel.width != width:
        height = max(1, int(round(panel.height * float(width) / float(panel.width))))
        panel = panel.resize((width, height), Image.Resampling.BILINEAR)
    return corr_vis.label_panel(label, np.asarray(panel))


def make_visualization(
    path: Path,
    views: List[Dict],
    predictions: Dict[str, torch.Tensor],
    gt_normals: torch.Tensor,
    masks: torch.Tensor,
    args,
) -> None:
    rows = []
    max_views = min(args.visualize_views, len(views))
    real_normals = fe.depth_to_normals(predictions["real"][0], torch.stack([v["camera_intrinsics"][0] for v in views]))
    zero_normals = fe.depth_to_normals(predictions["zero"][0], torch.stack([v["camera_intrinsics"][0] for v in views]))
    reverse_normals = fe.depth_to_normals(
        predictions["reverse_time"][0],
        torch.stack([v["camera_intrinsics"][0] for v in views]),
    )

    for view_idx in range(max_views):
        sample_view = corr_vis.select_sample(views[view_idx], 0)
        pos, neg = corr_vis.event_voxel_parts(sample_view, args.event_resize_bins)
        mask = masks[0, view_idx]
        real_error = corr_vis.normal_error_deg(real_normals[view_idx], gt_normals[0, view_idx], mask)
        zero_error = corr_vis.normal_error_deg(zero_normals[view_idx], gt_normals[0, view_idx], mask)
        reverse_error = corr_vis.normal_error_deg(reverse_normals[view_idx], gt_normals[0, view_idx], mask)
        event_delta = corr_vis.normal_error_deg(real_normals[view_idx], zero_normals[view_idx], mask)
        rows.append(
            [
                resized_panel("rgb", corr_vis.tensor_image_to_uint8(sample_view["img"]), args.panel_width),
                resized_panel("real event", corr_vis.event_rgb_from_parts(pos, neg), args.panel_width),
                resized_panel("gt normal", fe.normal_to_uint8(gt_normals[0, view_idx], mask), args.panel_width),
                resized_panel("real pred", fe.normal_to_uint8(real_normals[view_idx], mask), args.panel_width),
                resized_panel("real err", corr_vis.error_to_rgb(real_error.numpy(), mask.numpy()), args.panel_width),
                resized_panel("zero pred", fe.normal_to_uint8(zero_normals[view_idx], mask), args.panel_width),
                resized_panel("zero err", corr_vis.error_to_rgb(zero_error.numpy(), mask.numpy()), args.panel_width),
                resized_panel("real-zero delta", corr_vis.error_to_rgb(event_delta.numpy(), mask.numpy()), args.panel_width),
                resized_panel("reverse err", corr_vis.error_to_rgb(reverse_error.numpy(), mask.numpy()), args.panel_width),
            ]
        )
    corr_vis.make_grid(rows).save(path)


def evaluate(args, model, device: torch.device) -> List[Dict]:
    dataset = corr_vis.build_dataset(args, args.split)
    dataset, selected_counts = corr_vis.select_scene_samples(dataset, args.samples_per_scene)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=event_multiview_collate,
    )
    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    print(f"selected_samples={len(dataset)}, selected_per_scene={selected_counts}, conditions={CONDITIONS}")

    records: List[Dict] = []
    for sample_idx, views in enumerate(loader):
        if args.max_samples is not None and sample_idx >= args.max_samples:
            break
        predictions = {condition: predict_depth(model, views, condition, device) for condition in CONDITIONS}
        depth_gt = fe.stack_view_field(views, "depthmap").float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
        masks = normal_valid_mask(views, depth_gt)
        gt_normals = fe.depth_to_normals(depth_gt, intrinsics)
        pred_normals = {
            condition: fe.depth_to_normals(predictions[condition], intrinsics)
            for condition in CONDITIONS
        }
        real_normals = pred_normals["real"]

        for view_idx in range(len(views)):
            high_np, low_np = event_regions(views[view_idx], masks[0, view_idx], args)
            high = torch.from_numpy(high_np)
            low = torch.from_numpy(low_np)
            for condition in CONDITIONS:
                error = corr_vis.normal_error_deg(pred_normals[condition][0, view_idx], gt_normals[0, view_idx], masks[0, view_idx])
                normal_delta = corr_vis.normal_error_deg(
                    pred_normals[condition][0, view_idx],
                    real_normals[0, view_idx],
                    masks[0, view_idx],
                )
                depth_delta = (predictions[condition][0, view_idx] - predictions["real"][0, view_idx]).abs()
                records.append(
                    {
                        "sample_index": sample_idx,
                        "view_index": view_idx,
                        "condition": condition,
                        "normal_error_mean_deg": masked_mean_tensor(error, masks[0, view_idx]),
                        "high_event_normal_error_mean_deg": masked_mean_tensor(error, high),
                        "low_event_normal_error_mean_deg": masked_mean_tensor(error, low),
                        "depth_change_from_real_mean_abs": masked_mean_tensor(depth_delta, masks[0, view_idx]),
                        "normal_change_from_real_mean_deg": masked_mean_tensor(normal_delta, masks[0, view_idx]),
                    }
                )

        if sample_idx < args.max_visualizations:
            make_visualization(
                visual_dir / f"counterfactual_sample_{sample_idx:03d}.png",
                views,
                predictions,
                gt_normals,
                masks,
                args,
            )
    return records


def write_results(args, records: List[Dict]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    fields = (
        "normal_error_mean_deg",
        "high_event_normal_error_mean_deg",
        "low_event_normal_error_mean_deg",
        "depth_change_from_real_mean_abs",
        "normal_change_from_real_mean_deg",
    )
    conditions = {
        condition: {
            field: nanmean(row[field] for row in records if row["condition"] == condition)
            for field in fields
        }
        for condition in CONDITIONS
    }
    real = conditions["real"]
    comparison = {}
    for condition in CONDITIONS[1:]:
        result = conditions[condition]
        comparison[f"real_vs_{condition}"] = {
            "normal_error_advantage_deg": result["normal_error_mean_deg"] - real["normal_error_mean_deg"],
            "high_event_error_advantage_deg": (
                result["high_event_normal_error_mean_deg"] - real["high_event_normal_error_mean_deg"]
            ),
            "depth_output_change_mean_abs": result["depth_change_from_real_mean_abs"],
            "normal_output_change_mean_deg": result["normal_change_from_real_mean_deg"],
        }

    zero_change = comparison["real_vs_zero"]["normal_output_change_mean_deg"]
    summary = {
        "checkpoint": args.checkpoint,
        "model_variant": args.model_variant,
        "num_records": len(records),
        "num_view_predictions": len(records) // len(CONDITIONS),
        "conditions": conditions,
        "comparisons": comparison,
        "event_output_sensitivity_detected": bool(np.isfinite(zero_change) and zero_change > args.sensitivity_epsilon_deg),
        "sensitivity_epsilon_deg": args.sensitivity_epsilon_deg,
        "interpretation": {
            "disconnected_event_path": "If output changes are approximately zero, events do not affect predictions.",
            "useful_event_path": "Real events should yield positive normal-error advantage over zero/reversed events.",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual check of event contribution to StreamVGGT output")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-variant", default="base")
    parser.add_argument("--output-dir", default="finetune_vaild/results/event_counterfactual")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--active-scene-count", type=int, default=4)
    parser.add_argument("--samples-per-scene", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-visualizations", type=int, default=4)
    parser.add_argument("--visualize-views", type=int, default=4)
    parser.add_argument("--panel-width", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--event-spatial-transform", default="auto")
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--event-support-mode", default="temporal_polarity")
    parser.add_argument("--event-high-fraction", type=float, default=0.2)
    parser.add_argument("--event-low-fraction", type=float, default=0.2)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--event-hidden-dim", type=int, default=32)
    parser.add_argument("--event-num-bins", type=int, default=10)
    parser.add_argument("--event-count-cmax", type=float, default=3.0)
    parser.add_argument("--event-fusion-scale", type=float, default=1.0)
    parser.add_argument("--event-gate-downsample", type=int, default=4)
    parser.add_argument("--event-gate-smooth-kernel", type=int, default=5)
    parser.add_argument("--proposal-depth-lowpass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--proposal-use-depth-hf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-proposal-weight", type=float, default=0.0)
    parser.add_argument("--event-delta-highpass-kernel", type=int, default=0)
    parser.add_argument("--event-delta-patch-zero-mean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-delta-patch-size", type=int, default=14)
    parser.add_argument("--event-delta-abs-limit", type=float, default=0.0)
    parser.add_argument("--event-reliability-gate-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--event-reliability-gate-floor", type=float, default=0.10)
    parser.add_argument("--event-reliability-init-bias", type=float, default=0.0)
    parser.add_argument("--final-degrid-strength", type=float, default=0.0)
    parser.add_argument("--final-degrid-kernel", type=int, default=5)
    parser.add_argument("--head-frames-chunk-size", type=int, default=2)
    parser.add_argument("--refiner-hidden-dim", type=int, default=16)
    parser.add_argument("--refiner-num-blocks", type=int, default=2)
    parser.add_argument("--refiner-residual-scale", type=float, default=0.03)
    parser.add_argument("--refiner-refine-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sensitivity-epsilon-deg", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = corr_vis.build_model(args, device)
    records = evaluate(args, model, device)
    write_results(args, records)


if __name__ == "__main__":
    main()
