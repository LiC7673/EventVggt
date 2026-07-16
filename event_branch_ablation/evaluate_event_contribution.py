"""Paired counterfactual evaluation for additive event-branch experiments.

For every held-out sample, the same checkpoint is evaluated with:

* coarse_rgb: the frozen RGB VGGT prediction before the detail refiner;
* zero_event: the RGB/depth detail refiner with an all-zero event voxel;
* full_event: the complete additive event stream;
* geometry_event: the diffuse geometry-motion branch;
* material_event: the material-reflection branch;
* noise_event: the synthetic noise branch.

The comparison ``full_event - zero_event`` isolates the contribution of real
events from the capacity of the RGB/depth refinement head itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from ablation.eag3r_metrics_eval import (  # noqa: E402
    DepthMetrics,
    MeanAccumulator,
    cfg_from_checkpoint,
    move_views_to_device,
    pose_errors,
    stack_output,
    strip_module_prefix,
    torch_load,
)
from event_branch_ablation.common import (  # noqa: E402
    _build_decomposition_model,
    _build_geometry_model,
    _model_kwargs,
)
from event_branch_ablation.data import (  # noqa: E402
    FixedWindowAdditiveDataset,
    _build_base_dataset,
    switch_event_source,
)
from eventvggt.datasets.my_event_dataset import event_multiview_collate  # noqa: E402


EPS = 1e-8
CONDITION_EVENT_KEYS = {
    "zero_event": None,
    "full_event": "event_voxel",
    "geometry_event": "event_geometry_voxel",
    "material_event": "event_material_voxel",
    "noise_event": "event_noise_voxel",
}
DEFAULT_CONDITIONS = [
    "coarse_rgb",
    "zero_event",
    "full_event",
    "geometry_event",
    "material_event",
    "noise_event",
]


def _interior_normal_mask(
    valid_mask: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
) -> torch.Tensor:
    """Keep normals only where the complete 3x3 depth stencil is valid."""
    base = (
        valid_mask.bool()
        & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
        & (pred_depth > EPS) & (gt_depth > EPS)
    )
    shape = base.shape
    flat = base.float().reshape(-1, 1, shape[-2], shape[-1])
    # Average equals one iff all nine mask values are valid. This removes the
    # object silhouette where depth finite differences would touch background.
    interior = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1) >= (1.0 - 1e-6)
    return interior.reshape(shape)


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _safe_relative_reduction(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate) or abs(baseline) <= EPS:
        return float("nan")
    return 100.0 * (baseline - candidate) / abs(baseline)


class NormalMetrics:
    """Pixel-weighted depth-derived normal metrics used in paper tables."""

    def __init__(self) -> None:
        self.count = 0
        self.error_sum = 0.0
        self.error_sq_sum = 0.0
        self.threshold_counts = {11.25: 0, 22.5: 0, 30.0: 0}

    def update(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        pred_normal = fe.depth_to_normals(pred_depth, intrinsics)
        gt_normal = fe.depth_to_normals(gt_depth, intrinsics)
        mask = _interior_normal_mask(
            valid_mask, pred_depth, gt_depth
        )
        pred_normal = F.normalize(pred_normal.float(), dim=-1, eps=1e-6)
        gt_normal = F.normalize(gt_normal.float(), dim=-1, eps=1e-6)
        cosine = (pred_normal * gt_normal).sum(dim=-1).clamp(-1.0, 1.0)
        error = torch.rad2deg(torch.acos(cosine))
        mask &= torch.isfinite(error)
        if not mask.any():
            return
        values = error[mask]
        self.count += int(values.numel())
        self.error_sum += float(values.sum().detach().cpu())
        self.error_sq_sum += float(values.square().sum().detach().cpu())
        for threshold in self.threshold_counts:
            self.threshold_counts[threshold] += int((values < threshold).sum().detach().cpu())

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "normal_pixels": 0,
                "normal_mean_deg": float("nan"),
                "normal_rmse_deg": float("nan"),
                "normal_11_25": float("nan"),
                "normal_22_5": float("nan"),
                "normal_30": float("nan"),
            }
        count = float(self.count)
        return {
            "normal_pixels": self.count,
            "normal_mean_deg": self.error_sum / count,
            "normal_rmse_deg": math.sqrt(self.error_sq_sum / count),
            "normal_11_25": self.threshold_counts[11.25] / count,
            "normal_22_5": self.threshold_counts[22.5] / count,
            "normal_30": self.threshold_counts[30.0] / count,
        }


class ConditionAccumulator:
    def __init__(self) -> None:
        self.depth = DepthMetrics()
        self.depth_aligned = DepthMetrics()
        self.normal = NormalMetrics()
        self.pose = {name: MeanAccumulator() for name in ("ate", "rpe_trans", "rpe_rot_deg")}
        self.reliability = MeanAccumulator()

    def compute(self) -> Dict[str, float]:
        raw_depth = self.depth.compute()
        aligned = self.depth_aligned.compute()
        return {
            **raw_depth,
            **{f"median_{key}": value for key, value in aligned.items() if key != "depth_pixels"},
            **self.normal.compute(),
            **{key: value.compute() for key, value in self.pose.items()},
            "event_reliability_mean": self.reliability.compute(),
        }


def _ensure_model_cfg(cfg) -> None:
    OmegaConf.set_struct(cfg, False)
    for name in ("model", "data", "loss"):
        if not hasattr(cfg, name):
            setattr(cfg, name, OmegaConf.create({}))
        OmegaConf.set_struct(getattr(cfg, name), False)
    defaults = {
        "event_hidden_dim": 16,
        "event_num_bins": 10,
        "event_count_cmax": 3.0,
        "head_frames_chunk_size": 2,
        "refiner_residual_scale": 0.035,
        "event_delta_highpass_kernel": 9,
        "event_delta_patch_zero_mean": True,
        "event_delta_patch_size": 14,
        "event_delta_abs_limit": 0.025,
        "event_reliability_gate_floor": 0.20,
        "event_reliability_init_bias": 0.0,
        "refiner_refine_points": True,
        "refiner_use_checkpoint": False,
        "decomposition_hidden_dim": 24,
    }
    for key, value in defaults.items():
        if not hasattr(cfg.model, key):
            setattr(cfg.model, key, value)


def _detect_model_kind(state: Dict[str, torch.Tensor], requested: str, variant: str) -> str:
    if requested != "auto":
        return requested
    if "causal" in str(variant).lower():
        return "causal"
    if any(key.startswith("event_branch_decomposer.") for key in state):
        return "decomposition"
    return "geometry"


def _build_causal_model(cfg):
    from eventvggt.models.streamvggt_causal_additive_detail import StreamVGGT

    kwargs = _model_kwargs(cfg)
    kwargs["decomposition_hidden_dim"] = int(cfg.model.decomposition_hidden_dim)
    kwargs["event_support_tau"] = float(getattr(cfg.model, "event_support_tau", 0.50))
    kwargs["event_support_dilate_kernel"] = int(
        getattr(cfg.model, "event_support_dilate_kernel", 5)
    )
    kwargs["event_support_blur_kernel"] = int(
        getattr(cfg.model, "event_support_blur_kernel", 3)
    )
    return StreamVGGT(**kwargs)


def build_model(checkpoint: Path, args, device: torch.device):
    raw_checkpoint = torch_load(checkpoint)
    cfg = cfg_from_checkpoint(raw_checkpoint, args.config)
    _ensure_model_cfg(cfg)
    state = strip_module_prefix(fe.unwrap_state_dict(raw_checkpoint))
    kind = _detect_model_kind(state, args.model_kind, getattr(cfg.model, "variant", ""))
    if kind == "causal":
        model = _build_causal_model(cfg)
    elif kind == "decomposition":
        model = _build_decomposition_model(cfg)
    else:
        model = _build_geometry_model(cfg)
    message = model.load_state_dict(state, strict=False)
    print(
        f"[load] kind={kind} missing={len(message.missing_keys)} "
        f"unexpected={len(message.unexpected_keys)} checkpoint={checkpoint}"
    )
    if message.missing_keys:
        print(f"[load] missing sample: {message.missing_keys[:8]}")
    if message.unexpected_keys:
        print(f"[load] unexpected sample: {message.unexpected_keys[:8]}")
    model.to(device).eval()
    return model, cfg, kind


def build_heldout_loader(cfg, args):
    cfg.data.root = args.root or cfg.data.root
    cfg.data.num_views = int(args.num_views)
    cfg.data.resolution = list(args.resolution)
    cfg.data.initial_scene_idx = int(args.initial_scene_idx)
    cfg.data.active_scene_count = int(args.active_scene_count)
    cfg.data.test_frame_count = int(args.test_frame_count)
    cfg.data.scene_names = list(args.scene_names) if args.scene_names else None
    cfg.data.event_resize_method = args.event_resize_method
    cfg.data.event_resize_bins = int(args.event_resize_bins)
    cfg.data.eval_ldr_event_id = args.ldr_event_id
    cfg.data.random_train_ldr = False
    cfg.data.additive_event_root = args.additive_event_root
    cfg.data.additive_mask_dilate_kernel = int(args.mask_dilate_kernel)

    dataset = _build_base_dataset(cfg, args.split, args.ldr_event_id)
    switch_event_source(dataset, branch="full", root_name=args.additive_event_root)
    dataset = FixedWindowAdditiveDataset(
        dataset,
        primary_branch="full",
        attach_targets=True,
        root_name=args.additive_event_root,
        mask_dilate_kernel=args.mask_dilate_kernel,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    return dataset, loader


def _condition_views(views: List[Dict], condition: str) -> List[Dict]:
    if condition == "coarse_rgb":
        raise ValueError("coarse_rgb is extracted from the full-event forward pass")
    event_key = CONDITION_EVENT_KEYS[condition]
    conditioned: List[Dict] = []
    for view in views:
        current = dict(view)
        if condition == "zero_event":
            current["event_voxel"] = torch.zeros_like(view["event_voxel"])
        else:
            if event_key not in view:
                raise KeyError(f"Condition {condition} requires view field {event_key}")
            current["event_voxel"] = view[event_key]
        conditioned.append(current)
    return conditioned


def _sample_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    intrinsics: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, float]:
    mask = valid & torch.isfinite(pred) & torch.isfinite(gt) & (pred > EPS) & (gt > EPS)
    if not mask.any():
        return {"abs_rel": float("nan"), "delta1": float("nan"), "rmse_log": float("nan"), "normal_mean_deg": float("nan")}
    pred_value = pred[mask].float().clamp_min(EPS)
    gt_value = gt[mask].float().clamp_min(EPS)
    ratio = torch.maximum(pred_value / gt_value, gt_value / pred_value)
    abs_rel = ((pred_value - gt_value).abs() / gt_value).mean()
    rmse_log = torch.sqrt((torch.log(pred_value) - torch.log(gt_value)).square().mean())

    pred_normal = F.normalize(fe.depth_to_normals(pred, intrinsics).float(), dim=-1, eps=1e-6)
    gt_normal = F.normalize(fe.depth_to_normals(gt, intrinsics).float(), dim=-1, eps=1e-6)
    normal_mask = _interior_normal_mask(valid, pred, gt)
    angle = torch.rad2deg(torch.acos((pred_normal * gt_normal).sum(dim=-1).clamp(-1.0, 1.0)))
    normal_mask &= torch.isfinite(angle)
    normal_mean = angle[normal_mask].mean() if normal_mask.any() else angle.new_tensor(float("nan"))
    return {
        "abs_rel": float(abs_rel.detach().cpu()),
        "delta1": float((ratio < 1.25).float().mean().detach().cpu()),
        "rmse_log": float(rmse_log.detach().cpu()),
        "normal_mean_deg": float(normal_mean.detach().cpu()),
    }


def _update_condition(
    accumulator: ConditionAccumulator,
    condition: str,
    output,
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    intrinsics: torch.Tensor,
    gt_pose: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    accumulator.depth.update(depth_pred, depth_gt, valid, median_align=False)
    accumulator.depth_aligned.update(depth_pred, depth_gt, valid, median_align=True)
    accumulator.normal.update(depth_pred, depth_gt, intrinsics, valid)
    pose_key = "camera_pose_coarse" if condition == "coarse_rgb" else "camera_pose"
    pose = stack_output(output, pose_key)
    if pose is None and pose_key != "camera_pose":
        pose = stack_output(output, "camera_pose")
    if pose is not None:
        height, width = depth_pred.shape[-2:]
        pred_c2w, _ = fe.pose_encoding_to_c2w(pose.float(), image_size_hw=(height, width))
        for key, value in pose_errors(pred_c2w, gt_pose, scale_align=False).items():
            accumulator.pose[key].update(value)
    if condition != "coarse_rgb":
        reliability = stack_output(output, "event_reliability")
        if reliability is not None:
            accumulator.reliability.update(float(reliability.float().mean().detach().cpu()))


@torch.inference_mode()
def evaluate(model, loader, cfg, args, device: torch.device):
    conditions = list(dict.fromkeys(args.conditions))
    for condition in conditions:
        if condition not in DEFAULT_CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
    if "full_event" not in conditions:
        conditions.append("full_event")

    accumulators = {condition: ConditionAccumulator() for condition in conditions}
    records: List[Dict] = []
    start = time.time()
    evaluated_batches = 0
    for batch_idx, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        cpu_views = fe.maybe_denormalize_views(cpu_views)
        views = move_views_to_device(cpu_views, device)
        depth_gt = fe.stack_view_field(views, "depthmap").float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
        gt_pose = fe.stack_view_field(views, "camera_pose").float()
        valid = fe.build_valid_mask(
            views,
            depth_gt,
            depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
            depth_max=getattr(cfg.loss, "depth_max", None),
        )
        use_amp = args.amp != "none" and device.type == "cuda"
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            full_output = model(_condition_views(views, "full_event"))
        full_depth = stack_output(full_output, "depth")
        coarse_depth = stack_output(full_output, "depth_coarse")
        if full_depth is None or coarse_depth is None:
            raise RuntimeError("Model must return both depth and depth_coarse for paired evaluation")

        output_cache = {"full_event": (full_output, full_depth), "coarse_rgb": (full_output, coarse_depth)}
        for condition in conditions:
            if condition in output_cache:
                continue
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(_condition_views(views, condition))
            depth = stack_output(output, "depth")
            if depth is None:
                raise RuntimeError(f"Condition {condition} did not return depth")
            output_cache[condition] = (output, depth)

        for condition in conditions:
            output, depth = output_cache[condition]
            _update_condition(
                accumulators[condition], condition, output, depth, depth_gt, intrinsics, gt_pose, valid
            )
            batch_metrics = _sample_metrics(depth, depth_gt, intrinsics, valid)
            records.append({"batch_index": batch_idx, "condition": condition, **batch_metrics})

        evaluated_batches += 1
        if (batch_idx + 1) % args.print_freq == 0:
            print(f"[eval] {batch_idx + 1}/{len(loader)} elapsed={time.time() - start:.1f}s")

    return {key: value.compute() for key, value in accumulators.items()}, records, evaluated_batches


def _comparison(baseline: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    lower_is_better = ("abs_rel", "rmse_log", "normal_mean_deg", "normal_rmse_deg")
    higher_is_better = ("delta1", "normal_11_25", "normal_22_5", "normal_30")
    result = {}
    for key in lower_is_better:
        result[f"{key}_reduction"] = baseline[key] - candidate[key]
        result[f"{key}_relative_reduction_pct"] = _safe_relative_reduction(
            baseline[key], candidate[key]
        )
    for key in higher_is_better:
        result[f"{key}_increase"] = candidate[key] - baseline[key]
    return result


def _paired_ci(records: List[Dict], baseline: str, candidate: str) -> Dict[str, Dict[str, float]]:
    by_condition = {}
    for row in records:
        by_condition.setdefault(row["condition"], {})[int(row["batch_index"])] = row
    result = {}
    for metric in ("abs_rel", "delta1", "rmse_log", "normal_mean_deg"):
        values = []
        common = sorted(set(by_condition.get(baseline, {})) & set(by_condition.get(candidate, {})))
        for index in common:
            base_value = by_condition[baseline][index][metric]
            candidate_value = by_condition[candidate][index][metric]
            delta = (
                candidate_value - base_value
                if metric == "delta1"
                else base_value - candidate_value
            )
            if np.isfinite(delta):
                values.append(float(delta))
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean()) if array.size else float("nan")
        std = float(array.std(ddof=1)) if array.size > 1 else float("nan")
        ci95 = 1.96 * std / math.sqrt(array.size) if array.size > 1 else float("nan")
        result[metric] = {"mean_improvement": mean, "ci95": ci95, "n": int(array.size)}
    return result


def write_results(args, checkpoint: Path, kind: str, dataset, metrics, records, evaluated_batches) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    comparison_pairs = {
        "refiner_structure_zero_vs_coarse": ("coarse_rgb", "zero_event"),
        "full_event_net_gain": ("zero_event", "full_event"),
        "geometry_event_net_gain": ("zero_event", "geometry_event"),
        "geometry_vs_full": ("full_event", "geometry_event"),
        "material_event_net_gain": ("zero_event", "material_event"),
        "noise_event_net_gain": ("zero_event", "noise_event"),
    }
    for name, (baseline, candidate) in comparison_pairs.items():
        if baseline in metrics and candidate in metrics:
            comparisons[name] = {
                "baseline": baseline,
                "candidate": candidate,
                **_comparison(metrics[baseline], metrics[candidate]),
                "paired_ci": _paired_ci(records, baseline, candidate),
            }

    summary = {
        "checkpoint": str(checkpoint),
        "model_kind": kind,
        "split": args.split,
        "initial_scene_idx": args.initial_scene_idx,
        "active_scene_count": args.active_scene_count,
        "active_scenes": dataset.get_active_scenes(),
        "num_views": args.num_views,
        "evaluated_batches": evaluated_batches,
        "conditions": metrics,
        "comparisons": comparisons,
        "interpretation": {
            "event_causality": "Use full_event_net_gain, not final-vs-coarse, as the event-only gain.",
            "refiner_capacity": "refiner_structure_zero_vs_coarse measures RGB/depth refiner capacity without events.",
            "oracle": "geometry_event_net_gain estimates the clean geometry-event upper bound.",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "per_batch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    condition_rows = [{"condition": condition, **values} for condition, values in metrics.items()]
    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]) if condition_rows else [])
        if condition_rows:
            writer.writeheader()
            writer.writerows(condition_rows)

    print("\nPaired event contribution summary")
    print("condition         AbsRel    delta1   RMSElog  normal(deg)  <11.25")
    for condition, values in metrics.items():
        print(
            f"{condition:16s} {values['abs_rel']:8.5f} {values['delta1']:8.4f} "
            f"{values['rmse_log']:8.5f} {values['normal_mean_deg']:12.4f} "
            f"{values['normal_11_25']:8.4f}"
        )
    if "full_event_net_gain" in comparisons:
        gain = comparisons["full_event_net_gain"]
        print(
            "\nFull-event net gain over zero-event: "
            f"normal={gain['normal_mean_deg_reduction']:.4f} deg "
            f"({gain['normal_mean_deg_relative_reduction_pct']:.2f}%), "
            f"AbsRel={gain['abs_rel_reduction']:.6f} "
            f"({gain['abs_rel_relative_reduction_pct']:.2f}%)"
        )
    print(f"\nSaved evaluation to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired event-branch contribution evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--model-kind",
        choices=["auto", "geometry", "decomposition", "causal"],
        default="auto",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--initial-scene-idx", type=int, default=12)
    parser.add_argument("--active-scene-count", type=int, default=1)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--additive-event-root", default="events_additive")
    parser.add_argument("--mask-dilate-kernel", type=int, default=5)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--print-freq", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT_DIR / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, cfg, kind = build_model(checkpoint, args, device)
    dataset, loader = build_heldout_loader(cfg, args)
    print(
        f"[dataset] scenes={dataset.get_active_scenes()} samples={len(dataset)} "
        f"batches={len(loader)} conditions={args.conditions}"
    )
    metrics, records, evaluated_batches = evaluate(model, loader, cfg, args, device)
    write_results(args, checkpoint, kind, dataset, metrics, records, evaluated_batches)


if __name__ == "__main__":
    main()
