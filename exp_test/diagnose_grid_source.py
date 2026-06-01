"""Diagnose whether patch-grid artifacts come from coarse depth or detail residuals.

The script runs the same EventVGGT dataset/model path as ``finetune_event.py`` and
compares grid-boundary energy in:

  1. coarse depth from the VGGT/DPT head,
  2. final depth after the event/detail branch,
  3. log-depth residual ``log(final) - log(coarse)``.

If the residual has high patch-boundary energy, the detail branch is creating or
amplifying the grid. If coarse is already high and the residual is low, the final
artifact is mostly inherited from the coarse prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset  # noqa: E402
from exp_test.visualize_normal_error_event_corr import (  # noqa: E402
    label_panel,
    make_grid,
    select_scene_samples,
    tensor_image_to_uint8,
)


EPS = 1e-6


def stack_output_field(model_output, key: str) -> Optional[torch.Tensor]:
    values = []
    for result in model_output.ress:
        if key not in result:
            return None
        value = result[key]
        if value.ndim >= 4 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        values.append(value)
    return torch.stack(values, dim=1)


def safe_load_state_dict(model: torch.nn.Module, checkpoint_path: str) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state = fe.unwrap_state_dict(ckpt)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    skipped_event_detail_examples = []
    loaded_event_detail = 0
    skipped_event_detail = 0
    for key, value in raw_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
            if key.startswith("event_detail_refiner."):
                loaded_event_detail += 1
        else:
            skipped.append(key)
            if key.startswith("event_detail_refiner."):
                skipped_event_detail += 1
                if len(skipped_event_detail_examples) < 12:
                    model_shape = tuple(model_state[key].shape) if key in model_state else None
                    skipped_event_detail_examples.append(
                        {
                            "key": key,
                            "checkpoint_shape": tuple(value.shape),
                            "model_shape": model_shape,
                        }
                    )
    msg = model.load_state_dict(compatible, strict=False)
    print(f"Loaded compatible checkpoint tensors from {checkpoint_path}: {len(compatible)}")
    print(f"Missing keys: {len(msg.missing_keys)}, unexpected keys: {len(msg.unexpected_keys)}, skipped shape keys: {len(skipped)}")
    if skipped_event_detail > 0:
        print(
            "WARNING: event_detail_refiner checkpoint tensors were skipped. "
            "The diagnostic residual may be zero because the refiner was reinitialized. "
            "Check --event-hidden-dim, --event-num-bins, and model variant."
        )
        for example in skipped_event_detail_examples:
            print(
                "  skipped",
                example["key"],
                "ckpt=",
                example["checkpoint_shape"],
                "model=",
                example["model_shape"],
            )
    if skipped[:8]:
        print("Skipped examples:", ", ".join(skipped[:8]))
    return {
        "loaded": len(compatible),
        "skipped": len(skipped),
        "loaded_event_detail": loaded_event_detail,
        "skipped_event_detail": skipped_event_detail,
        "skipped_event_detail_examples": skipped_event_detail_examples,
    }


def _cfg_section(cfg: object, name: str) -> Dict:
    if isinstance(cfg, dict):
        value = cfg.get(name, {})
        return value if isinstance(value, dict) else {}
    return {}


def apply_checkpoint_config(args) -> Dict:
    if not args.use_checkpoint_config:
        return {}
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("cfg"), dict):
        return {}

    ckpt_cfg = ckpt["cfg"]
    model_cfg = _cfg_section(ckpt_cfg, "model")
    data_cfg = _cfg_section(ckpt_cfg, "data")
    applied = {}
    mapping = {
        "model_variant": "variant",
        "img_size": "img_size",
        "patch_size": "patch_size",
        "embed_dim": "embed_dim",
        "event_hidden_dim": "event_hidden_dim",
        "event_num_bins": "event_num_bins",
        "event_count_cmax": "event_count_cmax",
        "event_fusion_scale": "event_fusion_scale",
        "event_gate_downsample": "event_gate_downsample",
        "event_gate_smooth_kernel": "event_gate_smooth_kernel",
        "event_reliability_floor": "event_reliability_floor",
        "event_reliability_init_bias": "event_reliability_init_bias",
        "head_frames_chunk_size": "head_frames_chunk_size",
        "refiner_hidden_dim": "refiner_hidden_dim",
        "refiner_num_blocks": "refiner_num_blocks",
        "refiner_residual_scale": "refiner_residual_scale",
        "refiner_refine_points": "refiner_refine_points",
    }
    for arg_name, cfg_name in mapping.items():
        if cfg_name in model_cfg:
            setattr(args, arg_name, model_cfg[cfg_name])
            applied[arg_name] = model_cfg[cfg_name]
    if "event_resize_bins" in data_cfg:
        args.event_resize_bins = int(data_cfg["event_resize_bins"])
        applied["event_resize_bins"] = args.event_resize_bins
    print("Applied checkpoint model config:", applied)
    return applied


def build_model(args, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, int]]:
    cfg = SimpleNamespace(
        data=SimpleNamespace(event_resize_bins=args.event_resize_bins),
        model=SimpleNamespace(
            variant=args.model_variant,
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            event_hidden_dim=args.event_hidden_dim,
            event_num_bins=args.event_num_bins,
            event_count_cmax=args.event_count_cmax,
            event_fusion_scale=args.event_fusion_scale,
            event_gate_downsample=args.event_gate_downsample,
            event_gate_smooth_kernel=args.event_gate_smooth_kernel,
            event_reliability_floor=args.event_reliability_floor,
            event_reliability_init_bias=args.event_reliability_init_bias,
            proposal_depth_lowpass=args.proposal_depth_lowpass,
            event_proposal_weight=args.event_proposal_weight,
            head_frames_chunk_size=args.head_frames_chunk_size,
            refiner_hidden_dim=args.refiner_hidden_dim,
            refiner_num_blocks=args.refiner_num_blocks,
            refiner_residual_scale=args.refiner_residual_scale,
            refiner_refine_points=args.refiner_refine_points,
            refiner_use_checkpoint=False,
            exposure_match_dim=8,
            exposure_agreement_floor=0.25,
            exposure_forward_batch_chunk=1,
        ),
    )
    model = fe.build_event_model(cfg).to(device)
    load_info = safe_load_state_dict(model, args.checkpoint)
    model.eval()
    return model, load_info


def build_dataset(args):
    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=args.scene_names if args.scene_names else None,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.active_scene_count,
        split=args.split,
        test_frame_count=args.test_frame_count,
        ldr_event_id=args.ldr_event_id,
        event_spatial_transform=args.event_spatial_transform,
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_resize_bins,
        return_normal_gt=False,
        return_debug_event_fields=False,
    )
    dataset, scene_counts = select_scene_samples(dataset, args.samples_per_scene)
    print(f"Selected {len(dataset)} samples, scene_counts={scene_counts}")
    return dataset


def move_views_to_device(views: List[Dict], device: torch.device) -> List[Dict]:
    moved = []
    for view in views:
        out = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                out[key] = value.to(device, non_blocking=True)
            elif isinstance(value, list):
                out[key] = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in value]
            else:
                out[key] = value
        moved.append(out)
    return moved


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(value)
    if mask.sum() <= 0:
        return value.new_tensor(float("nan"))
    return value[mask].mean()


def _phase_cv(edge: torch.Tensor, valid: torch.Tensor, phase_index: torch.Tensor, patch_size: int) -> torch.Tensor:
    means = []
    for phase in range(patch_size):
        phase_mask = valid & (phase_index == phase)
        if phase_mask.sum() > 0:
            means.append(edge[phase_mask].mean())
    if len(means) < 2:
        return edge.new_tensor(float("nan"))
    values = torch.stack(means)
    return values.std(unbiased=False) / values.mean().abs().clamp_min(EPS)


def scalar_grid_metrics(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    patch_size: int,
    prefix: str,
) -> Dict[str, float]:
    """Grid metrics for [B,S,H,W] scalar maps."""
    value = value.float()
    mask = mask.bool() & torch.isfinite(value)
    _, _, height, width = value.shape
    metrics: Dict[str, float] = {}

    dx = (value[..., :, 1:] - value[..., :, :-1]).abs()
    mx = mask[..., :, 1:] & mask[..., :, :-1]
    dy = (value[..., 1:, :] - value[..., :-1, :]).abs()
    my = mask[..., 1:, :] & mask[..., :-1, :]

    col = torch.arange(1, width, device=value.device).view(1, 1, 1, width - 1)
    row = torch.arange(1, height, device=value.device).view(1, 1, height - 1, 1)
    bx = (col % patch_size) == 0
    by = (row % patch_size) == 0

    boundary_x = _safe_mean(dx, mx & bx)
    boundary_y = _safe_mean(dy, my & by)
    interior_x = _safe_mean(dx, mx & ~bx)
    interior_y = _safe_mean(dy, my & ~by)
    all_x = _safe_mean(dx, mx)
    all_y = _safe_mean(dy, my)
    boundary = torch.nanmean(torch.stack([boundary_x, boundary_y]))
    interior = torch.nanmean(torch.stack([interior_x, interior_y]))
    all_edge = torch.nanmean(torch.stack([all_x, all_y]))

    px = col.expand_as(dx)
    py = row.expand_as(dy)
    phase_cv = torch.nanmean(
        torch.stack(
            [
                _phase_cv(dx, mx, px, patch_size),
                _phase_cv(dy, my, py, patch_size),
            ]
        )
    )

    metrics[f"{prefix}_edge_mean"] = float(all_edge.detach().cpu())
    metrics[f"{prefix}_boundary_mean"] = float(boundary.detach().cpu())
    metrics[f"{prefix}_interior_mean"] = float(interior.detach().cpu())
    metrics[f"{prefix}_boundary_over_interior"] = float((boundary / interior.clamp_min(EPS)).detach().cpu())
    metrics[f"{prefix}_phase_cv"] = float(phase_cv.detach().cpu())
    return metrics


def normal_grid_metrics(
    normal: torch.Tensor,
    mask: torch.Tensor,
    *,
    patch_size: int,
    prefix: str,
) -> Dict[str, float]:
    normal = torch.nn.functional.normalize(normal.float(), dim=-1, eps=EPS)
    mask = mask.bool() & torch.isfinite(normal).all(dim=-1)
    _, _, height, width, _ = normal.shape

    dx = (normal[..., :, 1:, :] - normal[..., :, :-1, :]).norm(dim=-1)
    mx = mask[..., :, 1:] & mask[..., :, :-1]
    dy = (normal[..., 1:, :, :] - normal[..., :-1, :, :]).norm(dim=-1)
    my = mask[..., 1:, :] & mask[..., :-1, :]

    col = torch.arange(1, width, device=normal.device).view(1, 1, 1, width - 1)
    row = torch.arange(1, height, device=normal.device).view(1, 1, height - 1, 1)
    bx = (col % patch_size) == 0
    by = (row % patch_size) == 0

    boundary = torch.nanmean(torch.stack([_safe_mean(dx, mx & bx), _safe_mean(dy, my & by)]))
    interior = torch.nanmean(torch.stack([_safe_mean(dx, mx & ~bx), _safe_mean(dy, my & ~by)]))
    all_edge = torch.nanmean(torch.stack([_safe_mean(dx, mx), _safe_mean(dy, my)]))

    px = col.expand_as(dx)
    py = row.expand_as(dy)
    phase_cv = torch.nanmean(
        torch.stack(
            [
                _phase_cv(dx, mx, px, patch_size),
                _phase_cv(dy, my, py, patch_size),
            ]
        )
    )

    return {
        f"{prefix}_edge_mean": float(all_edge.detach().cpu()),
        f"{prefix}_boundary_mean": float(boundary.detach().cpu()),
        f"{prefix}_interior_mean": float(interior.detach().cpu()),
        f"{prefix}_boundary_over_interior": float((boundary / interior.clamp_min(EPS)).detach().cpu()),
        f"{prefix}_phase_cv": float(phase_cv.detach().cpu()),
    }


def signed_to_rgb(value: torch.Tensor, mask: Optional[torch.Tensor] = None, percentile: float = 99.0) -> np.ndarray:
    arr = value.detach().float().cpu().numpy()
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask.detach().cpu().numpy().astype(bool)
    if valid.any():
        scale = np.percentile(np.abs(arr[valid]), percentile)
    else:
        scale = 1.0
    scale = max(float(scale), EPS)
    x = np.clip(arr / scale, -1.0, 1.0)
    rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(x, 0.0, 1.0)
    rgb[..., 2] = np.clip(-x, 0.0, 1.0)
    rgb[..., 1] = 0.25 * (1.0 - np.abs(x))
    if mask is not None:
        rgb[~mask.detach().cpu().numpy().astype(bool)] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def gray_to_rgb(value: torch.Tensor, mask: Optional[torch.Tensor] = None, percentile: float = 99.0) -> np.ndarray:
    arr = value.detach().float().cpu().numpy()
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask.detach().cpu().numpy().astype(bool)
    if valid.any():
        lo, hi = np.percentile(arr[valid], [2.0, percentile])
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    x = np.clip((arr - lo) / max(float(hi - lo), EPS), 0.0, 1.0)
    rgb = np.repeat(x[..., None], 3, axis=-1)
    if mask is not None:
        rgb[~mask.detach().cpu().numpy().astype(bool)] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def get_output_map(output, key: str, batch_idx: int, view_idx: int) -> Optional[torch.Tensor]:
    stacked = stack_output_field(output, key)
    if stacked is None:
        return None
    value = stacked[batch_idx, view_idx]
    while value.ndim > 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    return value


def save_visuals(
    *,
    out_path: Path,
    views: List[Dict],
    output,
    depth_coarse: torch.Tensor,
    depth_final: torch.Tensor,
    depth_gt: torch.Tensor,
    residual_log: torch.Tensor,
    coarse_normals: torch.Tensor,
    final_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_idx: int,
) -> None:
    rows: List[List[Image.Image]] = []
    num_views = depth_final.shape[1]
    for view_idx in range(num_views):
        mask = valid_mask[sample_idx, view_idx]
        row = [
            label_panel(f"view{view_idx} rgb", tensor_image_to_uint8(views[view_idx]["img"][sample_idx])),
            label_panel("coarse_depth", fe.depth_to_uint8(depth_coarse[sample_idx, view_idx], mask)),
            label_panel("final_depth", fe.depth_to_uint8(depth_final[sample_idx, view_idx], mask)),
            label_panel("gt_depth", fe.depth_to_uint8(depth_gt[sample_idx, view_idx], mask)),
            label_panel("log(final/coarse)", signed_to_rgb(residual_log[sample_idx, view_idx], mask)),
            label_panel("coarse_normal", fe.normal_to_uint8(coarse_normals[sample_idx, view_idx], mask)),
            label_panel("final_normal", fe.normal_to_uint8(final_normals[sample_idx, view_idx], mask)),
            label_panel("gt_normal", fe.normal_to_uint8(gt_normals[sample_idx, view_idx], mask)),
        ]
        reliability = get_output_map(output, "event_reliability", sample_idx, view_idx)
        gate = get_output_map(output, "event_gate", sample_idx, view_idx)
        if reliability is not None:
            row.append(label_panel("event_reliability", gray_to_rgb(reliability, mask=None)))
        if gate is not None:
            row.append(label_panel("event_gate/filter", gray_to_rgb(gate, mask=None)))
        rows.append(row)
    make_grid(rows).save(out_path)


def mean_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for record in records for key in record})
    out = {}
    for key in keys:
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
        finite = np.isfinite(values)
        out[key] = float(values[finite].mean()) if finite.any() else float("nan")
    return out


def add_ratio_metrics(metrics: Dict[str, float]) -> None:
    def ratio(num: str, den: str) -> float:
        n = metrics.get(num, float("nan"))
        d = metrics.get(den, float("nan"))
        return float(n / max(d, EPS)) if np.isfinite(n) and np.isfinite(d) else float("nan")

    metrics["final_vs_coarse_depth_boundary_ratio"] = ratio(
        "final_log_depth_boundary_mean", "coarse_log_depth_boundary_mean"
    )
    metrics["residual_vs_final_depth_boundary_ratio"] = ratio(
        "residual_log_depth_boundary_mean", "final_log_depth_boundary_mean"
    )
    metrics["final_vs_coarse_normal_boundary_ratio"] = ratio(
        "final_normal_boundary_mean", "coarse_normal_boundary_mean"
    )


def add_event_input_metrics(record: Dict[str, float], views: List[Dict]) -> None:
    voxels = []
    for view in views:
        voxel = view.get("event_voxel")
        if torch.is_tensor(voxel):
            voxels.append(voxel.detach().float())
    if not voxels:
        record["event_voxel_present"] = 0.0
        record["event_voxel_abs_mean"] = 0.0
        record["event_voxel_nonzero_ratio"] = 0.0
        return
    stacked = torch.stack(voxels, dim=1)
    record["event_voxel_present"] = 1.0
    record["event_voxel_abs_mean"] = float(stacked.abs().mean().detach().cpu())
    record["event_voxel_nonzero_ratio"] = float((stacked.abs() > 0).float().mean().detach().cpu())


def add_residual_metrics(
    record: Dict[str, float],
    output,
    depth_final: torch.Tensor,
    depth_coarse: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    diff = (depth_final - depth_coarse).abs()
    valid = valid_mask.bool() & torch.isfinite(diff)
    record["final_minus_coarse_depth_abs_mean"] = float(_safe_mean(diff, valid).detach().cpu())
    depth_residual = stack_output_field(output, "depth_residual")
    if depth_residual is None:
        record["output_depth_residual_abs_mean"] = float("nan")
        return
    depth_residual = depth_residual.squeeze(-1).float()
    record["output_depth_residual_abs_mean"] = float(_safe_mean(depth_residual.abs(), valid).detach().cpu())


def interpretation(metrics: Dict[str, float]) -> str:
    coarse = metrics.get("coarse_normal_boundary_over_interior", float("nan"))
    final = metrics.get("final_normal_boundary_over_interior", float("nan"))
    residual = metrics.get("residual_log_depth_boundary_over_interior", float("nan"))
    final_vs_coarse = metrics.get("final_vs_coarse_normal_boundary_ratio", float("nan"))

    if not all(np.isfinite(v) for v in (coarse, final, residual, final_vs_coarse)):
        return "metrics_not_finite"
    if residual > 1.15 and final_vs_coarse > 1.05:
        return "detail_branch_likely_creates_or_amplifies_grid"
    if coarse > 1.15 and residual <= 1.05:
        return "grid_likely_inherited_from_coarse_prediction"
    if coarse > 1.10 and final_vs_coarse < 0.98:
        return "detail_branch_reduces_but_does_not_remove_coarse_grid"
    return "mixed_or_weak_grid_signature"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    applied_checkpoint_config = apply_checkpoint_config(args)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=event_multiview_collate,
    )
    model, load_info = build_model(args, device)

    records: List[Dict[str, float]] = []
    for sample_index, views in enumerate(loader):
        if sample_index >= args.max_samples:
            break
        views = move_views_to_device(views, device)
        views = fe.maybe_denormalize_views(views)

        output = model(views)
        depth_final = stack_output_field(output, "depth").squeeze(-1).float()
        depth_coarse = stack_output_field(output, "depth_coarse")
        if depth_coarse is None:
            depth_coarse = depth_final
        depth_coarse = depth_coarse.squeeze(-1).float()
        depth_gt = fe.stack_view_field(views, "depthmap").to(device=device, dtype=depth_final.dtype)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=device, dtype=depth_final.dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt)

        depth_gt_aligned, _ = fe.align_depth_scale(depth_final.detach(), depth_gt, valid_mask)
        coarse_normals = fe.depth_to_normals(depth_coarse.clamp_min(EPS), intrinsics)
        final_normals = fe.depth_to_normals(depth_final.clamp_min(EPS), intrinsics)
        gt_normals = fe.depth_to_normals(depth_gt_aligned.clamp_min(EPS), intrinsics)
        normal_mask = valid_mask.clone()
        normal_mask[..., 0, :] = False
        normal_mask[..., -1, :] = False
        normal_mask[..., :, 0] = False
        normal_mask[..., :, -1] = False

        log_coarse = torch.log(depth_coarse.clamp_min(EPS))
        log_final = torch.log(depth_final.clamp_min(EPS))
        residual_log = log_final - log_coarse

        record: Dict[str, float] = {}
        record.update(scalar_grid_metrics(log_coarse, valid_mask, patch_size=args.patch_size, prefix="coarse_log_depth"))
        record.update(scalar_grid_metrics(log_final, valid_mask, patch_size=args.patch_size, prefix="final_log_depth"))
        record.update(
            scalar_grid_metrics(residual_log, valid_mask, patch_size=args.patch_size, prefix="residual_log_depth")
        )
        record.update(normal_grid_metrics(coarse_normals, normal_mask, patch_size=args.patch_size, prefix="coarse_normal"))
        record.update(normal_grid_metrics(final_normals, normal_mask, patch_size=args.patch_size, prefix="final_normal"))
        add_event_input_metrics(record, views)
        add_residual_metrics(record, output, depth_final, depth_coarse, valid_mask)
        add_ratio_metrics(record)
        records.append(record)

        if sample_index < args.visual_samples:
            save_visuals(
                out_path=out_dir / f"grid_source_sample_{sample_index:03d}.png",
                views=views,
                output=output,
                depth_coarse=depth_coarse,
                depth_final=depth_final,
                depth_gt=depth_gt_aligned,
                residual_log=residual_log,
                coarse_normals=coarse_normals,
                final_normals=final_normals,
                gt_normals=gt_normals,
                valid_mask=valid_mask,
                sample_idx=0,
            )

    metrics = mean_metrics(records)
    add_ratio_metrics(metrics)
    summary = {
        "checkpoint": args.checkpoint,
        "model_variant": args.model_variant,
        "num_records": len(records),
        "patch_size": args.patch_size,
        "checkpoint_config_applied": applied_checkpoint_config,
        "checkpoint_load": load_info,
        "metrics_mean": metrics,
        "interpretation": interpretation(metrics),
    }
    with open(out_dir / "grid_source_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="exp_test/grid_source_diagnostics")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--event-spatial-transform", default="auto")
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=4)
    parser.add_argument("--samples-per-scene", type=int, default=1)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--visual-samples", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-mem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--model-variant", default="reliability_filter_detail")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--event-hidden-dim", type=int, default=16)
    parser.add_argument("--event-num-bins", type=int, default=10)
    parser.add_argument("--event-count-cmax", type=float, default=3.0)
    parser.add_argument("--event-fusion-scale", type=float, default=1.0)
    parser.add_argument("--event-gate-downsample", type=int, default=2)
    parser.add_argument("--event-gate-smooth-kernel", type=int, default=5)
    parser.add_argument("--event-reliability-floor", type=float, default=0.30)
    parser.add_argument("--event-reliability-init-bias", type=float, default=0.5)
    parser.add_argument("--proposal-depth-lowpass", action="store_true")
    parser.add_argument("--event-proposal-weight", type=float, default=0.0)
    parser.add_argument("--head-frames-chunk-size", type=int, default=2)
    parser.add_argument("--refiner-hidden-dim", type=int, default=16)
    parser.add_argument("--refiner-num-blocks", type=int, default=2)
    parser.add_argument("--refiner-residual-scale", type=float, default=0.03)
    parser.add_argument("--refiner-refine-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-checkpoint-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the model with the model/data config saved inside the checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
