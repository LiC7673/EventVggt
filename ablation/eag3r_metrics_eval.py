"""EAG3R-style metric evaluation for EventVGGT ablations.

The EAG3R paper reports depth metrics (Abs Rel, delta < 1.25, RMSE log) and
pose metrics (ATE, RPE trans, RPE rot).  This evaluator computes those metrics
on the same MyEventDataset test split, plus optional normal/detail diagnostics
that are useful for the paper story here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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
import finetune_no_event as nf  # noqa: E402
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset  # noqa: E402
from exp_test.visualize_normal_error_event_corr import (  # noqa: E402
    event_support_from_parts,
    event_voxel_parts,
    pearson_corr,
    ranked_event_regions,
    select_sample,
)


EPS = 1e-8


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


class DepthMetrics:
    def __init__(self):
        self.count = 0
        self.abs_rel = 0.0
        self.rmse_log_sq = 0.0
        self.delta1 = 0.0
        self.rmse_sq = 0.0
        self.mae = 0.0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, *, median_align: bool = False) -> None:
        pred = pred.float()
        gt = gt.float()
        mask = mask.bool() & torch.isfinite(pred) & torch.isfinite(gt) & (gt > EPS) & (pred > EPS)
        if median_align:
            pred = median_align_depth(pred, gt, mask)
            mask = mask & torch.isfinite(pred) & (pred > EPS)
        if not mask.any():
            return

        pred_v = pred[mask].clamp_min(EPS)
        gt_v = gt[mask].clamp_min(EPS)
        diff = pred_v - gt_v
        ratio = torch.maximum(pred_v / gt_v, gt_v / pred_v)
        log_diff = torch.log(pred_v) - torch.log(gt_v)
        n = int(pred_v.numel())
        self.count += n
        self.abs_rel += float((diff.abs() / gt_v).sum().detach().cpu())
        self.rmse_log_sq += float(log_diff.square().sum().detach().cpu())
        self.delta1 += float((ratio < 1.25).sum().detach().cpu())
        self.rmse_sq += float(diff.square().sum().detach().cpu())
        self.mae += float(diff.abs().sum().detach().cpu())

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "depth_pixels": 0,
                "abs_rel": float("nan"),
                "delta1": float("nan"),
                "rmse_log": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
            }
        count = float(self.count)
        return {
            "depth_pixels": int(self.count),
            "abs_rel": self.abs_rel / count,
            "delta1": self.delta1 / count,
            "rmse_log": math.sqrt(self.rmse_log_sq / count),
            "rmse": math.sqrt(self.rmse_sq / count),
            "mae": self.mae / count,
        }


class MeanAccumulator:
    def __init__(self):
        self.values: List[float] = []

    def update(self, value: float) -> None:
        if np.isfinite(value):
            self.values.append(float(value))

    def compute(self) -> float:
        return nanmean(self.values)


def median_align_depth(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    aligned = pred.clone()
    flat_pred = aligned.reshape(-1, *aligned.shape[-2:])
    flat_gt = gt.reshape(-1, *gt.shape[-2:])
    flat_valid = valid.reshape(-1, *valid.shape[-2:])
    for idx in range(flat_pred.shape[0]):
        mask = flat_valid[idx] & torch.isfinite(flat_pred[idx]) & (flat_pred[idx] > EPS)
        if int(mask.sum()) < 16:
            continue
        scale = flat_gt[idx][mask].median() / flat_pred[idx][mask].median().clamp_min(EPS)
        flat_pred[idx] = flat_pred[idx] * scale.clamp(1e-3, 1e3)
    return aligned


def stack_output(model_output, key: str) -> Optional[torch.Tensor]:
    ress = getattr(model_output, "ress", None)
    if not ress or not all(key in res for res in ress):
        return None
    value = torch.stack([res[key] for res in ress], dim=1)
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    return value


def move_views_to_device(views: List[Dict], device: torch.device) -> List[Dict]:
    out = []
    for view in views:
        moved = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=True)
            elif isinstance(value, list):
                moved[key] = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in value]
            else:
                moved[key] = value
        out.append(moved)
    return out


def rotation_angle_deg(rot: torch.Tensor) -> torch.Tensor:
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def pose_errors(pred_c2w: torch.Tensor, gt_c2w: torch.Tensor, *, scale_align: bool = False) -> Dict[str, float]:
    pred_c2w = fe.ensure_homogeneous(pred_c2w.float())
    gt_c2w = fe.ensure_homogeneous(gt_c2w.float())
    pred_aligned, _ = fe.align_c2w_by_first_frame(pred_c2w, gt_c2w)

    if scale_align and pred_aligned.shape[1] > 1:
        pred_delta = pred_aligned[:, :, :3, 3] - pred_aligned[:, :1, :3, 3]
        gt_delta = gt_c2w[:, :, :3, 3] - gt_c2w[:, :1, :3, 3]
        num = (pred_delta * gt_delta).sum(dim=(1, 2))
        den = pred_delta.square().sum(dim=(1, 2)).clamp_min(EPS)
        scale = (num / den).clamp(1e-3, 1e3).view(-1, 1, 1)
        pred_aligned = pred_aligned.clone()
        pred_aligned[:, :, :3, 3] = gt_c2w[:, :1, :3, 3] + scale * (
            pred_aligned[:, :, :3, 3] - pred_aligned[:, :1, :3, 3]
        )

    trans_error = (pred_aligned[:, :, :3, 3] - gt_c2w[:, :, :3, 3]).norm(dim=-1)
    ate = torch.sqrt(trans_error.square().mean()).item()

    if pred_aligned.shape[1] <= 1:
        return {"ate": float(ate), "rpe_trans": float("nan"), "rpe_rot_deg": float("nan")}

    pred_rel = torch.linalg.inv(pred_aligned[:, :-1]) @ pred_aligned[:, 1:]
    gt_rel = torch.linalg.inv(gt_c2w[:, :-1]) @ gt_c2w[:, 1:]
    rel_err = torch.linalg.inv(gt_rel) @ pred_rel
    rpe_trans = rel_err[..., :3, 3].norm(dim=-1).mean().item()
    rpe_rot = rotation_angle_deg(rel_err[..., :3, :3]).mean().item()
    return {"ate": float(ate), "rpe_trans": float(rpe_trans), "rpe_rot_deg": float(rpe_rot)}


def normal_error_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    views: List[Dict],
    args,
) -> Dict[str, float]:
    pred_normals = fe.depth_to_normals(pred_depth, intrinsics)
    gt_normals = fe.depth_to_normals(gt_depth, intrinsics)
    normal_mask = valid_mask.clone()
    normal_mask[..., 0, :] = False
    normal_mask[..., -1, :] = False
    normal_mask[..., :, 0] = False
    normal_mask[..., :, -1] = False
    cos = (F.normalize(pred_normals, dim=-1, eps=1e-6) * F.normalize(gt_normals, dim=-1, eps=1e-6)).sum(dim=-1)
    err = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
    valid_err = err[normal_mask]
    result = {"normal_error_deg": float(valid_err.mean().detach().cpu()) if valid_err.numel() else float("nan")}

    event_corr = []
    high_err = []
    low_err = []
    batch = pred_depth.shape[0]
    seq_len = pred_depth.shape[1]
    for b in range(batch):
        for s in range(seq_len):
            sample_view = select_sample(views[s], b)
            pos, neg = event_voxel_parts(sample_view, args.event_resize_bins)
            support = event_support_from_parts(pos, neg, mode=args.event_support_mode)
            mask_np = normal_mask[b, s].detach().cpu().numpy().astype(bool)
            err_np = err[b, s].detach().cpu().numpy()
            high_mask, low_mask = ranked_event_regions(
                support,
                mask_np,
                high_fraction=args.event_high_fraction,
                low_fraction=args.event_low_fraction,
            )
            event_corr.append(pearson_corr(support, err_np, mask_np))
            if high_mask.any():
                high_err.append(float(err_np[high_mask].mean()))
            if low_mask.any():
                low_err.append(float(err_np[low_mask].mean()))

    result.update(
        {
            "corr_event_normal_error": nanmean(event_corr),
            "high_event_normal_error_deg": nanmean(high_err),
            "low_event_normal_error_deg": nanmean(low_err),
            "high_minus_low_normal_error_deg": nanmean(high_err) - nanmean(low_err),
        }
    )
    return result


def cfg_from_checkpoint(ckpt, fallback_config: Optional[str]):
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        return OmegaConf.create(ckpt["cfg"])
    config_path = fallback_config or str(ROOT_DIR / "config" / "finetune_event.yaml")
    return OmegaConf.load(config_path)


def apply_spec_overrides(cfg, spec: Dict) -> None:
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "data", "loss", "train"):
        if hasattr(cfg, branch_name):
            OmegaConf.set_struct(getattr(cfg, branch_name), False)

    model_key_map = {
        "model_variant": "variant",
        "event_hidden_dim": "event_hidden_dim",
        "event_num_bins": "event_num_bins",
        "refiner_residual_scale": "refiner_residual_scale",
        "event_delta_highpass_kernel": "event_delta_highpass_kernel",
        "event_delta_patch_zero_mean": "event_delta_patch_zero_mean",
        "event_delta_patch_size": "event_delta_patch_size",
        "event_delta_abs_limit": "event_delta_abs_limit",
        "event_reliability_gate_enabled": "event_reliability_gate_enabled",
        "event_reliability_gate_floor": "event_reliability_gate_floor",
        "event_reliability_init_bias": "event_reliability_init_bias",
    }
    for source, target in model_key_map.items():
        if source in spec and spec[source] is not None:
            setattr(cfg.model, target, spec[source])


def build_dataset(cfg, args):
    ldr_event_id = args.ldr_event_id
    if ldr_event_id is None:
        ldr_event_id = getattr(cfg.data, "eval_ldr_event_id", None)
    if ldr_event_id is None:
        ldr_event_id = getattr(cfg.data, "ldr_event_id", "auto")
    if str(ldr_event_id).lower() in {"random", "any", "all", "multi", "*"}:
        ldr_event_id = "ev_5"

    return get_combined_dataset(
        root=str(getattr(args, "root", None) or cfg.data.root),
        num_views=int(args.num_views or cfg.data.num_views),
        resolution=tuple(args.resolution or cfg.data.resolution),
        fps=int(getattr(cfg.data, "fps", 120)),
        seed=int(getattr(cfg, "seed", 0)),
        scene_names=args.scene_names if args.scene_names else (cfg.data.scene_names if getattr(cfg.data, "scene_names", None) else None),
        initial_scene_idx=int(args.initial_scene_idx if args.initial_scene_idx is not None else getattr(cfg.data, "initial_scene_idx", 0)),
        active_scene_count=int(args.active_scene_count if args.active_scene_count is not None else getattr(cfg.data, "active_scene_count", 3)),
        split=args.split,
        test_frame_count=int(args.test_frame_count if args.test_frame_count is not None else getattr(cfg.data, "test_frame_count", 10)),
        ldr_event_id=str(ldr_event_id),
        event_spatial_transform=str(args.event_spatial_transform or getattr(cfg.data, "event_spatial_transform", "auto")),
        event_resize_method=str(args.event_resize_method or getattr(cfg.data, "event_resize_method", "voxel_antialias")),
        event_resize_bins=int(args.event_resize_bins or getattr(cfg.data, "event_resize_bins", 10)),
        return_normal_gt=True,
        return_debug_event_fields=False,
    )


def build_model(family: str, cfg, ckpt, device: torch.device):
    if family == "rgb":
        model = nf.build_rgb_model(cfg)
    else:
        model = fe.build_event_model(cfg)
    state = strip_module_prefix(fe.unwrap_state_dict(ckpt))
    msg = model.load_state_dict(state, strict=False)
    print(f"[load] family={family}, variant={getattr(cfg.model, 'variant', 'base')}, msg={msg}")
    model.to(device)
    model.eval()
    return model


def evaluate_experiment(spec: Dict, args, out_dir: Path) -> Dict[str, float]:
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT_DIR / checkpoint
    if not checkpoint.exists():
        if args.skip_missing:
            print(f"[skip] {spec.get('name', checkpoint.stem)} missing: {checkpoint}")
            return {}
        raise FileNotFoundError(checkpoint)

    ckpt = torch_load(checkpoint)
    cfg = cfg_from_checkpoint(ckpt, spec.get("config"))
    apply_spec_overrides(cfg, spec)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    family = str(spec.get("family", "event")).lower()
    model = build_model(family, cfg, ckpt, device)
    dataset = build_dataset(cfg, args)
    active_scenes = dataset.get_active_scenes() if hasattr(dataset, "get_active_scenes") else []
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    print(
        f"[eval] {spec['name']}: samples={len(dataset)}, batches={len(loader)}, "
        f"split={args.split}, scenes={active_scenes}, checkpoint={checkpoint}"
    )

    depth_metrics = DepthMetrics()
    depth_metrics_aligned = DepthMetrics()
    normal_acc = {key: MeanAccumulator() for key in (
        "normal_error_deg",
        "corr_event_normal_error",
        "high_event_normal_error_deg",
        "low_event_normal_error_deg",
        "high_minus_low_normal_error_deg",
    )}
    pose_acc = {key: MeanAccumulator() for key in ("ate", "rpe_trans", "rpe_rot_deg")}
    reliability_acc = MeanAccumulator()

    start = time.time()
    for batch_idx, views in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        views = fe.maybe_denormalize_views(views)
        views = move_views_to_device(views, device)
        with torch.no_grad():
            use_amp = args.amp != "none" and device.type == "cuda"
            amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(views)

        depth_pred = stack_output(output, "depth")
        pose_pred = stack_output(output, "camera_pose")
        reliability = stack_output(output, "event_reliability")
        if depth_pred is None:
            raise RuntimeError(f"{spec['name']} did not return depth")

        depth_gt = fe.stack_view_field(views, "depthmap").to(device=device, dtype=depth_pred.dtype)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=device, dtype=depth_pred.dtype)
        gt_c2w = fe.stack_view_field(views, "camera_pose").to(device=device, dtype=torch.float32)
        valid = fe.build_valid_mask(
            views,
            depth_gt,
            depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
            depth_max=getattr(cfg.loss, "depth_max", None),
        )

        depth_metrics.update(depth_pred, depth_gt, valid, median_align=False)
        depth_metrics_aligned.update(depth_pred, depth_gt, valid, median_align=True)

        normal_values = normal_error_metrics(depth_pred, depth_gt, intrinsics, valid, views, args)
        for key, value in normal_values.items():
            normal_acc[key].update(value)

        if pose_pred is not None:
            height, width = depth_pred.shape[-2:]
            pred_c2w, _ = fe.pose_encoding_to_c2w(pose_pred.float(), image_size_hw=(height, width))
            pose_values = pose_errors(pred_c2w, gt_c2w, scale_align=args.pose_scale_align)
            for key, value in pose_values.items():
                pose_acc[key].update(value)

        if reliability is not None:
            reliability_acc.update(float(reliability.detach().float().mean().cpu()))

        if (batch_idx + 1) % args.print_freq == 0:
            print(f"[eval] {spec['name']}: {batch_idx + 1}/{len(loader)} elapsed={time.time() - start:.1f}s")

    raw_depth = depth_metrics.compute()
    aligned_depth = {f"median_{k}": v for k, v in depth_metrics_aligned.compute().items() if k != "depth_pixels"}
    row = {
        "name": spec["name"],
        "family": family,
        "variant": str(getattr(cfg.model, "variant", "base")),
        "checkpoint": str(checkpoint),
        "split": args.split,
        "initial_scene_idx": int(args.initial_scene_idx if args.initial_scene_idx is not None else getattr(cfg.data, "initial_scene_idx", 0)),
        "active_scene_count": int(args.active_scene_count if args.active_scene_count is not None else getattr(cfg.data, "active_scene_count", 3)),
        "active_scenes": ";".join(active_scenes),
        **raw_depth,
        **aligned_depth,
        **{key: acc.compute() for key, acc in pose_acc.items()},
        **{key: acc.compute() for key, acc in normal_acc.items()},
        "event_reliability_mean": reliability_acc.compute(),
        "num_samples": len(dataset),
        "evaluated_batches": len(loader) if args.max_batches is None else min(len(loader), args.max_batches),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def load_specs(args) -> List[Dict]:
    if args.checkpoint:
        return [
            {
                "name": args.name,
                "family": args.family,
                "checkpoint": args.checkpoint,
                "config": args.config,
                "model_variant": args.model_variant,
            }
        ]
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT_DIR / manifest
    with manifest.open("r", encoding="utf-8") as handle:
        specs = json.load(handle)
    if not isinstance(specs, list):
        raise ValueError("Manifest must be a list of experiment specs")
    return specs


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "name",
        "family",
        "variant",
        "abs_rel",
        "delta1",
        "rmse_log",
        "ate",
        "rpe_trans",
        "rpe_rot_deg",
        "normal_error_deg",
        "corr_event_normal_error",
        "high_event_normal_error_deg",
        "low_event_normal_error_deg",
        "high_minus_low_normal_error_deg",
        "event_reliability_mean",
        "rmse",
        "mae",
        "median_abs_rel",
        "median_delta1",
        "median_rmse_log",
        "depth_pixels",
        "num_samples",
        "evaluated_batches",
        "checkpoint",
        "split",
        "initial_scene_idx",
        "active_scene_count",
        "active_scenes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved:\n  {csv_path}\n  {json_path}")
    print("\nEAG3R-style summary:")
    for row in rows:
        print(
            f"{row['name']:30s} "
            f"AbsRel={row.get('abs_rel', float('nan')):.5f} "
            f"d1={row.get('delta1', float('nan')):.4f} "
            f"RMSElog={row.get('rmse_log', float('nan')):.5f} "
            f"ATE={row.get('ate', float('nan')):.5f} "
            f"RPE_t={row.get('rpe_trans', float('nan')):.5f} "
            f"RPE_r={row.get('rpe_rot_deg', float('nan')):.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ablations with EAG3R-style depth/pose metrics")
    parser.add_argument("--manifest", default="ablation/eag3r_eval_manifest.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--name", default="custom")
    parser.add_argument("--family", default="event", choices=["event", "rgb"])
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=None)
    parser.add_argument("--ldr-event-id", default=None)
    parser.add_argument("--event-spatial-transform", default=None)
    parser.add_argument("--event-resize-method", default=None)
    parser.add_argument("--event-resize-bins", type=int, default=None)
    parser.add_argument("--event-support-mode", default="temporal_polarity")
    parser.add_argument("--event-high-fraction", type=float, default=0.2)
    parser.add_argument("--event-low-fraction", type=float, default=0.2)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=None)
    parser.add_argument("--active-scene-count", type=int, default=None)
    parser.add_argument("--test-frame-count", type=int, default=None)
    parser.add_argument("--pose-scale-align", action="store_true")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = str(ROOT_DIR / "ablation" / "results" / f"eag3r_metrics_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(args.out_dir)
    rows = []
    for spec in load_specs(args):
        row = evaluate_experiment(spec, args, out_dir)
        if row:
            rows.append(row)
    if not rows:
        raise RuntimeError("No rows were evaluated. Check checkpoint paths or remove --skip-missing.")
    write_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
