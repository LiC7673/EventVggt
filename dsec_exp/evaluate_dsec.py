"""Evaluate a checkpoint on every held-out DSEC test sequence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

import ablation.eag3r_metrics_eval as metric_utils
import finetune_event as fe
import finetune_no_event as nf
from dsec_exp.common import build_dsec_loader


def _state_dict(checkpoint):
    state = fe.unwrap_state_dict(checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _build_model(checkpoint, cfg, approach, device):
    if approach == "auto":
        approach = str(getattr(cfg, "approach", "full_img_reliability"))
    if approach == "rgb":
        cfg.model.variant = "base"
        model = nf.build_rgb_model(cfg)
    else:
        model = fe.build_event_model(cfg)
    state = _state_dict(checkpoint)
    model_state = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in model_state and model_state[key].shape == value.shape}
    if not compatible:
        raise RuntimeError("No checkpoint tensors match the requested DSEC model")
    core_keys = [
        key for key in model_state
        if key.startswith("aggregator.") or key.startswith("depth_head.") or key.startswith("point_head.")
    ]
    loaded_core = sum(key in compatible for key in core_keys)
    core_ratio = loaded_core / max(len(core_keys), 1)
    if core_ratio < 0.95:
        raise RuntimeError(
            f"Checkpoint/model mismatch: only {loaded_core}/{len(core_keys)} core RGB/depth tensors match "
            f"({100.0 * core_ratio:.1f}%). Check --approach and checkpoint family."
        )
    result = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)}/{len(model_state)} model tensors; "
        f"core coverage={100.0 * core_ratio:.1f}%; {result}"
    )
    return model.to(device).eval(), approach


def _event_rgb(voxel):
    value = voxel.detach().float().cpu().numpy()
    bins = value.shape[0] // 2
    pos = np.log1p(np.maximum(value[:bins], 0.0).sum(0))
    neg = np.log1p(np.maximum(value[bins : 2 * bins], 0.0).sum(0))
    scale = max(float(np.percentile(np.concatenate([pos.ravel(), neg.ravel()]), 99.5)), 1e-6)
    out = np.zeros((*pos.shape, 3), dtype=np.float32)
    out[..., 0] = np.clip(pos / scale, 0, 1)
    out[..., 2] = np.clip(neg / scale, 0, 1)
    return (out * 255).astype(np.uint8)


def _normal_stats(pred, gt, intrinsics, valid):
    pred_n = fe.depth_to_normals(pred, intrinsics)
    gt_n = fe.depth_to_normals(gt, intrinsics)
    mask = fe.normal_stencil_valid_mask(valid, pred, eps=0.1)
    cosine = (F.normalize(pred_n, dim=-1, eps=1e-6) * F.normalize(gt_n, dim=-1, eps=1e-6)).sum(-1)
    error = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))[mask]
    if not error.numel():
        return {key: float("nan") for key in ("normal_mean_deg", "normal_rmse_deg", "normal_11_25", "normal_22_5", "normal_30")}
    return {
        "normal_mean_deg": float(error.mean().cpu()),
        "normal_rmse_deg": float(error.square().mean().sqrt().cpu()),
        "normal_11_25": float((error < 11.25).float().mean().cpu()),
        "normal_22_5": float((error < 22.5).float().mean().cpu()),
        "normal_30": float((error < 30.0).float().mean().cpu()),
    }


def _save_visual(path, views, output, batch_index, max_views):
    pred = metric_utils.stack_output(output, "depth")
    gt = fe.stack_view_field(views, "depthmap").to(pred)
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(pred)
    valid = fe.build_valid_mask(views, gt, depth_min=0.1, depth_max=80.0)
    pred_n = fe.depth_to_normals(pred, intrinsics)
    gt_n = fe.depth_to_normals(gt, intrinsics)
    reliability = metric_utils.stack_output(output, "event_reliability")
    path.mkdir(parents=True, exist_ok=True)
    for view_index in range(min(len(views), max_views)):
        mask = valid[0, view_index]
        panels = [
            fe.make_labeled_panel("rgb", fe.tensor_rgb_to_uint8(views[view_index]["img"][0], mask)),
            fe.make_labeled_panel("event", _event_rgb(views[view_index]["event_voxel"][0])),
            fe.make_labeled_panel("gt_depth", fe.depth_to_uint8(gt[0, view_index], mask)),
            fe.make_labeled_panel("pred_depth", fe.depth_to_uint8(pred[0, view_index], mask)),
            fe.make_labeled_panel("pred_normal", fe.normal_to_uint8(pred_n[0, view_index], mask)),
            fe.make_labeled_panel("gt_normal", fe.normal_to_uint8(gt_n[0, view_index], mask)),
        ]
        if reliability is not None:
            panels.insert(4, fe.make_labeled_panel("event_reliability", fe.depth_to_uint8(reliability[0, view_index], mask)))
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
        offset = 0
        for panel in panels:
            canvas.paste(panel, (offset, 0))
            offset += panel.width
        canvas.save(path / f"batch_{batch_index:04d}_view_{view_index:02d}.png")


def _scene_loader(cfg, args, scene):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.root = args.root
    cfg.data.sequence_names = [scene]
    cfg.data.num_views = args.num_views
    cfg.data.test_clip_stride = args.clip_stride
    cfg.data.allow_unaligned_rgb = args.allow_unaligned_rgb
    cfg.batch_size = 1
    cfg.num_workers = args.num_workers
    cfg.pin_mem = False
    return build_dsec_loader(cfg, "test", rgb_only=False)


def _evaluate_scene(model, cfg, args, scene, device):
    loader = _scene_loader(cfg, args, scene)
    raw = metric_utils.DepthMetrics()
    aligned = metric_utils.DepthMetrics()
    normal_values = []
    reliability_values = []
    windows = 0
    for batch_index, views in enumerate(loader):
        if args.max_windows is not None and batch_index >= args.max_windows:
            break
        views = metric_utils.move_views_to_device(fe.maybe_denormalize_views(views), device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(views)
        pred = metric_utils.stack_output(output, "depth")
        if pred is None:
            raise RuntimeError("Model did not return depth")
        gt = fe.stack_view_field(views, "depthmap").to(pred)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(pred)
        valid = fe.build_valid_mask(views, gt, depth_min=0.1, depth_max=80.0)
        raw.update(pred, gt, valid, median_align=False)
        aligned.update(pred, gt, valid, median_align=True)
        normal_values.append(_normal_stats(pred, gt, intrinsics, valid))
        reliability = metric_utils.stack_output(output, "event_reliability")
        if reliability is not None:
            reliability_values.append(float(reliability.float().mean().cpu()))
        if batch_index < args.visual_batches:
            _save_visual(Path(args.output_dir) / "visuals" / scene, views, output, batch_index, args.visual_views)
        windows += 1
        if windows % args.print_freq == 0:
            print(f"[{scene}] {windows}/{len(loader)}", flush=True)
    row = {"scene": scene, "num_windows": windows, **raw.compute()}
    row.update({f"median_{key}": value for key, value in aligned.compute().items() if key != "depth_pixels"})
    for key in normal_values[0] if normal_values else []:
        finite = [value[key] for value in normal_values if np.isfinite(value[key])]
        row[key] = float(np.mean(finite)) if finite else float("nan")
    row["event_reliability_mean"] = float(np.mean(reliability_values)) if reliability_values else float("nan")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    parser.add_argument("--approach", choices=("auto", "rgb", "full_img_reliability"), default="auto")
    parser.add_argument("--output-dir", default="dsec_exp/results/evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--clip-stride", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--visual-batches", type=int, default=2)
    parser.add_argument("--visual-views", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--allow-unaligned-rgb", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = metric_utils.cfg_from_checkpoint(checkpoint, str(Path(__file__).parents[1] / "config" / "finetune_dsec_event.yaml"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, approach = _build_model(checkpoint, cfg, args.approach, device)
    scenes = sorted(path.name for path in (Path(args.root) / "test").iterdir() if path.is_dir())
    if not scenes:
        raise RuntimeError(f"No test scenes in {Path(args.root) / 'test'}")
    rows = [_evaluate_scene(model, cfg, args, scene, device) for scene in scenes]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for key in fields:
        if key in {"scene", "num_windows", "depth_pixels"}:
            continue
        values = [row.get(key, float("nan")) for row in rows]
        values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
        summary[key] = float(np.mean(values)) if values else float("nan")
    report = {"checkpoint": args.checkpoint, "approach": approach, "scenes": rows, "metrics_mean": summary}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
