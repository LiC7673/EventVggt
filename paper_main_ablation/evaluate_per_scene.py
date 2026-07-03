"""Evaluate one module-ablation checkpoint per scene and save visual panels."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ablation.eag3r_metrics_eval as metrics  # noqa: E402
import finetune_event as fe  # noqa: E402
from eventvggt.datasets.my_event_dataset import (  # noqa: E402
    event_multiview_collate,
    get_combined_dataset,
)
from paper_main_ablation.evaluate_main_table import _build_main_table_model  # noqa: E402


METRIC_COLUMNS = (
    "abs_rel", "delta1", "rmse_log", "rmse", "mae",
    "median_abs_rel", "median_delta1", "median_rmse_log", "median_rmse", "median_mae",
    "normal_error_deg", "corr_event_normal_error", "high_event_normal_error_deg",
    "low_event_normal_error_deg", "high_minus_low_normal_error_deg",
    "ate", "rpe_trans", "rpe_rot_deg", "event_reliability_mean",
)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "scene"


def _event_rgb(voxel: torch.Tensor) -> np.ndarray:
    value = voxel.detach().float().cpu().numpy()
    bins = max(value.shape[0] // 2, 1)
    pos = np.log1p(np.clip(value[:bins], 0.0, None).sum(axis=0))
    neg = np.log1p(np.clip(value[bins : 2 * bins], 0.0, None).sum(axis=0))
    joined = np.concatenate([pos.reshape(-1), neg.reshape(-1)])
    scale = max(float(np.percentile(joined, 99.5)) if joined.size else 1.0, 1.0e-6)
    image = np.zeros((*pos.shape, 3), dtype=np.float32)
    image[..., 0] = np.clip(pos / scale, 0.0, 1.0)
    image[..., 2] = np.clip(neg / scale, 0.0, 1.0)
    image[..., 1] = 0.2 * np.minimum(image[..., 0], image[..., 2])
    return (image * 255.0).round().astype(np.uint8)


def _save_visuals(path, views, output, batch_index, max_views):
    depth_pred = metrics.stack_output(output, "depth")
    if depth_pred is None:
        return
    depth_gt = fe.stack_view_field(views, "depthmap").to(depth_pred.device, depth_pred.dtype)
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(
        depth_pred.device, depth_pred.dtype
    )
    valid = fe.build_valid_mask(views, depth_gt)
    pred_normals = fe.depth_to_normals(depth_pred.clamp_min(1.0e-6), intrinsics)
    gt_normals = fe.depth_to_normals(depth_gt.clamp_min(1.0e-6), intrinsics)
    reliability = metrics.stack_output(output, "event_reliability")

    for frame_index in range(min(len(views), max_views)):
        mask = valid[0, frame_index]
        log_error = (
            torch.log(depth_pred[0, frame_index].clamp_min(1.0e-6))
            - torch.log(depth_gt[0, frame_index].clamp_min(1.0e-6))
        ).abs()
        if reliability is None:
            reliability_map = torch.zeros_like(depth_pred[0, frame_index])
        else:
            reliability_map = reliability[0, frame_index]
        panels = [
            fe.make_labeled_panel("rgb", fe.tensor_rgb_to_uint8(views[frame_index]["img"][0], mask)),
            fe.make_labeled_panel("event", _event_rgb(views[frame_index]["event_voxel"][0])),
            fe.make_labeled_panel("gt_depth", fe.depth_to_uint8(depth_gt[0, frame_index], mask)),
            fe.make_labeled_panel("pred_depth", fe.depth_to_uint8(depth_pred[0, frame_index], mask)),
            fe.make_labeled_panel("log_depth_error", fe.depth_to_uint8(log_error, mask)),
            fe.make_labeled_panel("event_reliability", fe.depth_to_uint8(reliability_map, mask)),
            fe.make_labeled_panel("pred_normal", fe.normal_to_uint8(pred_normals[0, frame_index], mask)),
            fe.make_labeled_panel("gt_normal", fe.normal_to_uint8(gt_normals[0, frame_index], mask)),
        ]
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        path.mkdir(parents=True, exist_ok=True)
        canvas.save(path / f"batch_{batch_index:04d}_view_{frame_index:02d}.png")


def _discover_scenes(cfg, args):
    if args.scene_manifest:
        with Path(args.scene_manifest).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        scenes = list(data.get("heldout_scenes", []))
        if len(scenes) != args.scene_count or len(set(scenes)) != args.scene_count:
            raise RuntimeError(
                f"Expected {args.scene_count} unique scenes in {args.scene_manifest}, got {scenes}"
            )
        return scenes
    probe = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=int(cfg.data.fps),
        seed=int(cfg.seed),
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.scene_count,
        split="all",
        test_frame_count=0,
        ldr_event_id=args.ldr_event_id,
        event_spatial_transform=str(getattr(cfg.data, "event_spatial_transform", "auto")),
        event_resize_method=str(getattr(cfg.data, "event_resize_method", "voxel_antialias")),
        event_resize_bins=args.event_resize_bins,
        return_normal_gt=True,
    )
    scenes = probe.get_active_scenes()
    if len(scenes) != args.scene_count or len(set(scenes)) != args.scene_count:
        raise RuntimeError(f"Expected {args.scene_count} unique held-out scenes, got {scenes}")
    return scenes


def _scene_loader(cfg, args, scene):
    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=int(cfg.data.fps),
        seed=int(cfg.seed),
        scene_names=[scene],
        initial_scene_idx=0,
        active_scene_count=1,
        split="all",
        test_frame_count=0,
        ldr_event_id=args.ldr_event_id,
        event_spatial_transform=str(getattr(cfg.data, "event_spatial_transform", "auto")),
        event_resize_method=str(getattr(cfg.data, "event_resize_method", "voxel_antialias")),
        event_resize_bins=args.event_resize_bins,
        return_normal_gt=True,
    )
    if dataset.get_active_scenes() != [scene]:
        raise RuntimeError(f"Failed to activate held-out scene {scene}")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )


def _evaluate_scene(model, cfg, args, scene, device, visual_root):
    loader = _scene_loader(cfg, args, scene)
    depth = metrics.DepthMetrics()
    aligned = metrics.DepthMetrics()
    normal_acc = {
        key: metrics.MeanAccumulator()
        for key in (
            "normal_error_deg", "corr_event_normal_error", "high_event_normal_error_deg",
            "low_event_normal_error_deg", "high_minus_low_normal_error_deg",
        )
    }
    pose_acc = {key: metrics.MeanAccumulator() for key in ("ate", "rpe_trans", "rpe_rot_deg")}
    reliability_acc = metrics.MeanAccumulator()
    normal_args = SimpleNamespace(
        event_resize_bins=args.event_resize_bins,
        event_support_mode="temporal_polarity",
        event_high_fraction=0.2,
        event_low_fraction=0.2,
    )
    evaluated = 0
    for batch_index, views in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        views = metrics.move_views_to_device(fe.maybe_denormalize_views(views), device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(views)
        pred = metrics.stack_output(output, "depth")
        pose = metrics.stack_output(output, "camera_pose")
        reliability = metrics.stack_output(output, "event_reliability")
        if pred is None:
            raise RuntimeError("Model output does not contain depth")
        gt = fe.stack_view_field(views, "depthmap").to(device=device, dtype=pred.dtype)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=device, dtype=pred.dtype)
        gt_pose = fe.stack_view_field(views, "camera_pose").to(device=device, dtype=torch.float32)
        valid = fe.build_valid_mask(views, gt)
        depth.update(pred, gt, valid, median_align=False)
        aligned.update(pred, gt, valid, median_align=True)
        for key, value in metrics.normal_error_metrics(
            pred, gt, intrinsics, valid, views, normal_args
        ).items():
            normal_acc[key].update(value)
        if pose is not None:
            height, width = pred.shape[-2:]
            pred_c2w, _ = fe.pose_encoding_to_c2w(
                pose.float(), image_size_hw=(height, width)
            )
            for key, value in metrics.pose_errors(
                pred_c2w, gt_pose, scale_align=args.pose_scale_align
            ).items():
                pose_acc[key].update(value)
        if reliability is not None:
            reliability_acc.update(float(reliability.float().mean().cpu()))
        if batch_index < args.visual_batches:
            _save_visuals(
                visual_root / _safe_name(scene),
                views,
                output,
                batch_index,
                args.visual_views,
            )
        evaluated += 1
        if evaluated % args.print_freq == 0:
            print(f"[scene] {scene}: {evaluated}/{len(loader)}", flush=True)

    row = {
        "model": args.name,
        "ldr": args.ldr_event_id,
        "scene": scene,
        **depth.compute(),
        **{f"median_{key}": value for key, value in aligned.compute().items() if key != "depth_pixels"},
        **{key: value.compute() for key, value in normal_acc.items()},
        **{key: value.compute() for key, value in pose_acc.items()},
        "event_reliability_mean": reliability_acc.compute(),
        "num_windows": evaluated,
    }
    return row


def _write(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    fields = ["model", "ldr", "scene", *METRIC_COLUMNS, "depth_pixels", "num_windows"]
    with (output / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "per_scene_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--ldr-event-id", required=True)
    parser.add_argument("--initial-scene-idx", type=int, default=12)
    parser.add_argument("--scene-count", type=int, default=4)
    parser.add_argument("--scene-manifest", default=None)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--visual-batches", type=int, default=1)
    parser.add_argument("--visual-views", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--pose-scale-align", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = metrics.torch_load(checkpoint_path)
    cfg = metrics.cfg_from_checkpoint(checkpoint, None)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_main_table_model("main_table", cfg, checkpoint, device)
    scenes = _discover_scenes(cfg, args)
    print(
        f"[eval] model={args.name} LDR={args.ldr_event_id} scenes={scenes} "
        f"checkpoint={checkpoint_path}"
    )
    start = time.time()
    rows = [
        _evaluate_scene(model, cfg, args, scene, device, output / "visuals")
        for scene in scenes
    ]
    _write(rows, output)
    print(f"[done] {len(rows)} scene rows in {time.time() - start:.1f}s -> {output}")


if __name__ == "__main__":
    main()
