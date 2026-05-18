"""Evaluate long-window highlight-event filtering.

Specular highlights tend to fire repeatedly in the same image area over a
longer time window. Geometry edges usually appear as a shorter sweep. This
script builds a temporal-occupancy/density mask from event voxel bins, removes
those persistent high-density regions, and compares event/geometry metrics
before and after filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from eventvggt.datasets.my_event_dataset import get_combined_dataset
from exp_test.first_cue_correlation_test import (
    EPS,
    binary_auc,
    box_blur,
    f1_score,
    gradients,
    gradient_magnitude,
    gray_to_rgb,
    label_panel,
    laplacian,
    make_grid,
    nanmean,
    normal_gradient,
    normalize_normals,
    normals_from_depth,
    pearson_corr,
    robust_normalize,
    sample_valid_pixels,
    signed_to_rgb,
    tensor_image_to_uint8,
    to_numpy,
)


def _odd_ksize(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def _morph_mask(mask: np.ndarray, ksize: int, op: str) -> np.ndarray:
    ksize = _odd_ksize(ksize)
    if ksize <= 1:
        return mask.astype(bool)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    value = mask.astype(np.uint8)
    if op == "dilate":
        out = cv2.dilate(value, kernel, iterations=1)
    elif op == "close":
        out = cv2.morphologyEx(value, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Unknown morphology op {op}")
    return out.astype(bool)


def _build_voxel_from_events(view: Dict, height: int, width: int, num_bins: int) -> np.ndarray:
    event_xy = to_numpy(view.get("event_xy", np.zeros((0, 2), dtype=np.int32))).astype(np.int64)
    event_t = to_numpy(view.get("event_t", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    event_p = to_numpy(view.get("event_p", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    voxel = np.zeros((2 * num_bins, height, width), dtype=np.float32)
    if event_xy.size == 0 or event_t.size == 0 or event_p.size == 0:
        return voxel

    time_range = to_numpy(view.get("event_time_range", np.array([0.0, 0.0], dtype=np.float32))).astype(np.float32)
    if time_range.size >= 2 and float(time_range[1]) > float(time_range[0]):
        t0, t1 = float(time_range[0]), float(time_range[1])
    else:
        t0, t1 = float(event_t.min()), float(event_t.max())
    duration = max(t1 - t0, EPS)
    bin_idx = np.floor((event_t - t0) / duration * num_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    channel = bin_idx + np.where(event_p > 0, 0, num_bins)
    x = np.clip(event_xy[:, 0], 0, width - 1)
    y = np.clip(event_xy[:, 1], 0, height - 1)
    flat = channel * (height * width) + y * width + x
    np.add.at(voxel.reshape(-1), flat, np.abs(event_p).astype(np.float32, copy=False))
    return voxel


def get_event_voxel(view: Dict, height: int, width: int, fallback_bins: int) -> np.ndarray:
    voxel = to_numpy(view.get("event_voxel", np.zeros((0, height, width), dtype=np.float32))).astype(np.float32)
    if voxel.ndim == 3 and voxel.shape[0] >= 2 and voxel.shape[1:] == (height, width):
        return np.nan_to_num(voxel, nan=0.0, posinf=0.0, neginf=0.0)
    if voxel.ndim == 3 and voxel.shape[0] >= 2:
        resized = [
            cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
            for channel in voxel
        ]
        return np.nan_to_num(np.stack(resized, axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return _build_voxel_from_events(view, height, width, fallback_bins)


def voxel_to_event_maps(voxel: np.ndarray) -> Dict[str, np.ndarray]:
    channels, height, width = voxel.shape
    num_bins = max(channels // 2, 1)
    pos_bins = np.maximum(voxel[:num_bins], 0.0)
    neg_bins = np.maximum(voxel[num_bins : 2 * num_bins], 0.0)
    pos = pos_bins.sum(axis=0).astype(np.float32)
    neg = neg_bins.sum(axis=0).astype(np.float32)
    event_count = pos + neg
    event_abs = np.log1p(event_count)
    if event_abs.max() > 0:
        event_abs = event_abs / max(float(np.percentile(event_abs[event_abs > 0], 99.0)), EPS)
    event_abs = np.clip(event_abs, 0.0, 1.0).astype(np.float32)
    event_signed = ((pos - neg) / (event_count + EPS)).astype(np.float32)

    support = pos_bins + neg_bins
    bin_values = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)[:, None, None]
    time_surface = np.where(support > 0.0, bin_values, 0.0).max(axis=0).astype(np.float32)
    time_grad = gradient_magnitude(box_blur(time_surface, iterations=2))
    egx, egy = gradients(box_blur(event_abs, iterations=2))
    event_orientation = np.arctan2(egy, egx).astype(np.float32)
    return {
        "event_abs": event_abs,
        "event_signed": event_signed,
        "event_time_grad": time_grad,
        "event_orientation": event_orientation,
        "event_pos": pos,
        "event_neg": neg,
        "support_bins": support.astype(np.float32),
    }


def build_highlight_mask(voxel: np.ndarray, valid_mask: np.ndarray, args) -> Dict[str, np.ndarray]:
    maps = voxel_to_event_maps(voxel)
    support = np.log1p(maps["support_bins"])
    num_bins = support.shape[0]
    active = np.zeros_like(support, dtype=bool)
    for bin_idx in range(num_bins):
        values = support[bin_idx][valid_mask & (support[bin_idx] > 0)]
        if values.size == 0:
            continue
        threshold = np.percentile(values, args.bin_active_percentile) * float(args.bin_active_scale)
        active[bin_idx] = support[bin_idx] >= max(float(threshold), EPS)

    occupancy = active.mean(axis=0).astype(np.float32)
    density = support.sum(axis=0).astype(np.float32)
    density_norm = robust_normalize(density, mask=valid_mask, percentile=args.density_norm_percentile)
    mean_support = support.mean(axis=0)
    std_support = support.std(axis=0)
    stability = np.clip(1.0 - std_support / (mean_support + EPS), 0.0, 1.0).astype(np.float32)
    score = (occupancy * density_norm * (args.stability_bias + (1.0 - args.stability_bias) * stability)).astype(np.float32)

    candidate = (
        valid_mask
        & (occupancy >= args.min_occupancy)
        & (density_norm >= args.min_density)
        & np.isfinite(score)
    )
    if candidate.any():
        score_threshold = np.percentile(score[candidate], args.highlight_percentile)
        highlight_mask = candidate & (score >= score_threshold)
    else:
        score_threshold = float("inf")
        highlight_mask = np.zeros_like(valid_mask, dtype=bool)

    if args.mask_close_ksize > 1:
        highlight_mask = _morph_mask(highlight_mask, args.mask_close_ksize, "close") & valid_mask
    if args.mask_dilate_ksize > 1:
        highlight_mask = _morph_mask(highlight_mask, args.mask_dilate_ksize, "dilate") & valid_mask

    return {
        **maps,
        "temporal_occupancy": occupancy,
        "density_norm": density_norm,
        "temporal_stability": stability,
        "highlight_score": score,
        "highlight_mask": highlight_mask.astype(bool),
        "highlight_score_threshold": np.array(score_threshold, dtype=np.float32),
    }


def compute_geometry_maps(view: Dict, args) -> Dict[str, np.ndarray]:
    depth = to_numpy(view["depthmap"]).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    height, width = depth.shape
    mask = to_numpy(view.get("valid_mask", view.get("mask", np.ones_like(depth, dtype=bool)))).astype(bool)
    mask &= np.isfinite(depth) & (depth > args.depth_min)

    intrinsics = to_numpy(view["camera_intrinsics"]).astype(np.float32)
    normal = to_numpy(view.get("normal_gt", np.zeros((height, width, 3), dtype=np.float32)))
    if normal.ndim == 3 and np.abs(normal).sum() > EPS:
        normal = normalize_normals(normal)
    else:
        normal = normals_from_depth(depth, intrinsics)

    rho = np.zeros_like(depth, dtype=np.float32)
    rho[mask] = 1.0 / np.maximum(depth[mask], args.depth_min)
    rho = box_blur(rho, iterations=1)
    normal_grad = normal_gradient(normal)
    inv_depth_lap = laplacian(rho)
    abs_inv_depth_lap = np.abs(inv_depth_lap)
    geom_detail = 0.5 * robust_normalize(normal_grad, mask) + 0.5 * robust_normalize(abs_inv_depth_lap, mask)
    geom_detail = box_blur(geom_detail, iterations=1)
    ggx, ggy = gradients(geom_detail)
    return {
        "depth": depth,
        "mask": mask,
        "normal_grad": normal_grad,
        "abs_inv_depth_lap": abs_inv_depth_lap,
        "inv_depth_lap": inv_depth_lap,
        "geom_detail": geom_detail.astype(np.float32),
        "geom_orientation": np.arctan2(ggy, ggx).astype(np.float32),
    }


def metric_pack(
    event_abs: np.ndarray,
    event_signed: np.ndarray,
    event_orientation: np.ndarray,
    geometry: Dict[str, np.ndarray],
    sampled: np.ndarray,
    rng: np.random.Generator,
    args,
) -> Dict[str, float]:
    flat = lambda x: x.reshape(-1)[sampled]
    e_abs = flat(event_abs)
    e_signed = flat(event_signed)
    geom = flat(geometry["geom_detail"])
    n_grad = flat(geometry["normal_grad"])
    lap_abs = flat(geometry["abs_inv_depth_lap"])
    lap_signed = flat(geometry["inv_depth_lap"])
    g_ori = flat(geometry["geom_orientation"])
    e_ori = flat(event_orientation)

    geom_thr = np.percentile(geom, args.geometry_percentile) if geom.size else float("inf")
    positive = e_abs[e_abs > 0]
    event_thr = np.percentile(positive, args.event_percentile) if positive.size else float("inf")
    high_geom = geom >= geom_thr
    event_edge = e_abs >= event_thr
    shuffled = e_abs.copy()
    rng.shuffle(shuffled)

    signed_valid = (np.abs(e_signed) > 0.05) & (np.abs(lap_signed) >= np.percentile(np.abs(lap_signed), 70.0))
    if signed_valid.sum() >= 8:
        direct = np.mean(np.sign(e_signed[signed_valid]) == np.sign(lap_signed[signed_valid]))
        flipped = np.mean(-np.sign(e_signed[signed_valid]) == np.sign(lap_signed[signed_valid]))
        signed_best = max(float(direct), float(flipped))
    else:
        direct = flipped = signed_best = float("nan")

    ori_weight = e_abs * robust_normalize(geom, None)
    ori_valid = np.isfinite(e_ori) & np.isfinite(g_ori) & (ori_weight > np.percentile(ori_weight, 70.0))
    orientation_alignment = (
        float(np.average(np.abs(np.cos(e_ori[ori_valid] - g_ori[ori_valid])), weights=ori_weight[ori_valid] + EPS))
        if ori_valid.sum() >= 8
        else float("nan")
    )

    return {
        "corr_event_abs_normal_grad": pearson_corr(e_abs, n_grad),
        "corr_event_abs_abs_laplacian_inv_depth": pearson_corr(e_abs, lap_abs),
        "corr_event_abs_geom_detail": pearson_corr(e_abs, geom),
        "auc_event_abs_high_geom_detail": binary_auc(e_abs, high_geom),
        "f1_event_edge_geom_edge": f1_score(event_edge, high_geom),
        "signed_curvature_acc_best": signed_best,
        "signed_curvature_acc_direct": float(direct),
        "signed_curvature_acc_flipped": float(flipped),
        "orientation_alignment": orientation_alignment,
        "shuffle_corr_event_abs_geom_detail": pearson_corr(shuffled, geom),
        "shuffle_auc_event_abs_high_geom_detail": binary_auc(shuffled, high_geom),
        "shuffle_f1_event_edge_geom_edge": f1_score(shuffled >= event_thr, high_geom),
        "event_pixels_ratio": float((e_abs > 0).mean()) if e_abs.size else float("nan"),
    }


def compute_frame(view: Dict, rng: np.random.Generator, args) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    geometry = compute_geometry_maps(view, args)
    height, width = geometry["depth"].shape
    voxel = get_event_voxel(view, height, width, args.fallback_bins)
    highlight = build_highlight_mask(voxel, geometry["mask"], args)

    raw_abs = highlight["event_abs"]
    raw_signed = highlight["event_signed"]
    raw_orientation = highlight["event_orientation"]
    keep = np.where(highlight["highlight_mask"], float(args.highlight_keep), 1.0).astype(np.float32)
    filtered_abs = raw_abs * keep
    filtered_signed = raw_signed * keep
    egx, egy = gradients(box_blur(filtered_abs, iterations=2))
    filtered_orientation = np.arctan2(egy, egx).astype(np.float32)

    sampled = sample_valid_pixels(geometry["mask"], args.max_pixels_per_frame, rng)
    raw_metrics = metric_pack(raw_abs, raw_signed, raw_orientation, geometry, sampled, rng, args)
    filtered_metrics = metric_pack(filtered_abs, filtered_signed, filtered_orientation, geometry, sampled, rng, args)

    metrics = {f"raw_{key}": value for key, value in raw_metrics.items()}
    metrics.update({f"filtered_{key}": value for key, value in filtered_metrics.items()})
    valid = geometry["mask"]
    raw_energy = float(raw_abs[valid].sum())
    removed_energy = float((raw_abs * highlight["highlight_mask"])[valid].sum())
    metrics.update(
        {
            "highlight_mask_ratio": float(highlight["highlight_mask"][valid].mean()) if valid.any() else float("nan"),
            "highlight_removed_energy_ratio": removed_energy / max(raw_energy, EPS),
            "valid_pixels": float(sampled.size),
        }
    )

    maps = {
        "rgb": tensor_image_to_uint8(view["img"]),
        "raw_event_abs": raw_abs,
        "filtered_event_abs": filtered_abs,
        "removed_event_abs": raw_abs * highlight["highlight_mask"],
        "event_signed": raw_signed,
        "temporal_occupancy": highlight["temporal_occupancy"],
        "density_norm": highlight["density_norm"],
        "temporal_stability": highlight["temporal_stability"],
        "highlight_score": highlight["highlight_score"],
        "highlight_mask": highlight["highlight_mask"].astype(np.float32),
        "geom_detail": geometry["geom_detail"],
        "normal_grad": geometry["normal_grad"],
        "abs_inv_depth_lap": geometry["abs_inv_depth_lap"],
        "mask": geometry["mask"],
    }
    return metrics, maps


def save_filter_visualization(out_dir: Path, maps: Dict[str, np.ndarray], label: str) -> None:
    mask = maps["mask"]
    rows = [
        [
            label_panel("rgb", maps["rgb"]),
            label_panel("raw_event_abs", gray_to_rgb(maps["raw_event_abs"])),
            label_panel("highlight_score", gray_to_rgb(maps["highlight_score"], mask=mask)),
            label_panel("highlight_mask", gray_to_rgb(maps["highlight_mask"], mask=mask, percentile=100.0)),
        ],
        [
            label_panel("temporal_occupancy", gray_to_rgb(maps["temporal_occupancy"], mask=mask, percentile=100.0)),
            label_panel("density_norm", gray_to_rgb(maps["density_norm"], mask=mask, percentile=100.0)),
            label_panel("removed_event_abs", gray_to_rgb(maps["removed_event_abs"])),
            label_panel("filtered_event_abs", gray_to_rgb(maps["filtered_event_abs"])),
        ],
        [
            label_panel("event_signed", signed_to_rgb(maps["event_signed"], percentile=100.0)),
            label_panel("geom_detail", gray_to_rgb(maps["geom_detail"], mask=mask)),
            label_panel("normal_grad", gray_to_rgb(maps["normal_grad"], mask=mask)),
            label_panel("abs_lap_inv_depth", gray_to_rgb(maps["abs_inv_depth_lap"], mask=mask)),
        ],
    ]
    make_grid(rows).save(out_dir / f"{label}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Highlight event density filter test")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/highlight_event_filter")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=3)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-min", type=float, default=1e-6)
    parser.add_argument("--max-pixels-per-frame", type=int, default=50000)
    parser.add_argument("--event-percentile", type=float, default=90.0)
    parser.add_argument("--geometry-percentile", type=float, default=90.0)
    parser.add_argument("--fallback-bins", type=int, default=10)
    parser.add_argument("--bin-active-percentile", type=float, default=65.0)
    parser.add_argument("--bin-active-scale", type=float, default=1.0)
    parser.add_argument("--min-occupancy", type=float, default=0.65)
    parser.add_argument("--min-density", type=float, default=0.25)
    parser.add_argument("--density-norm-percentile", type=float, default=99.0)
    parser.add_argument("--stability-bias", type=float, default=0.5)
    parser.add_argument("--highlight-percentile", type=float, default=80.0)
    parser.add_argument("--highlight-keep", type=float, default=0.0)
    parser.add_argument("--mask-close-ksize", type=int, default=3)
    parser.add_argument("--mask-dilate-ksize", type=int, default=3)
    parser.add_argument("--save-visuals", type=int, default=8)
    return parser.parse_args()


def iter_event_views(dataset, num_samples: int):
    sample_count = min(num_samples, len(dataset))
    for sample_idx in range(sample_count):
        views = dataset[sample_idx]
        for frame_idx, view in enumerate(views):
            if not bool(np.asarray(view.get("has_event", frame_idx > 0))):
                continue
            yield sample_idx, frame_idx, view


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=args.scene_names,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.active_scene_count,
        split=args.split,
        test_frame_count=args.test_frame_count,
        ldr_event_id=args.ldr_event_id,
    )
    if len(dataset) <= 0:
        raise RuntimeError(f"No samples found under {args.root}")

    records: List[Dict[str, float | str | int]] = []
    visual_count = 0
    for sample_idx, frame_idx, view in iter_event_views(dataset, args.num_samples):
        metrics, maps = compute_frame(view, rng, args)
        record: Dict[str, float | str | int] = {
            "sample_idx": sample_idx,
            "frame_idx": frame_idx,
            "label": str(view.get("label", "")),
            "instance": str(view.get("instance", "")),
            "ldr_event_id": str(view.get("ldr_event_id", args.ldr_event_id)),
        }
        record.update(metrics)
        records.append(record)
        if visual_count < args.save_visuals:
            save_filter_visualization(vis_dir, maps, f"sample_{sample_idx:04d}_frame_{frame_idx:02d}")
            visual_count += 1

    if not records:
        raise RuntimeError("No event-bearing frames were evaluated")

    metric_keys = [key for key in records[0].keys() if key not in {"sample_idx", "frame_idx", "label", "instance", "ldr_event_id"}]
    summary = {
        "num_records": len(records),
        "num_samples": min(args.num_samples, len(dataset)),
        "active_scenes": dataset.get_active_scenes(),
        "root": args.root,
        "ldr_event_id": args.ldr_event_id,
        "resolution": args.resolution,
        "filter": {
            "bin_active_percentile": args.bin_active_percentile,
            "min_occupancy": args.min_occupancy,
            "min_density": args.min_density,
            "highlight_percentile": args.highlight_percentile,
            "highlight_keep": args.highlight_keep,
            "mask_close_ksize": args.mask_close_ksize,
            "mask_dilate_ksize": args.mask_dilate_ksize,
        },
        "metrics_mean": {key: nanmean(float(row[key]) for row in records) for key in metric_keys},
    }

    csv_path = out_dir / "per_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Highlight Event Filter Test\n\n")
        f.write(f"- Dataset root: `{args.root}`\n")
        f.write(f"- Active scenes: `{summary['active_scenes']}`\n")
        f.write(f"- Evaluated event frames: `{len(records)}`\n")
        f.write(f"- Filter: `{summary['filter']}`\n")
        f.write(f"- Per-frame CSV: `{csv_path.name}`\n")
        f.write(f"- Visualizations: `visuals/`\n\n")
        f.write("## Mean Metrics\n\n")
        for key in metric_keys:
            f.write(f"- `{key}`: {summary['metrics_mean'][key]:.6f}\n")
        f.write("\n## Reading\n\n")
        f.write("- If filtered correlation/AUC rises while event_pixels_ratio drops, persistent highlight events were hurting the geometry cue.\n")
        f.write("- If highlight_mask_ratio is large and metrics drop, the filter is too aggressive; raise min_occupancy or highlight_percentile.\n")
        f.write("- Compare raw vs filtered shuffle metrics to check that gains are not from threshold bias.\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
