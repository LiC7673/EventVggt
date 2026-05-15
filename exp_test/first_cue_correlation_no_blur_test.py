"""No-blur evidence test for event cues and GT differential geometry.

This is a copy of first_cue_correlation_test.py with all blur operations
disabled, to test whether structure-only preprocessing is more reliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
except ImportError:  # pragma: no cover - real training env has torch
    torch = None

from eventvggt.datasets.my_event_dataset import get_combined_dataset


EPS = 1e-6


def to_numpy(value):
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def robust_normalize(value: np.ndarray, mask: np.ndarray | None = None, percentile: float = 99.0) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(value)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return np.zeros_like(value, dtype=np.float32)
    scale = np.percentile(np.abs(value[valid]), percentile)
    scale = max(float(scale), EPS)
    return np.clip(value / scale, 0.0, 1.0).astype(np.float32)


def signed_to_rgb(value: np.ndarray, mask: np.ndarray | None = None, percentile: float = 99.0) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(value)
    if mask is not None:
        valid &= mask
    scale = np.percentile(np.abs(value[valid]), percentile) if valid.any() else 1.0
    scale = max(float(scale), EPS)
    x = np.clip(value / scale, -1.0, 1.0)
    pos = np.clip(x, 0.0, 1.0)
    neg = np.clip(-x, 0.0, 1.0)
    rgb = np.stack([pos, 0.55 * (pos + neg), neg], axis=-1)
    if mask is not None:
        rgb[~mask] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def gray_to_rgb(value: np.ndarray, mask: np.ndarray | None = None, percentile: float = 99.0) -> np.ndarray:
    normalized = robust_normalize(value, mask=mask, percentile=percentile)
    if mask is not None:
        normalized = normalized.copy()
        normalized[~mask] = 0.0
    return (np.repeat(normalized[..., None], 3, axis=-1) * 255.0).round().astype(np.uint8)


def tensor_image_to_uint8(image) -> np.ndarray:
    image = to_numpy(image).astype(np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.min() < -0.1:
        image = (image + 1.0) * 0.5
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def label_panel(label: str, image: np.ndarray) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    bar = Image.new("RGB", (panel.width, 22), color=(18, 18, 18))
    ImageDraw.Draw(bar).text((5, 4), label, fill=(230, 230, 230))
    out = Image.new("RGB", (panel.width, panel.height + 22), color=(0, 0, 0))
    out.paste(bar, (0, 0))
    out.paste(panel, (0, 22))
    return out


def make_grid(rows: List[List[Image.Image]]) -> Image.Image:
    row_images = []
    for row in rows:
        width = sum(img.width for img in row)
        height = max(img.height for img in row)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        x = 0
        for img in row:
            canvas.paste(img, (x, 0))
            x += img.width
        row_images.append(canvas)

    width = max(img.width for img in row_images)
    height = sum(img.height for img in row_images)
    grid = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for img in row_images:
        grid.paste(img, (0, y))
        y += img.height
    return grid


def box_blur(value: np.ndarray, iterations: int = 1) -> np.ndarray:
    return value.astype(np.float32)


def _odd_kernel_size(ksize: int) -> int:
    ksize = int(ksize)
    if ksize <= 1:
        return 0
    return ksize if ksize % 2 == 1 else ksize + 1


def _window_reduce(value: np.ndarray, ksize: int, op: str) -> np.ndarray:
    ksize = _odd_kernel_size(ksize)
    if ksize <= 1:
        return value.astype(np.float32)
    radius = ksize // 2
    padded = np.pad(value.astype(np.float32), radius, mode="edge")
    h, w = value.shape
    windows = [
        padded[y : y + h, x : x + w]
        for y in range(ksize)
        for x in range(ksize)
    ]
    stacked = np.stack(windows, axis=0)
    if op == "median":
        return np.median(stacked, axis=0).astype(np.float32)
    if op == "max":
        return stacked.max(axis=0).astype(np.float32)
    if op == "min":
        return stacked.min(axis=0).astype(np.float32)
    raise ValueError(f"Unknown window op {op}")


def median_filter2d(value: np.ndarray, ksize: int) -> np.ndarray:
    return _window_reduce(value, ksize, "median")


def max_filter2d(value: np.ndarray, ksize: int) -> np.ndarray:
    return _window_reduce(value, ksize, "max")


def min_filter2d(value: np.ndarray, ksize: int) -> np.ndarray:
    return _window_reduce(value, ksize, "min")


def close_filter2d(value: np.ndarray, ksize: int) -> np.ndarray:
    return min_filter2d(max_filter2d(value, ksize), ksize)


def preprocess_map(
    value: np.ndarray,
    *,
    median_ksize: int = 0,
    close_ksize: int = 0,
    dilate_ksize: int = 0,
    blur_iters: int = 0,
) -> np.ndarray:
    out = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if median_ksize > 1:
        out = median_filter2d(out, median_ksize)
    if close_ksize > 1:
        out = close_filter2d(out, close_ksize)
    if dilate_ksize > 1:
        out = max_filter2d(out, dilate_ksize)
    if blur_iters > 0:
        out = box_blur(out, iterations=blur_iters)
    return out.astype(np.float32)


def gradients(value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    value = value.astype(np.float32)
    gx = np.zeros_like(value, dtype=np.float32)
    gy = np.zeros_like(value, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (value[:, 2:] - value[:, :-2])
    gy[1:-1, :] = 0.5 * (value[2:, :] - value[:-2, :])
    gx[:, 0] = value[:, 1] - value[:, 0]
    gx[:, -1] = value[:, -1] - value[:, -2]
    gy[0, :] = value[1, :] - value[0, :]
    gy[-1, :] = value[-1, :] - value[-2, :]
    return gx, gy


def gradient_magnitude(value: np.ndarray) -> np.ndarray:
    gx, gy = gradients(value)
    return np.sqrt(gx * gx + gy * gy + EPS).astype(np.float32)


def laplacian(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    padded = np.pad(value, 1, mode="edge")
    h, w = value.shape
    return (
        padded[1 : h + 1, 0:w]
        + padded[1 : h + 1, 2 : w + 2]
        + padded[0:h, 1 : w + 1]
        + padded[2 : h + 2, 1 : w + 1]
        - 4.0 * value
    ).astype(np.float32)


def normalize_normals(normal: np.ndarray) -> np.ndarray:
    normal = normal.astype(np.float32)
    if normal.max() > 2.0:
        normal = normal / 127.5 - 1.0
    elif normal.min() >= 0.0 and normal.max() <= 1.05:
        normal = normal * 2.0 - 1.0
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal / np.maximum(norm, EPS)


def normals_from_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    fx = max(float(intrinsics[0, 0]), EPS)
    fy = max(float(intrinsics[1, 1]), EPS)
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    points = np.stack([(xs - cx) / fx * depth, (ys - cy) / fy * depth, depth], axis=-1)
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1, :] = points[2:, :] - points[:-2, :]
    normal = np.cross(dy, dx)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal / np.maximum(norm, EPS)


def normal_gradient(normal: np.ndarray) -> np.ndarray:
    gx = np.zeros(normal.shape[:2], dtype=np.float32)
    gy = np.zeros(normal.shape[:2], dtype=np.float32)
    for c in range(3):
        dcx, dcy = gradients(normal[..., c])
        gx += dcx * dcx
        gy += dcy * dcy
    return np.sqrt(gx + gy + EPS).astype(np.float32)


def build_event_cues(view: Dict, height: int, width: int) -> Dict[str, np.ndarray]:
    event_xy = to_numpy(view.get("event_xy", np.zeros((0, 2), dtype=np.int32))).astype(np.int64)
    event_p = to_numpy(view.get("event_p", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    event_t = to_numpy(view.get("event_t", np.zeros((0,), dtype=np.float32))).astype(np.float32)

    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)
    time_surface = np.zeros((height, width), dtype=np.float32)
    if event_xy.size == 0:
        event_abs = pos
        return {
            "event_abs": event_abs,
            "event_signed": event_abs.copy(),
            "event_time_grad": event_abs.copy(),
            "event_orientation": event_abs.copy(),
            "event_pos": pos,
            "event_neg": neg,
        }

    x = np.clip(event_xy[:, 0], 0, width - 1)
    y = np.clip(event_xy[:, 1], 0, height - 1)
    flat = y * width + x
    pos_mask = event_p > 0
    np.add.at(pos.reshape(-1), flat[pos_mask], 1.0)
    np.add.at(neg.reshape(-1), flat[~pos_mask], 1.0)

    if event_t.size == event_xy.shape[0]:
        t_min = float(event_t.min())
        t_max = float(event_t.max())
        t_norm = (event_t - t_min) / max(t_max - t_min, 1.0)
        np.maximum.at(time_surface.reshape(-1), flat, t_norm.astype(np.float32))

    event_count = pos + neg
    event_abs = np.log1p(event_count)
    if event_abs.max() > 0:
        event_abs = event_abs / max(float(np.percentile(event_abs[event_abs > 0], 99.0)), EPS)
    event_abs = np.clip(event_abs, 0.0, 1.0).astype(np.float32)
    event_signed = ((pos - neg) / (event_count + EPS)).astype(np.float32)
    event_time_grad = gradient_magnitude(box_blur(time_surface, iterations=2))
    egx, egy = gradients(box_blur(event_abs, iterations=2))
    event_orientation = np.arctan2(egy, egx).astype(np.float32)
    return {
        "event_abs": event_abs,
        "event_signed": event_signed,
        "event_time_grad": event_time_grad,
        "event_orientation": event_orientation,
        "event_pos": pos,
        "event_neg": neg,
    }


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 8:
        return float("nan")
    a = a[valid]
    b = b[valid]
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    if denom <= EPS:
        return float("nan")
    return float((a * b).sum() / denom)


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.astype(np.float64)
    labels = labels.astype(bool)
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    denom = 2.0 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else float("nan")


def sample_valid_pixels(valid: np.ndarray, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size > max_pixels:
        idx = rng.choice(idx, size=max_pixels, replace=False)
    return idx


def compute_frame_metrics(view: Dict, rng: np.random.Generator, args) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
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
    inv_depth_grad = gradient_magnitude(rho)
    inv_depth_lap = laplacian(rho)
    abs_inv_depth_lap = np.abs(inv_depth_lap)
    geom_detail = 0.5 * robust_normalize(normal_grad, mask) + 0.5 * robust_normalize(abs_inv_depth_lap, mask)
    geom_detail = box_blur(geom_detail, iterations=1)
    geom_detail_raw = geom_detail.copy()
    normal_grad_raw = normal_grad.copy()
    abs_inv_depth_lap_raw = abs_inv_depth_lap.copy()
    geom_detail = preprocess_map(
        geom_detail,
        median_ksize=args.geometry_median_ksize,
        close_ksize=args.geometry_close_ksize,
        dilate_ksize=args.geometry_dilate_ksize,
        blur_iters=args.geometry_blur_iters,
    )
    normal_grad_metric = preprocess_map(
        normal_grad,
        median_ksize=args.geometry_median_ksize,
        close_ksize=0,
        dilate_ksize=0,
        blur_iters=args.geometry_blur_iters,
    )
    abs_inv_depth_lap_metric = preprocess_map(
        abs_inv_depth_lap,
        median_ksize=args.geometry_median_ksize,
        close_ksize=0,
        dilate_ksize=0,
        blur_iters=args.geometry_blur_iters,
    )
    ggx, ggy = gradients(geom_detail)
    geom_orientation = np.arctan2(ggy, ggx).astype(np.float32)

    event = build_event_cues(view, height, width)
    event_abs_raw = event["event_abs"].copy()
    event_time_grad_raw = event["event_time_grad"].copy()
    event["event_abs"] = preprocess_map(
        event["event_abs"],
        median_ksize=args.event_median_ksize,
        close_ksize=args.event_close_ksize,
        dilate_ksize=args.event_dilate_ksize,
        blur_iters=args.event_blur_iters,
    )
    event["event_time_grad"] = preprocess_map(
        event["event_time_grad"],
        median_ksize=args.event_median_ksize,
        close_ksize=args.event_close_ksize,
        dilate_ksize=args.event_dilate_ksize,
        blur_iters=args.event_blur_iters,
    )
    if args.event_signed_median_ksize > 1:
        event["event_signed"] = median_filter2d(event["event_signed"], args.event_signed_median_ksize)
    egx, egy = gradients(box_blur(event["event_abs"], iterations=2))
    event["event_orientation"] = np.arctan2(egy, egx).astype(np.float32)
    valid = mask & np.isfinite(geom_detail)
    sampled = sample_valid_pixels(valid, args.max_pixels_per_frame, rng)
    flat = lambda x: x.reshape(-1)[sampled]

    e_abs = flat(event["event_abs"])
    e_signed = flat(event["event_signed"])
    n_grad = flat(normal_grad_metric)
    lap_abs = flat(abs_inv_depth_lap_metric)
    geom = flat(geom_detail)
    lap_signed = flat(inv_depth_lap)

    geom_thr = np.percentile(geom, args.geometry_percentile) if geom.size else float("inf")
    positive_events = e_abs[e_abs > 0]
    event_thr = (
        np.percentile(positive_events, args.event_percentile)
        if positive_events.size
        else float("inf")
    )
    high_geom = geom >= geom_thr
    event_edge = e_abs >= event_thr

    shuffled = e_abs.copy()
    rng.shuffle(shuffled)

    signed_valid = (np.abs(e_signed) > 0.05) & (np.abs(lap_signed) >= np.percentile(np.abs(lap_signed), 70.0))
    if signed_valid.sum() >= 8:
        direct = np.mean(np.sign(e_signed[signed_valid]) == np.sign(lap_signed[signed_valid]))
        flipped = np.mean(-np.sign(e_signed[signed_valid]) == np.sign(lap_signed[signed_valid]))
        signed_acc = max(float(direct), float(flipped))
    else:
        direct = flipped = signed_acc = float("nan")

    e_ori = flat(event["event_orientation"])
    g_ori = flat(geom_orientation)
    ori_weight = e_abs * robust_normalize(geom, None)
    ori_valid = np.isfinite(e_ori) & np.isfinite(g_ori) & (ori_weight > np.percentile(ori_weight, 70.0))
    orientation_alignment = (
        float(np.average(np.abs(np.cos(e_ori[ori_valid] - g_ori[ori_valid])), weights=ori_weight[ori_valid] + EPS))
        if ori_valid.sum() >= 8
        else float("nan")
    )

    metrics = {
        "corr_event_abs_normal_grad": pearson_corr(e_abs, n_grad),
        "corr_event_abs_abs_laplacian_inv_depth": pearson_corr(e_abs, lap_abs),
        "corr_event_abs_geom_detail": pearson_corr(e_abs, geom),
        "auc_event_abs_high_geom_detail": binary_auc(e_abs, high_geom),
        "f1_event_edge_geom_edge": f1_score(event_edge, high_geom),
        "signed_curvature_acc_best": signed_acc,
        "signed_curvature_acc_direct": float(direct),
        "signed_curvature_acc_flipped": float(flipped),
        "orientation_alignment": orientation_alignment,
        "shuffle_corr_event_abs_geom_detail": pearson_corr(shuffled, geom),
        "shuffle_auc_event_abs_high_geom_detail": binary_auc(shuffled, high_geom),
        "shuffle_f1_event_edge_geom_edge": f1_score(shuffled >= event_thr, high_geom),
        "valid_pixels": float(sampled.size),
        "event_pixels_ratio": float((e_abs > 0).mean()) if e_abs.size else float("nan"),
    }
    maps = {
        "rgb": tensor_image_to_uint8(view["img"]),
        "event_abs": event["event_abs"],
        "event_abs_raw": event_abs_raw,
        "event_signed": event["event_signed"],
        "event_time_grad": event["event_time_grad"],
        "event_time_grad_raw": event_time_grad_raw,
        "normal_grad": normal_grad_metric,
        "normal_grad_raw": normal_grad_raw,
        "inv_depth_lap": inv_depth_lap,
        "abs_inv_depth_lap_raw": abs_inv_depth_lap_raw,
        "geom_detail": geom_detail,
        "geom_detail_raw": geom_detail_raw,
        "mask": mask,
    }
    return metrics, maps


def nanmean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def save_visualization(out_dir: Path, maps: Dict[str, np.ndarray], label: str) -> None:
    mask = maps["mask"]
    rows = [
        [
            label_panel("rgb", maps["rgb"]),
            label_panel("event_abs_raw", gray_to_rgb(maps["event_abs_raw"], mask=None)),
            label_panel("event_abs", gray_to_rgb(maps["event_abs"], mask=None)),
            label_panel("event_signed", signed_to_rgb(maps["event_signed"], mask=None, percentile=100.0)),
            label_panel("event_time_grad", gray_to_rgb(maps["event_time_grad"], mask=None)),
        ],
        [
            label_panel("geom_detail_raw", gray_to_rgb(maps["geom_detail_raw"], mask=mask)),
            label_panel("normal_grad", gray_to_rgb(maps["normal_grad"], mask=mask)),
            label_panel("abs_lap_inv_depth", gray_to_rgb(np.abs(maps["inv_depth_lap"]), mask=mask)),
            label_panel("signed_lap_inv_depth", signed_to_rgb(maps["inv_depth_lap"], mask=mask)),
            label_panel("geom_detail", gray_to_rgb(maps["geom_detail"], mask=mask)),
        ],
    ]
    make_grid(rows).save(out_dir / f"{label}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="first.md event/geometry cue correlation test")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/first_cue_correlation")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--num-views", type=int, default=4)
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
    parser.add_argument(
        "--preprocess-preset",
        default="none",
        choices=["none", "median", "event_support"],
        help="No-blur preprocessing preset. Blur args are ignored in this script.",
    )
    parser.add_argument("--event-median-ksize", type=int, default=0)
    parser.add_argument("--event-close-ksize", type=int, default=0)
    parser.add_argument("--event-dilate-ksize", type=int, default=0)
    parser.add_argument("--event-blur-iters", type=int, default=0, help="Ignored in no-blur script")
    parser.add_argument("--event-signed-median-ksize", type=int, default=0)
    parser.add_argument("--geometry-median-ksize", type=int, default=0)
    parser.add_argument("--geometry-close-ksize", type=int, default=0)
    parser.add_argument("--geometry-dilate-ksize", type=int, default=0)
    parser.add_argument("--geometry-blur-iters", type=int, default=0, help="Ignored in no-blur script")
    parser.add_argument("--save-visuals", type=int, default=8)
    args = parser.parse_args()
    apply_preprocess_preset(args)
    return args


def apply_preprocess_preset(args: argparse.Namespace) -> None:
    if args.preprocess_preset == "median":
        if args.event_median_ksize <= 1:
            args.event_median_ksize = 3
        if args.geometry_median_ksize <= 1:
            args.geometry_median_ksize = 3
    elif args.preprocess_preset == "event_support":
        # Events are sparse. Dilate/close before blur keeps support instead of
        # deleting it like a plain median filter often does.
        if args.event_close_ksize <= 1:
            args.event_close_ksize = 3
        if args.event_dilate_ksize <= 1:
            args.event_dilate_ksize = 5
        if args.geometry_median_ksize <= 1:
            args.geometry_median_ksize = 3
    args.event_blur_iters = 0
    args.geometry_blur_iters = 0


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
    sample_count = min(args.num_samples, len(dataset))
    if sample_count <= 0:
        raise RuntimeError(f"No samples found under {args.root}")

    records: List[Dict[str, float | str | int]] = []
    visual_count = 0
    for sample_idx in range(sample_count):
        views = dataset[sample_idx]
        for frame_idx, view in enumerate(views):
            if not bool(np.asarray(view.get("has_event", frame_idx > 0))):
                continue
            metrics, maps = compute_frame_metrics(view, rng, args)
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
                safe_label = f"sample_{sample_idx:04d}_frame_{frame_idx:02d}"
                save_visualization(vis_dir, maps, safe_label)
                visual_count += 1

    if not records:
        raise RuntimeError("No event-bearing frames were evaluated")

    metric_keys = [key for key in records[0].keys() if key not in {"sample_idx", "frame_idx", "label", "instance", "ldr_event_id"}]
    summary = {
        "num_records": len(records),
        "num_samples": sample_count,
        "active_scenes": dataset.get_active_scenes(),
        "root": args.root,
        "ldr_event_id": args.ldr_event_id,
        "resolution": args.resolution,
        "preprocessing": {
            "preprocess_preset": args.preprocess_preset,
            "event_median_ksize": args.event_median_ksize,
            "event_close_ksize": args.event_close_ksize,
            "event_dilate_ksize": args.event_dilate_ksize,
            "event_blur_iters": args.event_blur_iters,
            "event_signed_median_ksize": args.event_signed_median_ksize,
            "geometry_median_ksize": args.geometry_median_ksize,
            "geometry_close_ksize": args.geometry_close_ksize,
            "geometry_dilate_ksize": args.geometry_dilate_ksize,
            "geometry_blur_iters": args.geometry_blur_iters,
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
    m = summary["metrics_mean"]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# first.md Event Cue Reliability Test\n\n")
        f.write(f"- Dataset root: `{args.root}`\n")
        f.write(f"- Active scenes: `{summary['active_scenes']}`\n")
        f.write(f"- Evaluated event frames: `{len(records)}`\n")
        f.write(f"- Preprocessing: `{summary['preprocessing']}`\n")
        f.write(f"- Per-frame CSV: `{csv_path.name}`\n")
        f.write(f"- Visualizations: `visuals/`\n\n")
        f.write("## Mean Metrics\n\n")
        for key in metric_keys:
            f.write(f"- `{key}`: {m[key]:.6f}\n")
        f.write("\n## Reading\n\n")
        f.write("- `corr_event_abs_* > shuffled` supports that event density aligns with GT geometry detail.\n")
        f.write("- `auc_event_abs_high_geom_detail > 0.5` means event density predicts high-curvature/detail regions.\n")
        f.write("- `f1_event_edge_geom_edge` measures overlap between event edges and geometry differential edges.\n")
        f.write("- `signed_curvature_acc_best > 0.5` means event polarity contains curvature-sign information up to a global contrast flip.\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
