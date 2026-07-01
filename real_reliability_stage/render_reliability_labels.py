"""Render weak reliability labels from the original event stream.

This is stage 1 of the standalone reliability experiment. It uses the normal
EventVGGT dataloader and writes compact .npz samples:

    rgb, event_full, target_reliability, weight, components...

The target does not depend on Blender additive branches. By default it is a
weak geometry-alignment label: at pixels where an event exists, dilated GT
geometry detail is the reliability target. RGB and the complete temporal
voxel remain network inputs instead of being multiplied into the label.

The purpose is to train a ReliabilityNet before coupling it back to VGGT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset


def _as_numpy(value, sample_idx: int = 0):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] > sample_idx:
        return value[sample_idx]
    return value


def _safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _percentile_normalize(value: np.ndarray, mask: np.ndarray | None = None, percentile: float = 99.0) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    valid = value[np.asarray(mask, dtype=bool)] if mask is not None else value.reshape(-1)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size == 0:
        return np.zeros_like(value, dtype=np.float32)
    scale = max(float(np.percentile(valid, percentile)), 1e-6)
    return np.clip(value / scale, 0.0, 1.0).astype(np.float32)


def _sobel_mag_2d(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32, copy=False)
    dx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(np.maximum(dx * dx + dy * dy, 0.0)).astype(np.float32)


def _normal_detail(normal: np.ndarray) -> np.ndarray:
    if normal is None:
        return None
    normal = np.asarray(normal).astype(np.float32)
    if normal.ndim == 3 and normal.shape[0] == 3:
        normal = np.transpose(normal, (1, 2, 0))
    if normal.ndim != 3 or normal.shape[-1] != 3:
        return None
    if np.nanmax(np.abs(normal)) > 2.0:
        normal = normal / 127.5 - 1.0
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-6)
    detail = np.zeros(normal.shape[:2], dtype=np.float32)
    for channel in range(3):
        detail += _sobel_mag_2d(normal[..., channel])
    return detail


def _depth_detail(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth).astype(np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 1e-6)
    safe = np.where(valid, depth, 0.0)
    positive = safe[valid]
    if positive.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    floor = max(float(np.percentile(positive, 1.0)), 1e-6)
    log_depth = np.log(np.maximum(safe, floor))
    return _sobel_mag_2d(log_depth) * valid.astype(np.float32)


def _image_cues(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb).astype(np.float32)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.min() < -0.05:
        rgb01 = np.clip((rgb + 1.0) * 0.5, 0.0, 1.0)
    else:
        rgb01 = np.clip(rgb, 0.0, 1.0)
    gray = rgb01.mean(axis=-1)
    edge = _percentile_normalize(_sobel_mag_2d(gray), mask, percentile=99.0)
    saturation = ((rgb01 > 0.97).mean(axis=-1) + (rgb01 < 0.03).mean(axis=-1)).clip(0.0, 1.0)
    saturation = cv2.blur(saturation.astype(np.float32), (5, 5)) * mask.astype(np.float32)
    return edge.astype(np.float32), saturation.astype(np.float32)


def _event_cues(voxel: np.ndarray, *, threshold: float, persistence_floor: float, persistence_power: float):
    voxel = np.asarray(voxel).astype(np.float32)
    channels = voxel.shape[0]
    bins = channels // 2
    if bins <= 0:
        raise ValueError(f"Expected polarity-separated event voxel, got shape={voxel.shape}")
    pos = np.clip(voxel[:bins], 0.0, None)
    neg = np.clip(voxel[bins : 2 * bins], 0.0, None)
    per_bin = pos + neg
    energy = np.log1p(per_bin.sum(axis=0))
    support = _percentile_normalize(energy, percentile=99.5)
    active = per_bin > max(float(threshold), 0.0)
    persistence = active.mean(axis=0).astype(np.float32)
    temporal_score = persistence_floor + (1.0 - persistence_floor) * np.power(
        np.clip(1.0 - persistence, 0.0, 1.0),
        persistence_power,
    )
    polarity_conf = np.abs(pos.sum(axis=0) - neg.sum(axis=0)) / (per_bin.sum(axis=0) + 1e-6)
    return {
        "event_support": support.astype(np.float32),
        "event_energy": energy.astype(np.float32),
        "persistence": persistence.astype(np.float32),
        "temporal_score": temporal_score.astype(np.float32),
        "polarity_conf": polarity_conf.astype(np.float32),
    }


def _make_target(view: Dict, args) -> Dict[str, np.ndarray]:
    rgb = _as_numpy(view["img"])
    voxel = _as_numpy(view["event_voxel"])
    depth = _as_numpy(view["depthmap"])
    mask = _as_numpy(view.get("mask", np.ones_like(depth, dtype=bool))).astype(bool)
    normal = _as_numpy(view["normal_gt"]) if "normal_gt" in view else None

    event = _event_cues(
        voxel,
        threshold=args.event_bin_threshold,
        persistence_floor=args.persistence_floor,
        persistence_power=args.persistence_power,
    )
    normal_detail = _normal_detail(normal)
    depth_detail = _depth_detail(depth, mask)
    if normal_detail is None or np.max(normal_detail) <= 0:
        raw_geo = depth_detail
    else:
        raw_geo = args.normal_detail_weight * normal_detail + args.depth_detail_weight * depth_detail
    mask_u8 = mask.astype(np.uint8)
    if args.geometry_interior_erode > 1:
        erode_kernel = np.ones(
            (args.geometry_interior_erode, args.geometry_interior_erode), dtype=np.uint8
        )
        interior_mask = cv2.erode(mask_u8, erode_kernel, iterations=1).astype(bool)
    else:
        interior_mask = mask

    # Keep smooth-surface detail separate from the foreground/background jump;
    # otherwise the silhouette dominates percentile normalization.
    geo_core = _percentile_normalize(raw_geo, interior_mask, percentile=args.geometry_percentile)
    geo_core = geo_core * interior_mask.astype(np.float32)

    contour_kernel = np.ones((3, 3), dtype=np.uint8)
    silhouette = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, contour_kernel).astype(np.float32)
    geo = np.maximum(geo_core, args.geometry_silhouette_weight * silhouette)
    if args.geometry_dilate_kernel > 1:
        kernel = np.ones((args.geometry_dilate_kernel, args.geometry_dilate_kernel), dtype=np.uint8)
        geo_halo = cv2.dilate(geo, kernel, iterations=1)
        # Dilation only compensates a small event/geometry misregistration. A
        # full-strength max dilation turns thin normal/depth detail into broad
        # bands and teaches the reliability net an over-smoothed target.
        geo = np.maximum(geo, args.geometry_dilate_gain * geo_halo)
    geo = np.clip(geo, 0.0, 1.0) * mask.astype(np.float32)

    image_edge, saturation = _image_cues(rgb, mask)
    # RGB edges are supporting evidence, not a prerequisite. In particular,
    # saturated LDR regions are exactly where events are expected to help, so
    # saturation must not erase an otherwise geometry-aligned event target.
    usable_image_edge = image_edge * (1.0 - saturation)
    image_factor = args.image_support_floor + (1.0 - args.image_support_floor) * usable_image_edge
    saturation_factor = np.clip(1.0 - args.saturation_reject * saturation, 0.0, 1.0)
    temporal_factor = event["temporal_score"]
    polarity_factor = args.polarity_floor + (1.0 - args.polarity_floor) * event["polarity_conf"]

    event_support = event["event_support"] * mask.astype(np.float32)
    event_present = event_support >= args.event_support_min
    if args.target_mode == "geometry":
        # Image appearance and event timing are evidence available to the
        # network. Multiplying them into the target causes a trivial all-zero
        # solution in saturated scenes, so the weak label only asks whether an
        # observed event lies on GT geometry detail.
        target = geo
    else:
        if args.cue_fusion == "product":
            cue_factor = image_factor * saturation_factor * temporal_factor * polarity_factor
        else:
            cue_product = image_factor * saturation_factor * temporal_factor * polarity_factor
            cue_factor = np.power(np.clip(cue_product, 0.0, 1.0), 0.25)
        target = geo * cue_factor
    target = np.clip(target, 0.0, 1.0) * event_present.astype(np.float32) * mask.astype(np.float32)
    # Only event pixels strongly supervise reliability, but keep a weak empty
    # background term so the net does not output dense reliability everywhere.
    weight = args.empty_weight + (1.0 - args.empty_weight) * event_support
    weight = weight * mask.astype(np.float32)

    return {
        "rgb": rgb.astype(np.float32),
        "event_full": voxel.astype(np.float32),
        "target_reliability": target[None].astype(np.float32),
        "weight": weight[None].astype(np.float32),
        "event_support": event_support[None].astype(np.float32),
        "geometry_core": geo_core[None].astype(np.float32),
        "geometry_support": geo[None].astype(np.float32),
        "temporal_score": temporal_factor[None].astype(np.float32),
        "temporal_event_score": (temporal_factor * event_support)[None].astype(np.float32),
        "persistence": event["persistence"][None].astype(np.float32),
        "image_edge": image_edge[None].astype(np.float32),
        "saturation": saturation[None].astype(np.float32),
        "mask": mask[None].astype(np.float32),
    }


def _to_rgb01(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.min() < -0.05:
        rgb = (rgb + 1.0) * 0.5
    return np.clip(rgb, 0.0, 1.0)


def _gray_panel(value: np.ndarray) -> np.ndarray:
    value = np.squeeze(value)
    value = _percentile_normalize(value, percentile=99.5)
    return np.repeat((value * 255.0).round().astype(np.uint8)[..., None], 3, axis=-1)


def _event_panel(voxel: np.ndarray) -> np.ndarray:
    channels = voxel.shape[0]
    bins = channels // 2
    pos = np.log1p(np.clip(voxel[:bins], 0.0, None).sum(axis=0))
    neg = np.log1p(np.clip(voxel[bins : 2 * bins], 0.0, None).sum(axis=0))
    scale = max(float(np.percentile(np.concatenate([pos.reshape(-1), neg.reshape(-1)]), 99.5)), 1e-6)
    out = np.zeros((*pos.shape, 3), dtype=np.float32)
    out[..., 0] = np.clip(pos / scale, 0.0, 1.0)
    out[..., 2] = np.clip(neg / scale, 0.0, 1.0)
    out[..., 1] = 0.25 * np.minimum(out[..., 0], out[..., 2])
    return (out * 255.0).round().astype(np.uint8)


def _save_preview(path: Path, sample: Dict[str, np.ndarray]) -> None:
    panels = [
        (_to_rgb01(sample["rgb"]) * 255.0).round().astype(np.uint8),
        _event_panel(sample["event_full"]),
        _gray_panel(sample["geometry_core"]),
        _gray_panel(sample["geometry_support"]),
        _gray_panel(sample["event_support"]),
        _gray_panel(sample["temporal_event_score"]),
        _gray_panel(sample["saturation"]),
        _gray_panel(sample["target_reliability"]),
        _gray_panel(sample["weight"]),
    ]
    labels = [
        "rgb",
        "event",
        "geo_core",
        "geo_support",
        "event_sup",
        "temp@event",
        "sat",
        "target",
        "weight",
    ]
    height, width = panels[0].shape[:2]
    title_h = 20
    canvas = np.zeros((height + title_h, width * len(panels), 3), dtype=np.uint8)
    for idx, (label, panel) in enumerate(zip(labels, panels)):
        x = idx * width
        canvas[title_h:, x : x + width] = panel
        cv2.putText(canvas, label, (x + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def _build_loader(args, split: str):
    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=None,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.active_scene_count,
        split=split,
        test_frame_count=args.test_frame_count,
        ldr_event_id=args.ldr_event_id,
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_bins,
        return_normal_gt=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    return dataset, loader


def _iter_splits(args) -> Iterable[str]:
    if args.splits == "both":
        return ("train", "test")
    return (args.splits,)


def render_split(args, split: str) -> List[Dict[str, str]]:
    dataset, loader = _build_loader(args, split)
    out_root = Path(args.output_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.output_dir) / "preview" / split
    manifest = []
    preview_count = 0
    stats = {
        "event_pixels": 0,
        "target_sum_on_event": 0.0,
        "target_positive_on_event": 0,
        "geometry_sum_on_event": 0.0,
        "valid_pixels": 0,
    }
    for batch_idx, views in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        for view_idx, view in enumerate(views):
            sample = _make_target(view, args)
            event_support = np.squeeze(sample["event_support"])
            target = np.squeeze(sample["target_reliability"])
            geometry = np.squeeze(sample["geometry_support"])
            valid = np.squeeze(sample["mask"]) > 0
            event_pixels = (event_support >= args.event_support_min) & valid
            stats["event_pixels"] += int(event_pixels.sum())
            stats["valid_pixels"] += int(valid.sum())
            stats["target_sum_on_event"] += float(target[event_pixels].sum())
            stats["target_positive_on_event"] += int((target[event_pixels] >= 0.5).sum())
            stats["geometry_sum_on_event"] += float(geometry[event_pixels].sum())
            label_value = view.get("label", [f"batch{batch_idx:05d}_view{view_idx:02d}"])
            label = label_value[0] if isinstance(label_value, (list, tuple)) else str(label_value)
            instance_value = view.get("instance", [label])
            instance = instance_value[0] if isinstance(instance_value, (list, tuple)) else str(instance_value)
            filename = f"{batch_idx:06d}_v{view_idx:02d}_{_safe_name(label)}.npz"
            path = out_root / filename
            np.savez_compressed(path, **sample)
            manifest.append({"path": str(path), "label": str(label), "instance": str(instance), "view_idx": view_idx})
            if preview_count < args.preview_count:
                _save_preview(preview_dir / f"{Path(filename).stem}.png", sample)
                preview_count += 1
    manifest_path = Path(args.output_dir) / f"manifest_{split}.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": split,
                "root": args.root,
                "scenes": dataset.get_active_scenes(),
                "num_samples": len(manifest),
                "items": manifest,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    event_count = max(stats["event_pixels"], 1)
    valid_count = max(stats["valid_pixels"], 1)
    summary = {
        "target_mode": args.target_mode,
        "event_pixel_ratio": stats["event_pixels"] / valid_count,
        "target_mean_on_event": stats["target_sum_on_event"] / event_count,
        "target_positive_ratio_on_event": stats["target_positive_on_event"] / event_count,
        "geometry_mean_on_event": stats["geometry_sum_on_event"] / event_count,
        **stats,
    }
    stats_path = Path(args.output_dir) / f"label_stats_{split}.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"[render] split={split} scenes={dataset.get_active_scenes()} samples={len(manifest)} "
        f"event_ratio={summary['event_pixel_ratio']:.4f} "
        f"target_mean={summary['target_mean_on_event']:.4f} "
        f"target_pos={summary['target_positive_ratio_on_event']:.4f} -> {manifest_path}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="abl_event_exp/real_reliability_stage/labels")
    parser.add_argument("--splits", choices=["train", "test", "both"], default="both")
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=12)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-bins", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--event-bin-threshold", type=float, default=1.0e-5)
    parser.add_argument("--geometry-percentile", type=float, default=99.0)
    parser.add_argument("--geometry-interior-erode", type=int, default=3)
    parser.add_argument("--geometry-silhouette-weight", type=float, default=0.45)
    parser.add_argument("--geometry-dilate-kernel", type=int, default=5)
    parser.add_argument("--geometry-dilate-gain", type=float, default=0.25)
    parser.add_argument("--normal-detail-weight", type=float, default=0.7)
    parser.add_argument("--depth-detail-weight", type=float, default=0.3)
    parser.add_argument("--image-support-floor", type=float, default=0.70)
    parser.add_argument("--saturation-reject", type=float, default=0.0)
    parser.add_argument("--persistence-floor", type=float, default=0.50)
    parser.add_argument("--persistence-power", type=float, default=1.5)
    parser.add_argument("--polarity-floor", type=float, default=0.70)
    parser.add_argument("--event-support-min", type=float, default=0.01)
    parser.add_argument("--target-mode", choices=("geometry", "cue_modulated"), default="geometry")
    parser.add_argument("--cue-fusion", choices=("geometric", "product"), default="geometric")
    parser.add_argument("--empty-weight", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "render_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    for split in _iter_splits(args):
        render_split(args, split)


if __name__ == "__main__":
    main()
