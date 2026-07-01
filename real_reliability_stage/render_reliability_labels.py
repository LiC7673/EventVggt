"""Render weak reliability labels from the original event stream.

This is stage 1 of the standalone reliability experiment. It uses the normal
EventVGGT dataloader and writes compact .npz samples:

    rgb, event_full, target_reliability, weight, components...

The target does not depend on Blender additive branches. It is a weak label
from real observable cues:

    event support * GT geometry detail * temporal non-persistence * image cues

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
    geo = _percentile_normalize(raw_geo, mask, percentile=args.geometry_percentile)
    if args.geometry_dilate_kernel > 1:
        kernel = np.ones((args.geometry_dilate_kernel, args.geometry_dilate_kernel), dtype=np.uint8)
        geo = cv2.dilate(geo, kernel, iterations=1)

    image_edge, saturation = _image_cues(rgb, mask)
    image_factor = args.image_support_floor + (1.0 - args.image_support_floor) * image_edge
    saturation_factor = np.clip(1.0 - args.saturation_reject * saturation, 0.0, 1.0)
    temporal_factor = event["temporal_score"]
    polarity_factor = args.polarity_floor + (1.0 - args.polarity_floor) * event["polarity_conf"]

    target = geo * image_factor * saturation_factor * temporal_factor * polarity_factor
    target = np.clip(target, 0.0, 1.0) * mask.astype(np.float32)
    event_support = event["event_support"] * mask.astype(np.float32)
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
        "geometry_support": geo[None].astype(np.float32),
        "temporal_score": temporal_factor[None].astype(np.float32),
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
        _gray_panel(sample["geometry_support"]),
        _gray_panel(sample["event_support"]),
        _gray_panel(sample["temporal_score"]),
        _gray_panel(sample["saturation"]),
        _gray_panel(sample["target_reliability"]),
        _gray_panel(sample["weight"]),
    ]
    labels = ["rgb", "event", "geo", "event_sup", "temporal", "sat", "target", "weight"]
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
    for batch_idx, views in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        for view_idx, view in enumerate(views):
            sample = _make_target(view, args)
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
    print(f"[render] split={split} scenes={dataset.get_active_scenes()} samples={len(manifest)} -> {manifest_path}")
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
    parser.add_argument("--geometry-dilate-kernel", type=int, default=7)
    parser.add_argument("--normal-detail-weight", type=float, default=0.7)
    parser.add_argument("--depth-detail-weight", type=float, default=0.3)
    parser.add_argument("--image-support-floor", type=float, default=0.35)
    parser.add_argument("--saturation-reject", type=float, default=0.7)
    parser.add_argument("--persistence-floor", type=float, default=0.20)
    parser.add_argument("--persistence-power", type=float, default=1.5)
    parser.add_argument("--polarity-floor", type=float, default=0.50)
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
