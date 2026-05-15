"""Debug event loading stages for MyEventDataset.

Reports event counts before resize, after resize, and after the dataset mask
filter. This helps diagnose black event visualizations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eventvggt.datasets.my_event_dataset import get_combined_dataset


EPS = 1e-6


def xy_stats(event_xy: np.ndarray) -> Dict[str, object]:
    if event_xy.size == 0:
        return {
            "count": 0,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
        }
    return {
        "count": int(event_xy.shape[0]),
        "x_min": int(event_xy[:, 0].min()),
        "x_max": int(event_xy[:, 0].max()),
        "y_min": int(event_xy[:, 1].min()),
        "y_max": int(event_xy[:, 1].max()),
    }


def time_stats(event_t: np.ndarray) -> Dict[str, object]:
    if event_t.size == 0:
        return {"t_min": None, "t_max": None, "duration": 0.0}
    return {
        "t_min": float(event_t.min()),
        "t_max": float(event_t.max()),
        "duration": float(event_t.max() - event_t.min()),
    }


def polarity_stats(event_p: np.ndarray) -> Dict[str, int]:
    if event_p.size == 0:
        return {"positive": 0, "negative": 0, "zero": 0}
    return {
        "positive": int((event_p > 0).sum()),
        "negative": int((event_p < 0).sum()),
        "zero": int((event_p == 0).sum()),
    }


def render_events(event_xy: np.ndarray, event_p: np.ndarray, width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.float32)
    if event_xy.size == 0:
        return img.astype(np.uint8)
    x = np.clip(event_xy[:, 0].astype(np.int64), 0, width - 1)
    y = np.clip(event_xy[:, 1].astype(np.int64), 0, height - 1)
    flat = y * width + x
    pos = np.zeros(height * width, dtype=np.float32)
    neg = np.zeros(height * width, dtype=np.float32)
    pos_mask = event_p > 0
    np.add.at(pos, flat[pos_mask], 1.0)
    np.add.at(neg, flat[~pos_mask], 1.0)
    pos = np.log1p(pos.reshape(height, width))
    neg = np.log1p(neg.reshape(height, width))
    scale = max(float(pos.max()), float(neg.max()), EPS)
    img[..., 0] = pos / scale
    img[..., 1] = 0.25 * np.clip((pos + neg) / scale, 0.0, 1.0)
    img[..., 2] = neg / scale
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def save_stage_visuals(out_dir: Path, record: Dict, raw_event, resized_event, masked_event, resized) -> None:
    width, height = [int(v) for v in resized["dst_resolution"]]
    stages = [
        ("raw_xy_as_dst", raw_event),
        ("resized_before_mask", resized_event),
        ("after_mask", masked_event),
    ]
    for stage_name, event in stages:
        img = render_events(event["event_xy"], event["event_p"], width, height)
        Image.fromarray(img).save(
            out_dir / f"sample_{record['sample_idx']:04d}_frame_{record['local_frame_idx']:02d}_{stage_name}.png"
        )


def inspect_h5(h5_path: str) -> Dict[str, object]:
    with h5py.File(h5_path, "r") as h5:
        events = h5["events"]
        total = len(events)
        if total == 0:
            return {"total_events": 0}
        first = events[0]
        last = events[-1]
        step = max(1, total // 1000)
        sampled = events[::step][:1000]
        return {
            "total_events": int(total),
            "first_event": [float(x) for x in first],
            "last_event": [float(x) for x in last],
            "col_min": [float(sampled[:, col].min()) for col in range(sampled.shape[1])],
            "col_max": [float(sampled[:, col].max()) for col in range(sampled.shape[1])],
        }


def analyze_frame(dataset, sample_idx: int, local_frame_idx: int, out_dir: Path, args) -> Dict:
    scene_name, start_id = dataset.start_img_ids[sample_idx]
    scene_meta = dataset.active_scene_data[scene_name]
    frame_idx = start_id + local_frame_idx
    ldr_event_id = dataset._select_ldr_event(scene_meta, np.random.default_rng(args.seed), None)
    resized = dataset._load_view_data(
        scene_meta,
        frame_idx,
        tuple(args.resolution),
        ldr_event_id=ldr_event_id,
    )

    event_start, event_end = scene_meta["frame_event_index"][frame_idx]
    raw_event = dataset.load_event_slice(
        scene_meta["event_h5"],
        event_start,
        event_end,
        event_columns=scene_meta.get("event_columns"),
        time_origin=scene_meta.get("event_time_info", {}).get("origin", 0.0),
    )
    event_src_resolution = scene_meta.get("event_resolution", resized["src_resolution"])
    if np.asarray(event_src_resolution).reshape(-1).size < 2 or np.any(np.asarray(event_src_resolution) <= 0):
        event_src_resolution = resized["src_resolution"]
    event_spatial_transform = dataset._resolve_event_spatial_transform(scene_meta)
    event_y_flip = event_spatial_transform == "vflip"
    resized_event = dataset._resize_event_data(
        raw_event,
        src_resolution=event_src_resolution,
        dst_resolution=resized["dst_resolution"],
        spatial_transform=event_spatial_transform,
    )
    masked_event = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in resized_event.items()}
    mask_true_ratio = None
    kept_by_mask = None
    if "mask" in resized and resized_event["event_xy"].size > 0:
        mask = resized["mask"].astype(bool)
        xy = resized_event["event_xy"]
        valid_events = mask[xy[:, 1], xy[:, 0]]
        kept_by_mask = int(valid_events.sum())
        mask_true_ratio = float(mask.mean())
        for key in ("event_xy", "event_t", "event_p", "events"):
            masked_event[key] = masked_event[key][valid_events]
    elif "mask" in resized:
        mask_true_ratio = float(resized["mask"].astype(bool).mean())
        kept_by_mask = 0

    record = {
        "sample_idx": int(sample_idx),
        "local_frame_idx": int(local_frame_idx),
        "absolute_frame_idx": int(frame_idx),
        "scene": scene_name,
        "ldr_event_id": ldr_event_id,
        "event_h5": scene_meta["event_h5"],
        "event_time_info": scene_meta.get("event_time_info", {}),
        "event_index_range": [int(event_start), int(event_end)],
        "event_index_count": int(event_end - event_start),
        "image_src_resolution": [int(x) for x in resized["src_resolution"]],
        "event_src_resolution": [int(x) for x in event_src_resolution],
        "event_spatial_transform": event_spatial_transform,
        "event_y_flip": bool(event_y_flip),
        "dst_resolution": [int(x) for x in resized["dst_resolution"]],
        "mask_true_ratio": mask_true_ratio,
        "mask_kept_events": kept_by_mask,
        "raw": {
            **xy_stats(raw_event["event_xy"]),
            **time_stats(raw_event["event_t"]),
            **polarity_stats(raw_event["event_p"]),
        },
        "resized_before_mask": {
            **xy_stats(resized_event["event_xy"]),
            **time_stats(resized_event["event_t"]),
            **polarity_stats(resized_event["event_p"]),
        },
        "after_mask": {
            **xy_stats(masked_event["event_xy"]),
            **time_stats(masked_event["event_t"]),
            **polarity_stats(masked_event["event_p"]),
        },
    }
    save_stage_visuals(out_dir, record, raw_event, resized_event, masked_event, resized)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug dataset event loading")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/debug_event_loading")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--event-spatial-transform", default="auto", choices=["auto", "none", "hflip", "vflip", "rot180", "hflip_rot180"])
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=3)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        event_spatial_transform=args.event_spatial_transform,
    )
    records = []
    h5_seen = {}
    for offset in range(args.num_samples):
        sample_idx = min(args.sample_idx + offset, len(dataset) - 1)
        scene_name, _ = dataset.start_img_ids[sample_idx]
        h5_path = dataset.active_scene_data[scene_name]["event_h5"]
        if h5_path not in h5_seen:
            h5_seen[h5_path] = inspect_h5(h5_path)
        frame_indices = args.frames if args.frames is not None else list(range(args.num_views))
        for local_frame_idx in frame_indices:
            if local_frame_idx < 0 or local_frame_idx >= args.num_views:
                continue
            records.append(analyze_frame(dataset, sample_idx, local_frame_idx, out_dir, args))

    summary = {
        "root": args.root,
        "active_scenes": dataset.get_active_scenes(),
        "h5_stats": h5_seen,
        "records": records,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved debug images and summary to {out_dir}")


if __name__ == "__main__":
    main()
