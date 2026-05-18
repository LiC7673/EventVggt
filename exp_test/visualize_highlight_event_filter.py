"""Visualize persistent-highlight event filtering.

This is the quick visual counterpart of highlight_event_filter_test.py. It
loads the dataset, builds the long-window temporal-density highlight mask, and
saves raw/filtered event panels for a few event-bearing frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eventvggt.datasets.my_event_dataset import get_combined_dataset
from exp_test.highlight_event_filter_test import compute_frame, save_filter_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize highlight event filtering")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/highlight_event_filter_visuals")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--num-samples", type=int, default=8)
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
    parser.add_argument("--max-frames", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    manifest: List[Dict[str, object]] = []
    saved = 0
    for sample_idx in range(min(args.num_samples, len(dataset))):
        views = dataset[sample_idx]
        for frame_idx, view in enumerate(views):
            if not bool(np.asarray(view.get("has_event", frame_idx > 0))):
                continue
            metrics, maps = compute_frame(view, rng, args)
            label = f"sample_{sample_idx:04d}_frame_{frame_idx:02d}"
            save_filter_visualization(out_dir, maps, label)
            manifest.append(
                {
                    "file": f"{label}.png",
                    "sample_idx": sample_idx,
                    "frame_idx": frame_idx,
                    "label": str(view.get("label", "")),
                    "highlight_mask_ratio": metrics["highlight_mask_ratio"],
                    "highlight_removed_energy_ratio": metrics["highlight_removed_energy_ratio"],
                    "raw_corr_geom": metrics["raw_corr_event_abs_geom_detail"],
                    "filtered_corr_geom": metrics["filtered_corr_event_abs_geom_detail"],
                    "raw_auc": metrics["raw_auc_event_abs_high_geom_detail"],
                    "filtered_auc": metrics["filtered_auc_event_abs_high_geom_detail"],
                }
            )
            saved += 1
            if saved >= args.max_frames:
                break
        if saved >= args.max_frames:
            break

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": args.root,
                "active_scenes": dataset.get_active_scenes(),
                "filter": {
                    "bin_active_percentile": args.bin_active_percentile,
                    "min_occupancy": args.min_occupancy,
                    "min_density": args.min_density,
                    "highlight_percentile": args.highlight_percentile,
                    "highlight_keep": args.highlight_keep,
                    "mask_close_ksize": args.mask_close_ksize,
                    "mask_dilate_ksize": args.mask_dilate_ksize,
                },
                "frames": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {saved} visualizations to {out_dir}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
