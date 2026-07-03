"""Evaluate checkpoint-best once on the 12 manifest test scenes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ablation.eag3r_metrics_eval as evaluator  # noqa: E402
from paper_main_ablation.evaluate_main_table import _build_main_table_model  # noqa: E402
from paper_scale_training.scene_split_loader import load_scene_split  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_scene_split(args.scene_manifest)
    test_scenes = list(manifest["splits"]["test"])
    if len(test_scenes) != 12:
        raise ValueError(f"Expected 12 held-out test scenes, got {len(test_scenes)}")

    forwarded = [
        "evaluate_heldout_test",
        "--checkpoint", args.checkpoint,
        "--name", "Full_60train_12val_12test",
        "--family", "event",
        "--out-dir", args.output_dir,
        "--root", args.root,
        "--split", "all",
        "--active-scene-count", "12",
        "--test-frame-count", "0",
        "--num-views", str(args.num_views),
        "--ldr-event-id", args.ldr_event_id,
        "--event-resize-bins", "10",
        "--batch-size", "1",
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--amp", "bf16",
        "--scene-names",
        *test_scenes,
    ]
    if args.max_batches is not None:
        forwarded.extend(["--max-batches", str(args.max_batches)])
    sys.argv = forwarded
    evaluator.build_model = _build_main_table_model
    evaluator.main()


if __name__ == "__main__":
    main()
