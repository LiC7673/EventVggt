"""Select four held-out scenes shared by every requested LDR level."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventvggt.datasets.my_event_dataset import get_combined_dataset  # noqa: E402


def _format_ldr(value):
    value = str(value)
    return value if value.startswith("ev_") else f"ev_{value}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-scene-count", type=int, default=12)
    parser.add_argument("--test-scene-count", type=int, default=4)
    parser.add_argument("--train-ldr-levels", nargs="+", default=["2", "5", "10"])
    parser.add_argument("--ldr-levels", nargs="+", default=["1", "2", "5", "10"])
    parser.add_argument("--num-views", type=int, default=4)
    args = parser.parse_args()

    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=(518, 392),
        seed=0,
        active_scene_count=1,
        split="all",
        test_frame_count=0,
        ldr_event_id="random",
    )
    train_required = {_format_ldr(value) for value in args.train_ldr_levels}
    required = {_format_ldr(value) for value in args.ldr_levels}
    ordered = list(dataset.scenes)
    training_scenes = [
        scene
        for scene in ordered
        if train_required.issubset(
            set(dataset.scene_records[scene]["available_ldr_events"])
        )
    ][: args.train_scene_count]
    if len(training_scenes) != args.train_scene_count:
        raise RuntimeError(
            f"Need {args.train_scene_count} training scenes common to {sorted(train_required)}, "
            f"found {len(training_scenes)}"
        )
    eligible = []
    rejected = {}
    for scene in ordered:
        if scene in training_scenes:
            continue
        available = set(dataset.scene_records[scene]["available_ldr_events"])
        missing = sorted(required - available)
        if missing:
            rejected[scene] = missing
        else:
            eligible.append(scene)
    selected = eligible[: args.test_scene_count]
    if len(selected) != args.test_scene_count:
        raise RuntimeError(
            f"Need {args.test_scene_count} held-out scenes common to {sorted(required)}, "
            f"found {len(eligible)}. Missing by scene: {rejected}"
        )
    output = {
        "root": args.root,
        "training_scenes": training_scenes,
        "heldout_scenes": selected,
        "train_ldr_levels": sorted(train_required),
        "required_ldr_levels": sorted(required),
        "selection": "first eligible scenes after the fixed training prefix",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Held-out scenes: {selected}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
