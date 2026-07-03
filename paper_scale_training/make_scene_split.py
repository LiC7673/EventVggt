"""Create a deterministic, approximately material-stratified 60/12/12 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventvggt.datasets.my_event_dataset import get_combined_dataset


MATERIAL_WORDS = (
    "anodized", "copper", "gold", "silver", "metal", "chrome", "steel",
    "ceramic", "glass", "plastic", "diffuse", "rough", "mirror", "red",
    "blue", "green", "black", "white",
)


def _load_metadata(path):
    if not path:
        return {}
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(item["scene"]): item for item in data}
        return {str(key): value for key, value in data.items()}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["scene"]: row for row in csv.DictReader(handle)}


def _stratum(scene, metadata):
    item = metadata.get(scene, {})
    explicit = [str(item.get(key, "")).strip().lower() for key in ("material", "lighting", "motion")]
    explicit = [value for value in explicit if value]
    if explicit:
        return "|".join(explicit)
    lowered = scene.lower().replace("-", "_").replace(" ", "_")
    matched = [word for word in MATERIAL_WORDS if word in lowered]
    if matched:
        return "|".join(matched[:2])
    tokens = [token for token in lowered.split("_") if token]
    return "name:" + "_".join(tokens[-2:]) if tokens else "unknown"


def _assign_stratified(groups, capacities, seed):
    rng = np.random.default_rng(seed)
    assignments = {name: [] for name in capacities}
    remaining = dict(capacities)
    for key in sorted(groups, key=lambda value: (-len(groups[value]), value)):
        scenes = list(groups[key])
        rng.shuffle(scenes)
        for scene in scenes:
            candidates = [name for name, count in remaining.items() if count > 0]
            if not candidates:
                raise RuntimeError("Split capacities were exhausted before all scenes were assigned")
            # Prefer the split with the largest remaining fraction so every
            # material group follows the requested global proportions.
            scores = {name: remaining[name] / max(capacities[name], 1) for name in candidates}
            best_score = max(scores.values())
            best = sorted(name for name in candidates if scores[name] == best_score)
            chosen = best[int(rng.integers(0, len(best)))]
            assignments[chosen].append(scene)
            remaining[chosen] -= 1
    return assignments


def _explicit_assignments(scenes, metadata, capacities):
    declared = {
        scene: str(metadata.get(scene, {}).get("split", "")).strip().lower()
        for scene in scenes
    }
    if not any(declared.values()):
        return None
    valid = set(capacities)
    missing = [scene for scene, split in declared.items() if not split]
    invalid = {scene: split for scene, split in declared.items() if split and split not in valid}
    if missing or invalid:
        raise ValueError(
            "When metadata contains a split column, every valid scene must be assigned "
            f"to train/val/test. Missing={missing[:5]}, invalid={invalid}"
        )
    assignments = {
        split: sorted(scene for scene, assigned in declared.items() if assigned == split)
        for split in capacities
    }
    counts = {split: len(values) for split, values in assignments.items()}
    if counts != capacities:
        raise ValueError(f"Explicit metadata split must match {capacities}, got {counts}")
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output", default="abl_event_exp/paper_scale_60_12_12/scene_split.json")
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--val-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--metadata", default=None)
    args = parser.parse_args()

    dataset = get_combined_dataset(
        args.root,
        num_views=2,
        resolution=(518, 392),
        seed=args.seed,
        active_scene_count=1,
        split="all",
        test_frame_count=0,
        ldr_event_id="auto",
    )
    scenes = sorted(set(dataset.scenes))
    requested = args.train_count + args.val_count + args.test_count
    if len(scenes) != requested:
        raise RuntimeError(
            f"Expected exactly {requested} valid scenes for the requested split, found {len(scenes)}. "
            "Adjust --train-count/--val-count/--test-count after inspecting dataset.scenes."
        )

    metadata = _load_metadata(args.metadata)
    groups = defaultdict(list)
    strata = {}
    for scene in scenes:
        key = _stratum(scene, metadata)
        strata[scene] = key
        groups[key].append(scene)
    capacities = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    assignments = _explicit_assignments(scenes, metadata, capacities)
    split_strategy = "metadata_explicit" if assignments is not None else "automatic_stratified"
    if assignments is None:
        assignments = _assign_stratified(groups, capacities, args.seed)

    all_assigned = sum((values for values in assignments.values()), [])
    if len(all_assigned) != len(set(all_assigned)) or set(all_assigned) != set(scenes):
        raise RuntimeError("Scene split is not disjoint and exhaustive")
    fingerprint = hashlib.sha256("\n".join(sorted(scenes)).encode("utf-8")).hexdigest()
    output_data = {
        "root": str(Path(args.root)),
        "seed": args.seed,
        "dataset_scene_fingerprint": fingerprint,
        "counts": {key: len(value) for key, value in assignments.items()},
        "splits": assignments,
        "strata": strata,
        "split_strategy": split_strategy,
        "metadata_source": args.metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Valid scenes: {len(scenes)}")
    for split, values in assignments.items():
        print(f"{split}: {len(values)} scenes")
    print(f"Saved immutable scene split to {output}")


if __name__ == "__main__":
    main()
