"""Summarize four-scene RGB-finetuned pose metrics from per-EV CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
EXPOSURES = ("ev_0", "ev_1", "ev_2", "ev_5", "ev_10")
METRICS = ("ate", "rpe_trans", "rpe_rot_deg")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=r"E:\result\eventvgg\rgb_f")
    parser.add_argument("--condition", default="rgb_only_fixed_scale")
    parser.add_argument("--experiment", default="rgb_finetuned")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def finite_number(row, key, source):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {key} in {source}: scene={row.get('scene')!r}, "
            f"value={row.get(key)!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite {key} in {source}: scene={row.get('scene')!r}"
        )
    return value


def aggregate(rows, source, weighted):
    weights = (
        [finite_number(row, "evaluated_batches", source) for row in rows]
        if weighted
        else [1.0] * len(rows)
    )
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError(f"aggregate has no positive weight: {source}")
    return {
        metric: sum(
            finite_number(row, metric, source) * weight
            for row, weight in zip(rows, weights)
        ) / denominator
        for metric in METRICS
    }


def read_exposure(root, exposure, args):
    source = root / f"{exposure}.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    required = {
        "experiment", "scope", "scene", "ldr_event_id", "condition",
        "evaluated_batches", *METRICS,
    }
    if not all_rows:
        raise RuntimeError(f"empty CSV: {source}")
    missing_columns = sorted(required - set(all_rows[0]))
    if missing_columns:
        raise ValueError(f"{source} missing columns: {missing_columns}")

    rows = [
        row for row in all_rows
        if row["experiment"] == args.experiment
        and row["scope"] == "scene"
        and row["condition"] == args.condition
        and row["scene"] in SCENES
        and row["ldr_event_id"] == exposure
    ]
    counts = {scene: 0 for scene in SCENES}
    for row in rows:
        counts[row["scene"]] += 1
    bad = {scene: count for scene, count in counts.items() if count != 1}
    if len(rows) != len(SCENES) or bad:
        raise RuntimeError(
            f"{source}: expected one row per four scenes; "
            f"selected={len(rows)}, invalid_counts={bad}"
        )
    return source, rows


def main():
    args = parse_args()
    root = Path(args.input_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)

    per_exposure = {}
    per_scene = {}
    per_scene_exposure = {}
    all_rows = []
    inputs = []
    for exposure in EXPOSURES:
        source, rows = read_exposure(root, exposure, args)
        inputs.append(str(source.resolve()))
        all_rows.extend(rows)
        per_exposure[exposure] = {
            "scene_count": len(rows),
            "macro_average": aggregate(rows, source, weighted=False),
            "batch_weighted_average": aggregate(rows, source, weighted=True),
        }
        for row in rows:
            scene = row["scene"]
            per_scene_exposure.setdefault(scene, {})[exposure] = {
                metric: finite_number(row, metric, source)
                for metric in METRICS
            }

    for scene in SCENES:
        selected = [row for row in all_rows if row["scene"] == scene]
        per_scene[scene] = {
            "exposure_count": len(selected),
            "macro_average": aggregate(selected, root, weighted=False),
            "batch_weighted_average": aggregate(
                selected, root, weighted=True
            ),
        }

    payload = {
        "input_files": inputs,
        "selection": {
            "experiment": args.experiment,
            "scope": "scene",
            "condition": args.condition,
            "scenes": list(SCENES),
            "exposures": list(EXPOSURES),
            "selected_rows": len(all_rows),
        },
        "per_scene_exposure": per_scene_exposure,
        "per_scene": per_scene,
        "per_exposure": per_exposure,
        "all_4_scenes_x_5_exposures": {
            "macro_average": aggregate(all_rows, root, weighted=False),
            "batch_weighted_average": aggregate(
                all_rows, root, weighted=True
            ),
        },
    }
    output = (
        Path(args.output).expanduser()
        if args.output
        else root / "rgb_f_four_scene_pose_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Four scenes x five exposures (20 rows)")
    print("Scene                                        Exposure  ATE         RPE_trans   RPE_rot_deg")
    for scene in SCENES:
        for exposure in EXPOSURES:
            value = per_scene_exposure[scene][exposure]
            print(
                f"{scene:<44} {exposure:<8}  {value['ate']:.9f}  "
                f"{value['rpe_trans']:.9f}  {value['rpe_rot_deg']:.9f}"
            )

    print("\nAverage over five exposures for each scene")
    print("Scene                                        ATE         RPE_trans   RPE_rot_deg")
    for scene in SCENES:
        value = per_scene[scene]["macro_average"]
        print(
            f"{scene:<44} {value['ate']:.9f}  "
            f"{value['rpe_trans']:.9f}  {value['rpe_rot_deg']:.9f}"
        )

    print("\nAverage over four scenes for each exposure")
    print("Exposure  ATE         RPE_trans   RPE_rot_deg")
    for exposure in EXPOSURES:
        value = per_exposure[exposure]["macro_average"]
        print(
            f"{exposure:<8}  {value['ate']:.9f}  "
            f"{value['rpe_trans']:.9f}  {value['rpe_rot_deg']:.9f}"
        )
    value = payload["all_4_scenes_x_5_exposures"]["macro_average"]
    print(
        f"{'AVERAGE':<8}  {value['ate']:.9f}  "
        f"{value['rpe_trans']:.9f}  {value['rpe_rot_deg']:.9f}"
    )
    print(
        f"LaTeX: & {value['ate']:.4f} & {value['rpe_trans']:.4f} "
        f"& {value['rpe_rot_deg']:.4f} \\\\"
    )
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
