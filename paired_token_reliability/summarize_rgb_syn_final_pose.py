"""Summarize RGB syn_final pose metrics for four scenes and five EV levels."""
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
    parser.add_argument(
        "--input", default=r"E:\result\eventvgg\hdreff\syn_final.csv"
    )
    parser.add_argument("--condition", default="rgb_only_fixed_scale")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {key}: scene={row.get('scene')!r}, "
            f"EV={row.get('ldr_event_id')!r}, value={row.get(key)!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite {key}: scene={row.get('scene')!r}, "
            f"EV={row.get('ldr_event_id')!r}"
        )
    return value


def aggregate(rows, weighted):
    result = {}
    weights = (
        [number(row, "evaluated_batches") for row in rows]
        if weighted else [1.0] * len(rows)
    )
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError("aggregate has no positive weight")
    for metric in METRICS:
        result[metric] = sum(
            number(row, metric) * weight
            for row, weight in zip(rows, weights)
        ) / denominator
    return result


def main():
    args = parse_args()
    source = Path(args.input).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    required = {
        "scope", "scene", "ldr_event_id", "condition", "evaluated_batches",
        *METRICS,
    }
    if not all_rows:
        raise RuntimeError(f"empty CSV: {source}")
    missing_columns = sorted(required - set(all_rows[0]))
    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")

    rows = [
        row for row in all_rows
        if row["scope"] == "scene"
        and row["condition"] == args.condition
        and row["scene"] in SCENES
        and row["ldr_event_id"] in EXPOSURES
    ]
    expected = {(scene, exposure) for scene in SCENES for exposure in EXPOSURES}
    counts = {}
    for row in rows:
        key = (row["scene"], row["ldr_event_id"])
        counts[key] = counts.get(key, 0) + 1
    missing = sorted(expected - set(counts))
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if len(rows) != 20 or missing or duplicates:
        raise RuntimeError(
            f"expected exactly 20 rows; selected={len(rows)}, "
            f"missing={missing}, duplicates={duplicates}"
        )

    per_exposure = {}
    for exposure in EXPOSURES:
        selected = [row for row in rows if row["ldr_event_id"] == exposure]
        per_exposure[exposure] = {
            "scene_count": len(selected),
            "macro_average": aggregate(selected, weighted=False),
            "batch_weighted_average": aggregate(selected, weighted=True),
        }
    payload = {
        "input": str(source.resolve()),
        "selection": {
            "scope": "scene",
            "condition": args.condition,
            "scenes": list(SCENES),
            "exposures": list(EXPOSURES),
            "selected_rows": len(rows),
        },
        "per_exposure": per_exposure,
        "all_4_scenes_x_5_exposures": {
            "macro_average": aggregate(rows, weighted=False),
            "batch_weighted_average": aggregate(rows, weighted=True),
        },
    }
    output = (
        Path(args.output).expanduser()
        if args.output
        else source.with_name(f"{source.stem}_four_scene_pose_summary.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
