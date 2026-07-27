"""Summarize pose metrics for exactly four scenes and five exposures."""
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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=r"E:\result\eventvgg\our\depth_pose_report.csv",
    )
    parser.add_argument("--condition", default="final_event_refined")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON. Default: <input_stem>_pose_summary.json",
    )
    return parser.parse_args()


def finite_float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {key!r} for scene={row.get('scene')!r}, "
            f"exposure={row.get('exposure')!r}: {row.get(key)!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite {key!r} for scene={row.get('scene')!r}, "
            f"exposure={row.get('exposure')!r}: {value}"
        )
    return value


def average(rows, key, weighted=False):
    values = [finite_float(row, key) for row in rows]
    if not weighted:
        return sum(values) / len(values)
    weights = [finite_float(row, "evaluated_batches") for row in rows]
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError("sum(evaluated_batches) must be positive")
    return sum(value * weight for value, weight in zip(values, weights)) / denominator


def main():
    args = arguments()
    source = Path(args.input).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    required = {
        "scope", "scene", "exposure", "condition", "evaluated_batches", *METRICS
    }
    if not raw:
        raise RuntimeError(f"empty CSV: {source}")
    missing_columns = sorted(required - set(raw[0]))
    if missing_columns:
        raise ValueError(f"CSV lacks required columns: {missing_columns}")

    selected = [
        row for row in raw
        if row["scope"] == "scene"
        and row["condition"] == args.condition
        and row["scene"] in SCENES
        and row["exposure"] in EXPOSURES
    ]
    expected_pairs = {(scene, exposure) for scene in SCENES for exposure in EXPOSURES}
    counts = {}
    for row in selected:
        key = (row["scene"], row["exposure"])
        counts[key] = counts.get(key, 0) + 1
    missing_pairs = sorted(expected_pairs - set(counts))
    duplicate_pairs = sorted(key for key, count in counts.items() if count != 1)
    unexpected_count = len(selected) != len(expected_pairs)
    if missing_pairs or duplicate_pairs or unexpected_count:
        raise RuntimeError(
            "expected exactly one final row for each of 4 scenes x 5 exposures; "
            f"selected={len(selected)}, missing={missing_pairs}, "
            f"duplicates={duplicate_pairs}"
        )

    by_exposure = {}
    for exposure in EXPOSURES:
        rows = [row for row in selected if row["exposure"] == exposure]
        by_exposure[exposure] = {
            "scene_count": len(rows),
            "macro_average": {
                metric: average(rows, metric, weighted=False) for metric in METRICS
            },
            "batch_weighted_average": {
                metric: average(rows, metric, weighted=True) for metric in METRICS
            },
        }
    summary = {
        "input": str(source.resolve()),
        "selection": {
            "scope": "scene",
            "condition": args.condition,
            "scenes": list(SCENES),
            "exposures": list(EXPOSURES),
            "selected_rows": len(selected),
        },
        "per_exposure": by_exposure,
        "all_4_scenes_x_5_exposures": {
            "macro_average": {
                metric: average(selected, metric, weighted=False)
                for metric in METRICS
            },
            "batch_weighted_average": {
                metric: average(selected, metric, weighted=True)
                for metric in METRICS
            },
        },
    }

    output = (
        Path(args.output).expanduser()
        if args.output
        else source.with_name(f"{source.stem}_pose_summary.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Exposure  ATE         RPE_trans   RPE_rot_deg")
    for exposure in EXPOSURES:
        values = by_exposure[exposure]["macro_average"]
        print(
            f"{exposure:<8}  {values['ate']:.9f}  "
            f"{values['rpe_trans']:.9f}  {values['rpe_rot_deg']:.9f}"
        )
    values = summary["all_4_scenes_x_5_exposures"]["macro_average"]
    print(
        f"{'AVERAGE':<8}  {values['ate']:.9f}  "
        f"{values['rpe_trans']:.9f}  {values['rpe_rot_deg']:.9f}"
    )
    print(
        "LaTeX: "
        f"& {values['ate']:.4f} & {values['rpe_trans']:.4f} "
        f"& {values['rpe_rot_deg']:.4f} \\\\"
    )
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
