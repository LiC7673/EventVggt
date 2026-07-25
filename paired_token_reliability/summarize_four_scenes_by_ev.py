"""Aggregate four-scene metrics by exposure from an existing metrics CSV.

Depth metrics are pooled using ``depth_pixels``:

* AbsRel and delta1: pixel-weighted arithmetic mean;
* RMSE and RMSElog: square, pixel-weight, pool, then take the square root.

Pose metrics are clip-level quantities, so their scene means are weighted by
``evaluated_batches``.  Only ``scope=scene`` rows are consumed; pre-existing
ALL/aggregate rows in the input CSV are deliberately ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
METRICS = (
    "abs_rel",
    "delta1",
    "rmse_log",
    "rmse",
    "ate",
    "rpe_trans",
    "rpe_rot_deg",
)
RMS_METRICS = {"rmse_log", "rmse"}
POSE_METRICS = {"ate", "rpe_trans", "rpe_rot_deg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=r"E:\result\eventvgg\hdreff\metrics.csv",
        help="Input per-scene metrics CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV. Default: <input-dir>/four_scenes_all_by_ev.csv",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument(
        "--exposures",
        nargs="+",
        default=["ev_0", "ev_1", "ev_2", "ev_5", "ev_10"],
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional exact experiment filter; otherwise every experiment is aggregated.",
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="Optional exact condition filter; otherwise every condition is aggregated.",
    )
    return parser.parse_args()


def finite_float(row, key):
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def weighted_metric(rows, key):
    weighted_sum = 0.0
    weight_sum = 0.0
    weight_key = "evaluated_batches" if key in POSE_METRICS else "depth_pixels"
    for row in rows:
        value = finite_float(row, key)
        weight = finite_float(row, weight_key)
        if value is None or weight is None or weight <= 0:
            continue
        if key in RMS_METRICS:
            value = value * value
        weighted_sum += weight * value
        weight_sum += weight
    if weight_sum <= 0:
        return float("nan")
    pooled = weighted_sum / weight_sum
    return math.sqrt(max(pooled, 0.0)) if key in RMS_METRICS else pooled


def aggregate_group(rows, scenes):
    present = {row["scene"] for row in rows}
    missing = [scene for scene in scenes if scene not in present]
    if missing:
        raise RuntimeError(f"Missing requested scene rows: {missing}")
    return {
        "scene": "ALL_4_SCENES",
        "scene_count": len(scenes),
        "evaluated_batches": sum(
            int(finite_float(row, "evaluated_batches") or 0) for row in rows
        ),
        "depth_pixels": sum(
            int(finite_float(row, "depth_pixels") or 0) for row in rows
        ),
        **{key: weighted_metric(rows, key) for key in METRICS},
    }


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "four_scenes_all_by_ev.csv"
    )
    json_path = (
        Path(args.json_output)
        if args.json_output
        else output_path.with_suffix(".json")
    )

    requested_scenes = list(dict.fromkeys(args.scenes))
    requested_exposures = list(dict.fromkeys(args.exposures))
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))

    selected = []
    for row in source_rows:
        if row.get("scope") != "scene":
            continue
        if row.get("scene") not in requested_scenes:
            continue
        if row.get("ldr_event_id") not in requested_exposures:
            continue
        if args.experiment is not None and row.get("experiment") != args.experiment:
            continue
        if args.condition is not None and row.get("condition") != args.condition:
            continue
        selected.append(row)
    if not selected:
        raise RuntimeError("No matching scene rows found in the input CSV")

    grouped = defaultdict(list)
    for row in selected:
        key = (
            row.get("experiment", ""),
            row.get("condition", ""),
            row["ldr_event_id"],
        )
        grouped[key].append(row)

    exposure_rank = {name: index for index, name in enumerate(requested_exposures)}
    result_rows = []
    for (experiment, condition, exposure), rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            exposure_rank.get(item[0][2], 10**6),
        ),
    ):
        aggregate = aggregate_group(rows, requested_scenes)
        result_rows.append(
            {
                "experiment": experiment,
                "condition": condition,
                "ldr_event_id": exposure,
                **aggregate,
            }
        )

    # Require every experiment/condition to contain every requested exposure.
    pairs = {(row["experiment"], row["condition"]) for row in result_rows}
    for pair in pairs:
        actual = {
            row["ldr_event_id"]
            for row in result_rows
            if (row["experiment"], row["condition"]) == pair
        }
        missing = [value for value in requested_exposures if value not in actual]
        if missing:
            raise RuntimeError(
                f"Missing exposures for experiment={pair[0]!r}, "
                f"condition={pair[1]!r}: {missing}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "experiment",
        "condition",
        "ldr_event_id",
        "scene",
        "scene_count",
        "evaluated_batches",
        "depth_pixels",
        *METRICS,
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
    json_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "scenes": requested_scenes,
                "aggregation": {
                    "abs_rel_delta1": "depth-pixel weighted",
                    "rmse_rmse_log": "pooled squared error, then square root",
                    "pose": "evaluated-batch weighted scene means",
                },
                "rows": result_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved CSV:  {output_path}")
    print(f"Saved JSON: {json_path}")
    for row in result_rows:
        print(
            f"{row['experiment']} | {row['condition']} | {row['ldr_event_id']} "
            f"AbsRel={row['abs_rel']:.6f} d1={row['delta1']:.6f} "
            f"RMSElog={row['rmse_log']:.6f} RMSE={row['rmse']:.6f} "
            f"ATE={row['ate']:.6f} RPE_t={row['rpe_trans']:.6f} "
            f"RPE_r={row['rpe_rot_deg']:.6f}"
        )


if __name__ == "__main__":
    main()
