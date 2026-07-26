"""Aggregate five per-exposure RGB-finetuned CSV files into one JSON.

Expected input files:

    ev_0.csv, ev_1.csv, ev_2.csv, ev_5.csv, ev_10.csv

Only the requested four ``scope=scene`` rows are used. Existing aggregate rows
inside the CSV files are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


EXPOSURES = ("ev_0", "ev_1", "ev_2", "ev_5", "ev_10")
SCENES = (
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
        "--input-dir",
        default=r"E:\result\eventvgg\rgb_no_f",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <input-dir>/four_scenes_metrics_by_ev.json",
    )
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--experiment", default="rgb_finetuned")
    parser.add_argument("--condition", default="rgb_only_fixed_scale")
    return parser.parse_args()


def finite_float(row, key):
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def pool_metric(rows, key):
    weight_key = "evaluated_batches" if key in POSE_METRICS else "depth_pixels"
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = finite_float(row, key)
        weight = finite_float(row, weight_key)
        if value is None or weight is None or weight <= 0:
            continue
        if key in RMS_METRICS:
            value = value * value
        numerator += weight * value
        denominator += weight
    if denominator <= 0:
        return float("nan")
    pooled = numerator / denominator
    return math.sqrt(max(pooled, 0.0)) if key in RMS_METRICS else pooled


def load_exposure(path, exposure, scenes, experiment, condition):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    rows = [
        row
        for row in source
        if row.get("scope") == "scene"
        and row.get("scene") in scenes
        and row.get("ldr_event_id") == exposure
        and (experiment is None or row.get("experiment") == experiment)
        and (condition is None or row.get("condition") == condition)
    ]
    present = {row.get("scene") for row in rows}
    missing = [scene for scene in scenes if scene not in present]
    if missing:
        raise RuntimeError(f"{path.name}: missing four-scene rows: {missing}")
    duplicates = {
        scene: sum(row.get("scene") == scene for row in rows)
        for scene in scenes
    }
    duplicates = {scene: count for scene, count in duplicates.items() if count != 1}
    if duplicates:
        raise RuntimeError(f"{path.name}: expected one row per scene, got {duplicates}")

    return {
        "source_csv": str(path),
        "scene_count": len(scenes),
        "evaluated_batches": sum(
            int(finite_float(row, "evaluated_batches") or 0) for row in rows
        ),
        "depth_pixels": sum(
            int(finite_float(row, "depth_pixels") or 0) for row in rows
        ),
        "metrics": {key: pool_metric(rows, key) for key in METRICS},
        "per_scene": {
            row["scene"]: {
                key: finite_float(row, key)
                for key in METRICS
            }
            for row in rows
        },
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    output = (
        Path(args.output)
        if args.output
        else input_dir / "four_scenes_metrics_by_ev.json"
    )
    scenes = list(dict.fromkeys(args.scenes))

    by_exposure = {
        exposure: load_exposure(
            input_dir / f"{exposure}.csv",
            exposure,
            scenes,
            args.experiment,
            args.condition,
        )
        for exposure in EXPOSURES
    }
    payload = {
        "input_directory": str(input_dir),
        "experiment": args.experiment,
        "condition": args.condition,
        "exposure_order": list(EXPOSURES),
        "scenes": scenes,
        "aggregation": {
            "abs_rel": "depth-pixel weighted mean",
            "delta1": "depth-pixel weighted mean",
            "rmse_log": "pooled pixel squared log error, then square root",
            "rmse": "pooled pixel squared error, then square root",
            "ate_rpe": "evaluated-batch weighted scene means",
        },
        "by_exposure": by_exposure,
        "latex_rows": {
            "abs_rel": "       " + " ".join(
                f"& {by_exposure[ev]['metrics']['abs_rel']:.4f}" for ev in EXPOSURES
            ),
            "delta1": "       " + " ".join(
                f"& {by_exposure[ev]['metrics']['delta1']:.4f}" for ev in EXPOSURES
            ),
            "rmse_log": "       " + " ".join(
                f"& {by_exposure[ev]['metrics']['rmse_log']:.4f}" for ev in EXPOSURES
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Saved: {output}")
    for exposure in EXPOSURES:
        values = by_exposure[exposure]["metrics"]
        print(
            f"{exposure}: AbsRel={values['abs_rel']:.6f} "
            f"d1={values['delta1']:.6f} RMSElog={values['rmse_log']:.6f} "
            f"RMSE={values['rmse']:.6f} ATE={values['ate']:.6f} "
            f"RPE_t={values['rpe_trans']:.6f} "
            f"RPE_r={values['rpe_rot_deg']:.6f}"
        )
    print("\nLaTeX:")
    for key in ("abs_rel", "delta1", "rmse_log"):
        print(payload["latex_rows"][key])


if __name__ == "__main__":
    main()
