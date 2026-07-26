"""Summarize the non-finetuned RGB four-scene CSV by exposure.

Only ``scope=scene`` rows are pooled.  AbsRel/delta1 use depth-pixel weights,
RMSE/RMSElog pool squared errors before the square root, and pose metrics use
evaluated-batch weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
METRICS = (
    "abs_rel", "delta1", "rmse_log", "rmse",
    "ate", "rpe_trans", "rpe_rot_deg",
)
RMS_METRICS = {"rmse_log", "rmse"}
POSE_METRICS = {"ate", "rpe_trans", "rpe_rot_deg"}


EXPOSURE_ORDER = ("ev_0", "ev_1", "ev_2", "ev_5", "ev_10")


def finite_float(row, key):
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def weighted_metric(rows, key):
    weight_key = "evaluated_batches" if key in POSE_METRICS else "depth_pixels"
    numerator = denominator = 0.0
    for row in rows:
        value = finite_float(row, key)
        weight = finite_float(row, weight_key)
        if value is None or weight is None or weight <= 0:
            continue
        numerator += weight * (value * value if key in RMS_METRICS else value)
        denominator += weight
    if denominator <= 0:
        return float("nan")
    pooled = numerator / denominator
    return math.sqrt(max(pooled, 0.0)) if key in RMS_METRICS else pooled


def aggregate_group(rows, scenes):
    present = {row.get("scene") for row in rows}
    missing = [scene for scene in scenes if scene not in present]
    if missing:
        raise RuntimeError(f"missing scene rows: {missing}")
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=r"E:\result\eventvgg\rgb_no_f\4_scene.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: input CSV directory",
    )
    parser.add_argument(
        "--experiment", default="rgb_pretrained_no_finetune"
    )
    parser.add_argument("--condition", default="rgb_only_fixed_scale")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(args.output_dir) if args.output_dir else source.parent
    output.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    scenes = list(DEFAULT_SCENES)
    result = []
    for exposure in EXPOSURE_ORDER:
        selected = [
            row for row in rows
            if row.get("scope") == "scene"
            and row.get("scene") in scenes
            and row.get("ldr_event_id") == exposure
            and row.get("experiment") == args.experiment
            and row.get("condition") == args.condition
        ]
        if len(selected) != len(scenes):
            present = [row.get("scene") for row in selected]
            raise RuntimeError(
                f"{exposure}: expected {len(scenes)} scene rows, got "
                f"{len(selected)}; present={present}"
            )
        result.append({
            "experiment": args.experiment,
            "condition": args.condition,
            "ldr_event_id": exposure,
            **aggregate_group(selected, scenes),
        })

    fields = (
        "experiment", "condition", "ldr_event_id", "scene", "scene_count",
        "evaluated_batches", "depth_pixels", *METRICS,
    )
    csv_path = output / "rgb_no_f_four_scenes_by_ev.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)

    latex = {
        key: "       " + " ".join(
            f"& {row[key]:.4f}" for row in result
        )
        for key in ("abs_rel", "delta1", "rmse_log")
    }
    payload = {
        "input": str(source),
        "experiment": args.experiment,
        "condition": args.condition,
        "exposure_order": list(EXPOSURE_ORDER),
        "scenes": scenes,
        "aggregation": {
            "abs_rel_delta1": "depth-pixel weighted mean",
            "rmse_rmse_log": "pixel-weighted squared error, then square root",
            "ate_rpe": "evaluated-batch weighted scene means",
        },
        "rows": result,
        "latex_rows": latex,
    }
    json_path = output / "rgb_no_f_four_scenes_by_ev.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    for row in result:
        print(
            f"{row['ldr_event_id']}: AbsRel={row['abs_rel']:.6f} "
            f"d1={row['delta1']:.6f} RMSElog={row['rmse_log']:.6f} "
            f"RMSE={row['rmse']:.6f} ATE={row['ate']:.6f} "
            f"RPE_t={row['rpe_trans']:.6f} "
            f"RPE_r={row['rpe_rot_deg']:.6f}"
        )
    print("\nLaTeX:")
    for key in ("abs_rel", "delta1", "rmse_log"):
        print(latex[key])


if __name__ == "__main__":
    main()
