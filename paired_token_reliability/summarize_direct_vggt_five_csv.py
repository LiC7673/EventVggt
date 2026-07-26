"""Aggregate Direct+VGGT ev_0/1/2/5/10 CSV files over four scenes."""
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
    "abs_rel", "delta1", "rmse_log", "rmse",
    "ate", "rpe_trans", "rpe_rot_deg",
)
RMS_METRICS = {"rmse_log", "rmse"}
POSE_METRICS = {"ate", "rpe_trans", "rpe_rot_deg"}


def finite(row, key):
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def scene_name(row):
    active = str(row.get("active_scenes", "")).strip()
    if active:
        return active
    name = str(row.get("name", ""))
    for scene in SCENES:
        if name.endswith(scene):
            return scene
    return ""


def pooled(rows, metric):
    weight_key = "evaluated_batches" if metric in POSE_METRICS else "depth_pixels"
    numerator = denominator = 0.0
    for row in rows:
        value = finite(row, metric)
        weight = finite(row, weight_key)
        if value is None or weight is None or weight <= 0:
            continue
        numerator += weight * (
            value * value if metric in RMS_METRICS else value
        )
        denominator += weight
    if denominator <= 0:
        return float("nan")
    result = numerator / denominator
    return math.sqrt(max(result, 0.0)) if metric in RMS_METRICS else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default=r"E:\result\eventvgg\dirct+Vggt"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Default: input directory"
    )
    args = parser.parse_args()
    source = Path(args.input_dir)
    output = Path(args.output_dir) if args.output_dir else source
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    results = []
    per_scene = {}
    for exposure in EXPOSURES:
        path = source / f"{exposure}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        selected = {}
        for row in rows:
            scene = scene_name(row)
            if scene in SCENES:
                if scene in selected:
                    raise RuntimeError(
                        f"{path.name}: duplicate row for {scene}"
                    )
                selected[scene] = row
        missing = [scene for scene in SCENES if scene not in selected]
        if missing:
            raise RuntimeError(f"{path.name}: missing scenes {missing}")
        ordered = [selected[scene] for scene in SCENES]
        metric_values = {key: pooled(ordered, key) for key in METRICS}
        result = {
            "exposure": exposure,
            "scene": "ALL_4_SCENES",
            "scene_count": len(SCENES),
            "evaluated_batches": sum(
                int(finite(row, "evaluated_batches") or 0) for row in ordered
            ),
            "depth_pixels": sum(
                int(finite(row, "depth_pixels") or 0) for row in ordered
            ),
            **metric_values,
        }
        results.append(result)
        per_scene[exposure] = {
            scene: {key: finite(selected[scene], key) for key in METRICS}
            for scene in SCENES
        }

    fields = (
        "exposure", "scene", "scene_count", "evaluated_batches",
        "depth_pixels", *METRICS,
    )
    csv_path = output / "direct_vggt_four_scenes_by_ev.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    latex = {
        key: "       " + " ".join(
            f"& {row[key]:.4f}" for row in results
        )
        for key in ("abs_rel", "delta1", "rmse_log")
    }
    payload = {
        "input_directory": str(source),
        "exposure_order": list(EXPOSURES),
        "scenes": list(SCENES),
        "aggregation": {
            "abs_rel_delta1": "depth-pixel weighted mean",
            "rmse_rmse_log": "pixel-weighted squared error, then square root",
            "ate_rpe": "evaluated-batch weighted scene means",
        },
        "rows": results,
        "per_scene": per_scene,
        "latex_rows": latex,
    }
    json_path = output / "direct_vggt_four_scenes_by_ev.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    for row in results:
        print(
            f"{row['exposure']}: AbsRel={row['abs_rel']:.6f} "
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
