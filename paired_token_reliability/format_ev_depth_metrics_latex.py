"""Format five-exposure depth metrics as three LaTeX table rows.

The input may be either:

1. the original per-scene ``metrics.csv``; or
2. ``four_scenes_all_by_ev.csv`` produced by
   ``summarize_four_scenes_by_ev.py``.

Output order is always ev_0, ev_1, ev_2, ev_5, ev_10.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
EXPOSURES = ("ev_0", "ev_1", "ev_2", "ev_5", "ev_10")
DISPLAY_METRICS = (
    ("abs_rel", r"Abs Rel"),
    ("delta1", r"$\delta < 1.25$"),
    ("rmse_log", r"$\mathrm{RMSE}_{\log}$"),
)
RMS_METRICS = {"rmse_log", "rmse"}
POSE_METRICS = {"ate", "rpe_trans", "rpe_rot_deg"}


def _finite(row, key):
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _weighted(rows, key):
    weight_key = "evaluated_batches" if key in POSE_METRICS else "depth_pixels"
    numerator = denominator = 0.0
    for row in rows:
        value, weight = _finite(row, key), _finite(row, weight_key)
        if value is None or weight is None or weight <= 0:
            continue
        numerator += weight * (value * value if key in RMS_METRICS else value)
        denominator += weight
    if denominator <= 0:
        return float("nan")
    value = numerator / denominator
    return math.sqrt(max(value, 0.0)) if key in RMS_METRICS else value


def aggregate_group(rows, scenes):
    present = {row.get("scene") for row in rows}
    missing = [scene for scene in scenes if scene not in present]
    if missing:
        raise RuntimeError(f"Missing requested scene rows: {missing}")
    return {
        key: _weighted(rows, key)
        for key in ("abs_rel", "delta1", "rmse_log", "rmse",
                    "ate", "rpe_trans", "rpe_rot_deg")
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=r"E:\result\eventvgg\hdreff\metrics.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <input-directory>/depth_metrics_ev_latex.txt",
    )
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Prefix each row with its metric label.",
    )
    return parser.parse_args()


def select_identity(rows, args):
    identities = sorted(
        {
            (row.get("experiment", ""), row.get("condition", ""))
            for row in rows
            if (args.experiment is None or row.get("experiment") == args.experiment)
            and (args.condition is None or row.get("condition") == args.condition)
        }
    )
    if not identities:
        raise RuntimeError("No matching experiment/condition found")
    if len(identities) > 1:
        choices = "\n  ".join(
            f"--experiment {experiment!r} --condition {condition!r}"
            for experiment, condition in identities
        )
        raise RuntimeError(
            "Input contains multiple experiment/condition combinations. "
            "Select one explicitly:\n  " + choices
        )
    return identities[0]


def exposure_rows(source_rows, args):
    experiment, condition = select_identity(source_rows, args)
    is_raw = "scope" in source_rows[0]
    result = {}
    for exposure in EXPOSURES:
        rows = [
            row
            for row in source_rows
            if row.get("experiment", "") == experiment
            and row.get("condition", "") == condition
            and row.get("ldr_event_id") == exposure
        ]
        if is_raw:
            rows = [
                row
                for row in rows
                if row.get("scope") == "scene" and row.get("scene") in args.scenes
            ]
            if not rows:
                raise RuntimeError(f"No four-scene source rows for {exposure}")
            result[exposure] = aggregate_group(rows, args.scenes)
        else:
            candidates = [
                row
                for row in rows
                if row.get("scene") in {"ALL_4_SCENES", "ALL"}
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one aggregate row for {exposure}, got {len(candidates)}"
                )
            result[exposure] = candidates[0]
    return experiment, condition, result


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty CSV: {input_path}")

    experiment, condition, values = exposure_rows(rows, args)
    lines = []
    for key, label in DISPLAY_METRICS:
        numbers = " ".join(
            f"& {float(values[exposure][key]):.{args.precision}f}"
            for exposure in EXPOSURES
        )
        prefix = f"{label} " if args.with_labels else "       "
        lines.append(f"{prefix}{numbers}")

    text = "\n".join(lines) + "\n"
    output = (
        Path(args.output)
        if args.output
        else input_path.parent / "depth_metrics_ev_latex.txt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    print(
        f"% experiment={experiment}, condition={condition}, "
        f"order={','.join(EXPOSURES)}"
    )
    print(text, end="")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
