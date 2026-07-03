"""Collect all per-scene/LDR jobs into paper-facing CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    "abs_rel", "delta1", "rmse_log", "rmse", "mae",
    "normal_error_deg", "ate", "rpe_trans", "rpe_rot_deg",
    "event_reliability_mean",
)

MODULE_FLAGS = {
    "A0_RGB_only": (0, 0, 0, 0),
    "A1_Direct_event": (1, 0, 0, 0),
    "A2_w_o_Reliability": (1, 1, 1, 0),
    "A3_w_o_MultiLDR": (1, 1, 0, 1),
    "A4_w_o_Detail": (1, 0, 1, 1),
    "A5_Full": (1, 1, 1, 1),
}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--expected-models", type=int, default=6)
    parser.add_argument("--expected-ldrs", type=int, default=4)
    parser.add_argument("--expected-scenes", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.results_root)
    files = sorted(root.glob("*/ev_*/per_scene_metrics.csv"))
    expected_jobs = args.expected_models * args.expected_ldrs
    if len(files) != expected_jobs:
        raise RuntimeError(f"Expected {expected_jobs} evaluation jobs, found {len(files)} under {root}")

    rows = []
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as handle:
            job_rows = list(csv.DictReader(handle))
        if len(job_rows) != args.expected_scenes:
            raise RuntimeError(f"Expected {args.expected_scenes} scenes in {path}, got {len(job_rows)}")
        rows.extend(job_rows)
    _write_csv(root / "all_scene_metrics.csv", rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["ldr"])].append(row)
    means = []
    for (model, ldr), values in sorted(groups.items()):
        flags = MODULE_FLAGS.get(model, (None, None, None, None))
        means.append(
            {
                "model": model,
                "ldr": ldr,
                "event": flags[0],
                "detail_gt": flags[1],
                "multi_ldr": flags[2],
                "reliability": flags[3],
                "num_scenes": len(values),
                **{
                    metric: _mean([_number(row.get(metric)) for row in values])
                    for metric in METRICS
                },
            }
        )
    _write_csv(root / "mean_metrics_by_model_ldr.csv", means)

    overall_groups = defaultdict(list)
    for row in rows:
        overall_groups[row["model"]].append(row)
    overall = []
    for model, values in sorted(overall_groups.items()):
        flags = MODULE_FLAGS.get(model, (None, None, None, None))
        overall.append(
            {
                "model": model,
                "event": flags[0],
                "detail_gt": flags[1],
                "multi_ldr": flags[2],
                "reliability": flags[3],
                "num_scene_ldr_cases": len(values),
                **{
                    metric: _mean([_number(row.get(metric)) for row in values])
                    for metric in METRICS
                },
            }
        )
    _write_csv(root / "mean_metrics_all_ldrs.csv", overall)
    (root / "all_scene_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Collected {len(rows)} scene-level rows from {len(files)} jobs")
    print(f"  {root / 'all_scene_metrics.csv'}")
    print(f"  {root / 'mean_metrics_by_model_ldr.csv'}")
    print(f"  {root / 'mean_metrics_all_ldrs.csv'}")


if __name__ == "__main__":
    main()
