"""Aggregate ablation evaluation JSON files into the requested CSV and analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ab_st1_st2 import METHODS


def _difference(left, right):
    if not (math.isfinite(left) and math.isfinite(right)):
        return float("nan")
    return left - right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments/ablation")
    args = parser.parse_args()
    root = Path(args.root)
    results = {}
    rows = []
    for method in METHODS:
        path = root / method / "evaluation.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        results[method] = result
        metrics = result["metrics"]
        contribution = result["contribution"]
        rows.append(
            {
                "Method": method,
                "Depth": metrics.get("abs_rel", float("nan")),
                "Normal": metrics.get("normal_mean_deg", float("nan")),
                "Pose": metrics.get("ate", float("nan")),
                "C_mean": contribution.get("mean", float("nan")),
                "C_std": contribution.get("std", float("nan")),
            }
        )
    with (root / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("Method", "Depth", "Normal", "Pose", "C_mean", "C_std")
        )
        writer.writeheader()
        writer.writerows(rows)

    rgb = results["rgb_only"]["metrics"]
    raw = results["raw_event"]["metrics"]
    ours = results["ours"]["metrics"]
    no_multi = results["no_multildr"]
    analysis = {
        "raw_event_better_than_rgb_only": {
            "abs_rel_reduction": _difference(rgb["abs_rel"], raw["abs_rel"]),
            "normal_deg_reduction": _difference(rgb["normal_mean_deg"], raw["normal_mean_deg"]),
        },
        "learned_contribution_better_than_raw_event": {
            "abs_rel_reduction": _difference(raw["abs_rel"], ours["abs_rel"]),
            "normal_deg_reduction": _difference(raw["normal_mean_deg"], ours["normal_mean_deg"]),
        },
        "multildr_selectivity_gain": {
            "abs_rel_reduction": _difference(no_multi["metrics"]["abs_rel"], ours["abs_rel"]),
            "normal_deg_reduction": _difference(
                no_multi["metrics"]["normal_mean_deg"], ours["normal_mean_deg"]
            ),
            "ours_C_std": results["ours"]["contribution"]["std"],
            "no_multildr_C_std": no_multi["contribution"]["std"],
            "std_increase": _difference(
                results["ours"]["contribution"]["std"], no_multi["contribution"]["std"]
            ),
        },
        "spatial_variation": {
            method: {
                "C_std": result["contribution"]["std"],
                "collapse": result["contribution"]["collapse"],
            }
            for method, result in results.items()
        },
    }
    (root / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
