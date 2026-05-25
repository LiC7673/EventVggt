"""Summarize normal-error/event-correlation validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    "view0_normal_error_mean_deg",
    "view0_high_event_normal_error_mean_deg",
    "view0_low_event_normal_error_mean_deg",
    "view0_high_minus_low_error_deg",
    "view0_corr_event_gt_detail_pearson",
    "view0_corr_event_pred_detail_pearson",
    "pair_high_event_normal_error_mean_deg",
    "pair_low_event_normal_error_mean_deg",
    "pair_high_minus_low_error_deg",
    "pair_corr_event_error_pearson",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge normal-error validation summary files")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    return parser.parse_args()


def format_value(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = []
    for method in args.methods:
        summary_path = run_dir / method / "summary.json"
        if not summary_path.exists():
            rows.append({"method": method, "error": f"missing {summary_path}"})
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        metrics = summary.get("metrics_mean", {})
        row = {"method": method, "num_records": summary.get("num_records", 0)}
        row.update({key: metrics.get(key) for key in KEYS})
        rows.append(row)

    result = {
        "run_dir": str(run_dir),
        "methods": rows,
        "notes": {
            "normal_error": "Lower is better.",
            "high_minus_low_error": "Compare its reduction versus baseline; negative means high-event areas are easier.",
            "pred_detail_corr": "Higher agreement with event-supported predicted detail is better only together with lower normal error.",
        },
    }
    with (run_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        "method".ljust(24),
        "normal".rjust(10),
        "high_evt".rjust(10),
        "low_evt".rjust(10),
        "high-low".rjust(10),
        "evt_gt".rjust(10),
        "evt_pred".rjust(10),
    )
    for row in rows:
        if "error" in row:
            print(row["method"].ljust(24), row["error"])
            continue
        print(
            row["method"].ljust(24),
            format_value(row.get("view0_normal_error_mean_deg")).rjust(10),
            format_value(row.get("view0_high_event_normal_error_mean_deg")).rjust(10),
            format_value(row.get("view0_low_event_normal_error_mean_deg")).rjust(10),
            format_value(row.get("view0_high_minus_low_error_deg")).rjust(10),
            format_value(row.get("view0_corr_event_gt_detail_pearson")).rjust(10),
            format_value(row.get("view0_corr_event_pred_detail_pearson")).rjust(10),
        )
    print(f"Saved {run_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
