"""Collect counterfactual event-input probes into one compact comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_row(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    zero = summary["comparisons"]["real_vs_zero"]
    reverse = summary["comparisons"]["real_vs_reverse_time"]
    swap = summary["comparisons"]["real_vs_swap_polarity"]
    return {
        "method": summary_path.parent.name,
        "variant": summary.get("model_variant", ""),
        "sensitive": bool(summary.get("event_output_sensitivity_detected", False)),
        "zero_normal_change_deg": zero["normal_output_change_mean_deg"],
        "zero_error_advantage_deg": zero["normal_error_advantage_deg"],
        "zero_high_event_advantage_deg": zero["high_event_error_advantage_deg"],
        "reverse_normal_change_deg": reverse["normal_output_change_mean_deg"],
        "reverse_error_advantage_deg": reverse["normal_error_advantage_deg"],
        "swap_normal_change_deg": swap["normal_output_change_mean_deg"],
        "swap_error_advantage_deg": swap["normal_error_advantage_deg"],
    }


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize event counterfactual output sensitivity")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = [load_row(path) for path in sorted(run_dir.glob("*/summary.json"))]
    if not rows:
        raise FileNotFoundError(f"No per-model summary.json found under {run_dir}")

    csv_path = run_dir / "counterfactual_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("Event counterfactual comparison")
    print(
        "| method | variant | event sensitive | zero output delta (deg) | "
        "zero error advantage (deg) | reverse output delta (deg) | reverse error advantage (deg) |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['method']} | {row['variant']} | {row['sensitive']} | "
            f"{fmt(row['zero_normal_change_deg'])} | {fmt(row['zero_error_advantage_deg'])} | "
            f"{fmt(row['reverse_normal_change_deg'])} | {fmt(row['reverse_error_advantage_deg'])} |"
        )
    print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
