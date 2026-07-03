"""Convert evaluator output into the compact, auditable paper main table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODULES = {
    "M0_matched_RGB": (0, 0, 0, 0),
    "M1_event_residual": (1, 0, 0, 0),
    "M2_event_detail_GT": (1, 1, 0, 0),
    "M3_event_detail_MultiLDR": (1, 1, 1, 0),
    "M4_full_reliability": (1, 1, 1, 1),
}


def _float(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    by_name = {row["name"]: row for row in source_rows}
    missing = [name for name in MODULES if name not in by_name]
    if missing:
        raise RuntimeError(f"Main-table evaluation is incomplete; missing rows: {missing}")

    rows = []
    previous = None
    for name, flags in MODULES.items():
        source = by_name[name]
        row = {
            "model": name,
            "event_residual": flags[0],
            "detail_gt": flags[1],
            "multi_ldr": flags[2],
            "reliability": flags[3],
            "abs_rel": _float(source, "abs_rel"),
            "delta1": _float(source, "delta1"),
            "rmse_log": _float(source, "rmse_log"),
            "normal_error_deg": _float(source, "normal_error_deg"),
            "ate": _float(source, "ate"),
            "rpe_trans": _float(source, "rpe_trans"),
            "rpe_rot_deg": _float(source, "rpe_rot_deg"),
        }
        if previous is None:
            row.update(
                {
                    "abs_rel_reduction_vs_prev": float("nan"),
                    "delta1_gain_vs_prev": float("nan"),
                    "normal_reduction_vs_prev_deg": float("nan"),
                }
            )
        else:
            row.update(
                {
                    "abs_rel_reduction_vs_prev": previous["abs_rel"] - row["abs_rel"],
                    "delta1_gain_vs_prev": row["delta1"] - previous["delta1"],
                    "normal_reduction_vs_prev_deg": (
                        previous["normal_error_deg"] - row["normal_error_deg"]
                    ),
                }
            )
        rows.append(row)
        previous = row

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nClean paper main table")
    print("model                         E D L R   AbsRel  delta1  RMSElog  Normal")
    for row in rows:
        print(
            f"{row['model']:29s} "
            f"{row['event_residual']} {row['detail_gt']} {row['multi_ldr']} {row['reliability']} "
            f"{row['abs_rel']:8.5f} {row['delta1']:7.4f} "
            f"{row['rmse_log']:8.5f} {row['normal_error_deg']:7.3f}"
        )
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
