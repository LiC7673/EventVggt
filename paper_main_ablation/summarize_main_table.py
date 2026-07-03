"""Convert evaluator output into the compact, auditable paper main table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODULES = {
    "A0_RGB_only": (0, 0, 0, 0),
    "A1_Direct_event": (1, 0, 0, 0),
    "A2_w_o_Reliability": (1, 1, 1, 0),
    "A3_w_o_MultiLDR": (1, 1, 0, 1),
    "A4_w_o_Detail": (1, 0, 1, 1),
    "A5_Full": (1, 1, 1, 1),
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
    rgb_source = by_name["A0_RGB_only"]
    full_source = by_name["A5_Full"]
    rgb_abs_rel = _float(rgb_source, "abs_rel")
    rgb_normal = _float(rgb_source, "normal_error_deg")
    full_abs_rel = _float(full_source, "abs_rel")
    full_normal = _float(full_source, "normal_error_deg")
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
        row.update(
            {
                "abs_rel_reduction_vs_rgb": rgb_abs_rel - row["abs_rel"],
                "normal_reduction_vs_rgb_deg": rgb_normal - row["normal_error_deg"],
                "abs_rel_gap_to_full": row["abs_rel"] - full_abs_rel,
                "normal_gap_to_full_deg": row["normal_error_deg"] - full_normal,
            }
        )
        rows.append(row)

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
