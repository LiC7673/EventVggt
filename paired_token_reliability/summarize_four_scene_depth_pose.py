"""Create a compact depth/pose report from the shared four-scene evaluator."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = (
    "mae",
    "abs_rel",
    "delta1",
    "rmse_log",
    "rmse",
    "ate",
    "rpe_trans",
    "rpe_rot_deg",
)


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _format(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Evaluator metrics.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Keep per-scene rows and the pixel-weighted four-scene result at each EV.
    selected = [
        row for row in rows
        if row.get("scope") in {"scene", "all_scenes_pixel_weighted"}
    ]
    fields = ("scope", "scene", "exposure", "condition", "evaluated_batches") + METRICS
    compact_path = output / "depth_pose_report.csv"
    with compact_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "Depth and pose evaluation",
        "Depth protocol: fixed scale from evaluator; no test-GT fitting.",
        "Pose protocol: first-frame SE(3) alignment; no pose-scale fitting; "
        "ATE RMSE and adjacent-view RPE.",
        "",
    ]
    for row in selected:
        label = (
            f"{row.get('scope')} | {row.get('scene')} | "
            f"{row.get('exposure')} | {row.get('condition')}"
        )
        values = " ".join(
            f"{name}={_format(_number(row, name))}" for name in METRICS
        )
        lines.append(f"{label}: {values}")
    (output / "depth_pose_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved {compact_path} and {output / 'depth_pose_report.txt'}")


if __name__ == "__main__":
    main()
