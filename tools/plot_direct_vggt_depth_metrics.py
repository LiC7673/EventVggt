#!/usr/bin/env python3
"""Plot Direct+VGGT depth metrics across exposure levels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


LEVELS = [
    # (0.00, "0.csv"),
    (0.25, "0.25.csv"),
    (0.50, "0.50.csv"),
    (0.75, "0.75.csv"),
    # (1.00, "1.0.csv"),
]
EXPOSURES = ["ev_0", "ev_1", "ev_2", "ev_5", "ev_10"]
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def tight_limits(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = max(0.15 * span, 0.01 * max(abs(lo), abs(hi), 1e-6))
    return lo - pad, hi + pad


def read_summary(root: Path) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for alpha, filename in LEVELS:
        with (root / filename).open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        rows = [
            r for r in rows
            if r.get("scope") == "all_scenes_pixel_weighted"
            and r.get("condition") == "rgb_plus_direct_event"
        ]
        by_exposure = {r["exposure"]: r for r in rows}
        missing = [e for e in EXPOSURES if e not in by_exposure]
        if missing:
            raise RuntimeError(f"{filename} missing exposures: {missing}")
        for exposure in EXPOSURES:
            row = by_exposure[exposure]
            output.append(
                {
                    "alpha_full": alpha,
                    "exposure": exposure,
                    "abs_rel": float(row["abs_rel"]),
                    "rmse_log": float(row["rmse_log"]),
                }
            )
    return output


def draw(summary: list[dict[str, float | str]], metric: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=200)
    x = list(range(len(EXPOSURES)))
    values: list[float] = []

    for (alpha, _), color in zip(LEVELS, COLORS):
        rows = [r for r in summary if math.isclose(float(r["alpha_full"]), alpha)]
        by_exposure = {str(r["exposure"]): float(r[metric]) for r in rows}
        y = [by_exposure[e] for e in EXPOSURES]
        values.extend(y)
        ax.plot(
            x,
            y,
            color=color,
            marker="o",
            markersize=6.5,
            linewidth=2.35,
            label=rf"$\alpha={alpha:g}$",
            zorder=3,
        )
        # Direction arrow at the final segment: ev_5 -> ev_10.
        ax.annotate(
            "",
            xy=(x[-1], y[-1]),
            xytext=(x[-2] + 0.12, y[-2] + 0.12 * (y[-1] - y[-2])),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.35, mutation_scale=13),
            zorder=4,
        )

    ax.set_xticks(x, [e.replace("_", "") for e in EXPOSURES])
    ax.set_xlim(-0.12, len(EXPOSURES) - 0.88)
    ax.set_ylim(*tight_limits(values))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Exposure level", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Direct Event + StreamVGGT", fontsize=14, pad=16)
    ax.legend(
        title="Full-event ratio",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        columnspacing=1.4,
        handlelength=2.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path(r"E:\result\eventvgg\our")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/direct_vggt_depth_metrics")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(args.input_dir)
    with (args.output_dir / "depth_metric_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["alpha_full", "exposure", "abs_rel", "rmse_log"]
        )
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "depth_metric_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    draw(summary, "abs_rel", "Abs Rel ↓", args.output_dir / "abs_rel_by_exposure.png")
    draw(
        summary,
        "rmse_log",
        r"RMSE$_{\log}$ ↓",
        args.output_dir / "rmse_log_by_exposure.png",
    )
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
