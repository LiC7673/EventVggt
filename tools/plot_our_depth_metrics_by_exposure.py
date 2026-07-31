#!/usr/bin/env python3
"""Plot Our method over exposure levels and export LaTeX-style table rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
    }
)

LEVELS = [
    (0.00, "0.csv"),
    (0.25, "0.25.csv"),
    (0.50, "0.5.csv"),
    (0.75, "0.75.csv"),
    (1.00, "1.0.csv"),
]
EXPOSURES = ["ev_0", "ev_1", "ev_2", "ev_5", "ev_10"]
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def tight_limits(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = max(span * 0.15, max(abs(lo), abs(hi), 1e-6) * 0.01)
    return lo - pad, hi + pad


def summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        r for r in rows
        if r.get("scope") == "all_scenes_pixel_weighted"
        and r.get("condition") == "final_event_refined"
    ]


def read_summary(root: Path) -> tuple[list[dict[str, float | str]], list[str]]:
    output: list[dict[str, float | str]] = []
    notes: list[str] = []
    for alpha, filename in LEVELS:
        rows = summary_rows(root / filename)
        by_exposure = {r["exposure"]: r for r in rows}
        missing = [e for e in EXPOSURES if e not in by_exposure]
        if alpha == 1.0 and missing and (root / "4_scene.csv").exists():
            by_exposure = {
                r["exposure"]: r for r in summary_rows(root / "4_scene.csv")
            }
            notes.append(
                "alpha=1.0 used 4_scene.csv because 1.0.csv lacks a complete "
                "four-scene/all-exposure summary."
            )
            missing = [e for e in EXPOSURES if e not in by_exposure]
        if missing:
            raise RuntimeError(f"{filename} missing complete exposures: {missing}")
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
    return output, notes


def draw(
    summary: list[dict[str, float | str]],
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    # Compact, paper-friendly figure; all text is fixed at 11 pt.
    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
    x = list(range(len(EXPOSURES)))
    all_values: list[float] = []

    for (alpha, _), color in zip(LEVELS, COLORS):
        rows = [r for r in summary if math.isclose(float(r["alpha_full"]), alpha)]
        by_exposure = {str(r["exposure"]): float(r[metric]) for r in rows}
        y = [by_exposure[e] for e in EXPOSURES]
        all_values.extend(y)
        ax.plot(
            x,
            y,
            color=color,
            marker="o",
            markersize=5.4,
            linewidth=2.0,
            label=rf"$\alpha={alpha:g}$",
            zorder=3,
        )

    ax.set_xticks(x, [e.replace("_", "") for e in EXPOSURES])
    ax.set_xlim(-0.12, len(EXPOSURES) - 0.88)
    ax.set_ylim(*tight_limits(all_values))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(
        title="",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        frameon=False,
        columnspacing=1.15,
        handlelength=1.8,
    )
    # Replace the ordinary left/bottom spines with arrowed axes.
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    axis_arrow = dict(
        arrowstyle="-|>",
        color="#202124",
        lw=1.25,
        mutation_scale=13,
        shrinkA=0,
        shrinkB=0,
    )
    ax.annotate(
        "",
        xy=(1.018, 0),
        xytext=(-0.015, 0),
        xycoords="axes fraction",
        arrowprops=axis_arrow,
        annotation_clip=False,
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(0, 1.025),
        xytext=(0, -0.015),
        xycoords="axes fraction",
        arrowprops=axis_arrow,
        annotation_clip=False,
        zorder=6,
    )
    # Put each axis name beside its arrow head.
    ax.text(
        1.1,
        0.15,
        "Exposure level",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.text(
        -0.15,
        1.05,
        ylabel,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        rotation=0,
    )
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="both",
        color="#C9D2DC",
        linewidth=0.55,
        alpha=0.38,
    )
    fig.subplots_adjust(left=0.12, right=0.965, bottom=0.18, top=0.80)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def latex_rows(summary: list[dict[str, float | str]]) -> str:
    lines = [
        r"& $ev_0$ & $ev_1$ & $ev_2$ & $ev_5$ & $ev_{10}$ \\",
        r"\midrule",
    ]
    for metric, title in (
        ("abs_rel", r"Abs Rel $\downarrow$"),
        ("rmse_log", r"RMSE$_{\log}$ $\downarrow$"),
    ):
        lines.append(rf"\multicolumn{{6}}{{c}}{{{title}}} \\")
        for alpha, _ in LEVELS:
            rows = [r for r in summary if math.isclose(float(r["alpha_full"]), alpha)]
            values = {str(r["exposure"]): float(r[metric]) for r in rows}
            cells = " & ".join(f"{values[e]:.4f}" for e in EXPOSURES)
            lines.append(rf"$\alpha={alpha:g}$ & {cells} \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path(r"E:\result\eventvgg\our")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/our_depth_metrics_by_exposure"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, notes = read_summary(args.input_dir)
    with (args.output_dir / "depth_metric_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["alpha_full", "exposure", "abs_rel", "rmse_log"]
        )
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "depth_metric_summary.json").write_text(
        json.dumps({"data": summary, "notes": notes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "latex_table_rows.txt").write_text(
        latex_rows(summary), encoding="utf-8"
    )
    draw(summary, "abs_rel", "Abs Rel ↓", args.output_dir / "abs_rel_by_exposure.png")
    draw(
        summary,
        "rmse_log",
        r"RMSE$_{\log}$ ↓",
        args.output_dir / "rmse_log_by_exposure.png",
    )
    print(f"Saved results to {args.output_dir.resolve()}")
    for note in notes:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
