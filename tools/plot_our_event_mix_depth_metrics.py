#!/usr/bin/env python3
"""Plot four-scene depth metrics across event-mixing levels.

Expected inputs: 0.csv, 0.25.csv, 0.5.csv, 0.75.csv, and 1.0.csv.
Only complete four-scene, final_event_refined results are plotted. Incomplete
levels remain NaN rather than being silently mixed with a different population.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FILES = [
    (0.00, "0.csv"),
    (0.25, "0.25.csv"),
    (0.50, "0.5.csv"),
    (0.75, "0.75.csv"),
    (1.00, "1.0.csv"),
]
EXPOSURES = ["ev_0", "ev_1", "ev_2", "ev_5", "ev_10"]
COLORS = {
    "ev_0": "#0072B2",
    "ev_1": "#009E73",
    "ev_2": "#E69F00",
    "ev_5": "#D55E00",
    "ev_10": "#CC79A7",
}


def weighted_mean(rows: list[dict[str, str]], key: str) -> float:
    weights = [float(r["depth_pixels"]) for r in rows]
    values = [float(r[key]) for r in rows]
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total if total else math.nan


def tight_limits(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    lo, hi = min(finite), max(finite)
    span = hi - lo
    pad = max(span * 0.18, max(abs(lo), abs(hi), 1e-6) * 0.012)
    return lo - pad, hi + pad


def plot_metric(
    summary: list[dict[str, object]],
    metric: str,
    ylabel: str,
    output: Path,
    ax=None,
    show_title: bool = True,
) -> None:
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9.2, 5.8), dpi=180)

    all_values: list[float] = []
    label_offsets = {"ev_0": -14, "ev_1": 8, "ev_2": 17, "ev_5": 8, "ev_10": 8}
    for exposure in EXPOSURES:
        rows = [r for r in summary if r["exposure"] == exposure]
        xs = [float(r["alpha_full"]) * 100 for r in rows]
        ys = [float(r[metric]) for r in rows]
        all_values.extend(ys)
        ax.plot(
            xs,
            ys,
            color=COLORS[exposure],
            marker="o",
            markersize=6.5,
            linewidth=2.2,
            label=exposure.replace("_", " "),
            zorder=3,
        )
        for x, y in zip(xs, ys):
            if math.isfinite(y):
                ax.annotate(
                    f"{y:.4f}",
                    (x, y),
                    xytext=(0, label_offsets[exposure]),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if label_offsets[exposure] >= 0 else "top",
                    fontsize=8,
                    color=COLORS[exposure],
                    fontweight="semibold",
                )

    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(*tight_limits(all_values))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Non-geometry event mixing ratio", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#EEF1F5", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    if show_title:
        ax.set_title("Final event-refined depth (four-scene pixel-weighted)", pad=34)

    if standalone:
        plt.tight_layout()
        plt.savefig(output, bbox_inches="tight", facecolor="white")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(r"E:\result\eventvgg\our"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/our_event_mix_depth_metrics"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for alpha, filename in FILES:
        path = args.input_dir / filename
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        final_rows = [
            r for r in rows
            if r.get("scope") == "scene" and r.get("condition") == "final_event_refined"
        ]
        scenes = sorted({r["scene"] for r in final_rows})
        for exposure in EXPOSURES:
            selected = [r for r in final_rows if r["exposure"] == exposure]
            complete = len(selected) == 4 and len({r["scene"] for r in selected}) == 4
            diagnostics.append(
                {
                    "file": filename,
                    "alpha_full": alpha,
                    "exposure": exposure,
                    "rows": len(selected),
                    "scenes": sorted({r["scene"] for r in selected}),
                    "complete_four_scene_result": complete,
                }
            )
            summary.append(
                {
                    "alpha_full": alpha,
                    "exposure": exposure,
                    "abs_rel": weighted_mean(selected, "abs_rel") if complete else math.nan,
                    "rmse_log": weighted_mean(selected, "rmse_log") if complete else math.nan,
                    "scene_count": len({r["scene"] for r in selected}),
                    "source_file": filename,
                }
            )

    csv_path = args.output_dir / "depth_metric_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    plot_metric(summary, "abs_rel", "Abs Rel ↓", args.output_dir / "abs_rel_vs_event_mix.png")
    plot_metric(summary, "rmse_log", r"RMSE$_{\log}$ ↓", args.output_dir / "rmse_log_vs_event_mix.png")

    fig, axes = plt.subplots(1, 2, figsize=(17.2, 5.8), dpi=180)
    plot_metric(summary, "abs_rel", "Abs Rel ↓", None, axes[0], show_title=False)
    plot_metric(summary, "rmse_log", r"RMSE$_{\log}$ ↓", None, axes[1], show_title=False)
    fig.suptitle("Depth robustness under increasing non-geometry events", y=1.01, fontsize=15)
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "depth_metrics_vs_event_mix.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(f"Saved results to {args.output_dir.resolve()}")
    incomplete = [d for d in diagnostics if not d["complete_four_scene_result"]]
    if incomplete:
        print("WARNING: incomplete four-scene points were kept as NaN:")
        for d in incomplete:
            print(f"  {d['file']} {d['exposure']}: {d['rows']} rows, scenes={d['scenes']}")


if __name__ == "__main__":
    main()
