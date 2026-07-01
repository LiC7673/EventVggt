"""Plots for trainable event components in controlled frozen-coarse runs."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt


PLOT_METRICS = (
    ("detail_gt_loss", "GT Detail Loss", False),
    ("img_event_rel_loss", "Image-guided Reliability Loss", False),
    ("branch_token_loss", "Branch Token Loss", False),
    ("normal_refinement_gain", "Normal Refinement Gain", True),
    ("depth_refinement_gain", "Depth Refinement Gain", True),
    ("event_gate_mean", "Event Gate Mean", True),
)


def generate_event_ablation_plots(cfg) -> None:
    metrics_path = Path(cfg.output_dir) / "metrics.json"
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        entries = [entries]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for axis, (key, title, draw_zero) in zip(axes.flat, PLOT_METRICS):
        for split, color, marker in (("train", "tab:blue", "o"), ("test", "tab:red", "s")):
            values = []
            for entry in entries:
                metrics = entry.get(split, {})
                value = metrics.get(key)
                # Test branch-token supervision is intentionally unavailable;
                # do not draw its synthetic zero placeholder.
                if value is None or (key == "branch_token_loss" and split == "test" and value == 0):
                    continue
                values.append((entry.get("step", 0), value))
            if values:
                steps, metric_values = zip(*values)
                axis.plot(steps, metric_values, color=color, marker=marker, markersize=3, label=split)
        if draw_zero:
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        axis.set_title(title)
        axis.set_xlabel("Global Step")
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
    fig.suptitle("Controlled Event Ablation: Trainable Components", fontsize=15)
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "event_ablation_plots.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def install_event_plot_hook(fe_module) -> None:
    original = fe_module.generate_loss_plots

    def generate_all(cfg):
        original(cfg)
        generate_event_ablation_plots(cfg)

    fe_module.generate_loss_plots = generate_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate_event_ablation_plots(SimpleNamespace(output_dir=args.output_dir))
