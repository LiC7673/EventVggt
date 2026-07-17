"""Functional validation of C by deleting high/low/random-scored raw events.

The contribution map is predicted once from the intact full event stream.  For
each removal ratio, exactly the same number of raw events is then removed per
view.  The event voxel is rebuilt from the surviving x/y/t/p events before a
new geometry forward pass.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import finetune_event as fe
from ablation.eag3r_metrics_eval import move_views_to_device, stack_output
from event_branch_ablation.evaluate_event_contribution import ConditionAccumulator, _update_condition
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import build_model
import real_reliability_stage.evaluate_stage2_heldout as protocol


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
METHODS = ("remove_high", "remove_random", "remove_low")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--scene-names", nargs="+", default=list(SCENES))
    p.add_argument("--exposures", default="0,1,2,5,10")
    p.add_argument("--ratios", default="0.10,0.20,0.30,0.50")
    p.add_argument("--test-frame-count", type=int, default=120)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--event-resize-bins", type=int, default=5)
    p.add_argument("--depth-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--random-repeats", type=int, default=3,
                   help="independent random deletions averaged at each ratio")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    return p.parse_args()


def clone_views(views):
    copied = []
    for view in views:
        current = {}
        for key, value in view.items():
            if torch.is_tensor(value): current[key] = value.clone()
            elif isinstance(value, list):
                current[key] = [item.clone() if torch.is_tensor(item) else item for item in value]
            else: current[key] = value
        copied.append(current)
    return copied


def voxelize(xy, timestamp, polarity, time_range, channels, height, width):
    """Rebuild the same two-polarity linear-time voxel representation."""
    bins = channels // 2
    voxel = torch.zeros((channels, height, width), device=xy.device, dtype=torch.float32)
    if xy.numel() == 0: return voxel
    x, y = xy[:, 0].long(), xy[:, 1].long()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height) & torch.isfinite(timestamp) & torch.isfinite(polarity) & (polarity != 0)
    if not valid.any(): return voxel
    x, y, timestamp, polarity = x[valid], y[valid], timestamp[valid].float(), polarity[valid].float()
    t0, t1 = time_range[0].float(), time_range[1].float()
    continuous = ((timestamp - t0) / (t1 - t0).clamp_min(1e-12) * max(bins - 1, 0)).clamp(0, max(bins - 1, 0))
    left = torch.floor(continuous).long(); right = (left + 1).clamp_max(bins - 1)
    rw = continuous - left.float(); lw = 1. - rw
    offset = torch.where(polarity > 0, torch.zeros_like(left), torch.full_like(left, bins))
    magnitude = polarity.abs()
    flat = voxel.view(-1)
    spatial = y * width + x
    flat.scatter_add_(0, (left + offset) * (height * width) + spatial, magnitude * lw)
    flat.scatter_add_(0, (right + offset) * (height * width) + spatial, magnitude * rw)
    return voxel


def attach_full_voxels(views, bins):
    """Voxelize preserved raw events before the intact reference forward."""
    for view in views:
        batch, _, height, width = view["img"].shape
        voxel = torch.zeros((batch, 2 * bins, height, width), device=view["img"].device, dtype=torch.float32)
        for sample_index in range(batch):
            voxel[sample_index] = voxelize(
                view["event_xy"][sample_index], view["event_t"][sample_index],
                view["event_p"][sample_index], view["event_time_range"][sample_index],
                2 * bins, height, width,
            )
        view["event_voxel"] = voxel
    return views


def delete_by_score(views, contribution, method, ratio, seed):
    result = clone_views(views)
    removed = total = 0
    for view_index, view in enumerate(result):
        voxel = view["event_voxel"]
        batch, channels, height, width = voxel.shape
        rebuilt = torch.zeros_like(voxel, dtype=torch.float32)
        for sample_index in range(batch):
            xy = view["event_xy"][sample_index]
            ts = view["event_t"][sample_index]
            pol = view["event_p"][sample_index]
            count = int(xy.shape[0]); remove_count = min(int(round(count * ratio)), count)
            total += count; removed += remove_count
            if remove_count == 0:
                keep = torch.ones(count, dtype=torch.bool, device=xy.device)
            else:
                x = xy[:, 0].long().clamp(0, width - 1); y = xy[:, 1].long().clamp(0, height - 1)
                score = contribution[sample_index, view_index, y, x].float()
                if method == "remove_high": order = torch.argsort(score, descending=True, stable=True)
                elif method == "remove_low": order = torch.argsort(score, descending=False, stable=True)
                elif method == "remove_random":
                    generator = torch.Generator(device="cpu").manual_seed(seed + 100003 * view_index + 1009 * sample_index + int(ratio * 1000))
                    order = torch.randperm(count, generator=generator, device="cpu").to(xy.device)
                else: raise ValueError(method)
                keep = torch.ones(count, dtype=torch.bool, device=xy.device); keep[order[:remove_count]] = False
            kept_xy, kept_ts, kept_pol = xy[keep], ts[keep], pol[keep]
            view["event_xy"][sample_index] = kept_xy
            view["event_t"][sample_index] = kept_ts
            view["event_p"][sample_index] = kept_pol
            rebuilt[sample_index] = voxelize(
                kept_xy, kept_ts, kept_pol, view["event_time_range"][sample_index],
                channels, height, width,
            ).to(voxel.dtype)
        view["event_voxel"] = rebuilt
        view["has_event"] = torch.full_like(view["has_event"], bool(total > removed))
    return result, removed, total


def update(accumulator, output, views, cfg, name):
    depth = stack_output(output, "depth")
    gt = fe.stack_view_field(views, "depthmap").float()
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
    poses = fe.stack_view_field(views, "camera_pose").float()
    valid = fe.build_valid_mask(views, gt, depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)), depth_max=getattr(cfg.loss, "depth_max", None))
    _update_condition(accumulator, name, output, depth, gt, intrinsics, poses, valid)


@torch.inference_mode()
def evaluate_loader(model, loader, cfg, args, accumulators, counters, scene_index):
    device = next(model.parameters()).device
    enabled = args.amp != "none" and device.type == "cuda"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    for batch_index, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches: break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        views = attach_full_voxels(views, args.event_resize_bins)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            intact = model(views)
        update(accumulators["intact"], intact, views, cfg, "intact")
        contribution = stack_output(intact, "event_contribution").float().clamp(0, 1)
        if contribution.ndim != 4:
            raise RuntimeError(f"event_contribution must be [B,V,H,W], got {tuple(contribution.shape)}")
        for ratio in args.ratio_values:
            for method_index, method in enumerate(METHODS):
                key = f"{method}@{ratio:.2f}"
                repeats = args.random_repeats if method == "remove_random" else 1
                for repeat in range(repeats):
                    altered, removed, total = delete_by_score(
                        views, contribution, method, ratio,
                        args.seed + scene_index * 10000019 + batch_index * 10007
                        + method_index * 101 + repeat * 1000003,
                    )
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                        output = model(altered)
                    update(accumulators[key], output, altered, cfg, key)
                    counters[key]["removed"] += removed; counters[key]["total"] += total


def metric_delta(metric, baseline, value):
    if metric == "delta1": return baseline - value  # positive means degradation
    return value - baseline


def summarize(accumulators, counters, ratios):
    metrics = {key: value.compute() for key, value in accumulators.items()}
    baseline = metrics["intact"]
    rows = []
    for ratio in ratios:
        for method in METHODS:
            key = f"{method}@{ratio:.2f}"; value = metrics[key]
            row = {"condition": method, "removed_ratio_requested": ratio,
                   "removed_ratio_actual": counters[key]["removed"] / max(counters[key]["total"], 1)}
            row.update(value)
            for metric in ("mae", "abs_rel", "rmse_log", "delta1", "normal_mean_deg"):
                row[f"delta_{metric}"] = metric_delta(metric, baseline[metric], value[metric])
            terms = []
            for metric in ("abs_rel", "rmse_log", "normal_mean_deg"):
                if np.isfinite(baseline[metric]) and baseline[metric] > 1e-12:
                    terms.append(value[metric] / baseline[metric])
            row["geometry_error_normalized"] = float(np.mean(terms)) if terms else float("nan")
            row["delta_geometry"] = row["geometry_error_normalized"] - 1.
            rows.append(row)
    return baseline, rows


def save_outputs(out, checkpoint, args, baseline, rows):
    out.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": str(checkpoint), "event_source": "decomposition_full",
               "scenes": args.scene_names, "exposures": args.exposure_values,
               "random_repeats": args.random_repeats,
               "baseline_intact": baseline, "results": rows,
               "geometry_error_definition": "mean(AbsRel/AbsRel_intact, RMSElog/RMSElog_intact, Nmean/Nmean_intact)"}
    (out / "causal_deletion.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "causal_deletion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    specs = (("geometry_error_normalized", "Normalized geometry error"), ("abs_rel", "AbsRel"),
             ("rmse_log", "RMSElog"), ("normal_mean_deg", "Normal mean error (deg)"))
    colors = {"remove_high": "#d62728", "remove_random": "#ff7f0e", "remove_low": "#2ca02c"}
    for axis, (metric, title) in zip(axes.reshape(-1), specs):
        for method in METHODS:
            selected = [row for row in rows if row["condition"] == method]
            axis.plot([100 * row["removed_ratio_actual"] for row in selected], [row[metric] for row in selected], marker="o", label=method.replace("remove_", "remove "), color=colors[method])
        axis.set_xlabel("Removed raw events (%)"); axis.set_ylabel(title); axis.grid(alpha=.3); axis.legend()
    fig.suptitle("Contribution causal deletion: high should degrade fastest")
    fig.tight_layout(); fig.savefig(out / "causal_deletion_curves.png", dpi=180); plt.close(fig)


def main():
    args = parse_args(); args.ratio_values = [float(x) for x in args.ratios.split(",") if x.strip()]
    if any(x <= 0 or x >= 1 for x in args.ratio_values): raise ValueError("ratios must lie strictly between 0 and 1")
    args.exposure_values = [f"ev_{x.strip().removeprefix('ev_')}" for x in args.exposures.split(",") if x.strip()]
    checkpoint = Path(args.checkpoint).expanduser(); checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, cfg = build_model(checkpoint, device, args.depth_scale)
    OmegaConf.set_struct(cfg, False); OmegaConf.set_struct(cfg.data, False)
    cfg.data.event_source_mode = "decomposition_full"
    keys = ["intact"] + [f"{method}@{ratio:.2f}" for ratio in args.ratio_values for method in METHODS]
    accumulators = {key: ConditionAccumulator() for key in keys}
    counters = {key: defaultdict(int) for key in keys}
    for scene_index, scene in enumerate(args.scene_names):
        for exposure in args.exposure_values:
            print(f"[causal deletion] scene={scene} exposure={exposure}", flush=True)
            ns = SimpleNamespace(root=args.root, num_views=args.num_views, resolution=args.resolution,
                scene_names=[scene], initial_scene_idx=0, active_scene_count=1,
                test_frame_count=args.test_frame_count, ldr_event_id=exposure,
                # Preserve raw x/y/t/p. The evaluator itself voxelizes the
                # intact and deleted streams with the same linear-time kernel.
                event_resize_method="nearest", event_resize_bins=args.event_resize_bins,
                window_stride=args.window_stride, batch_size=1, num_workers=args.num_workers,
                pin_memory=False, max_batches=args.max_batches)
            dataset, loader = protocol.build_loader(cfg, ns)
            if list(dataset.get_active_scenes()) != [scene]: raise RuntimeError(f"loader did not select {scene}")
            evaluate_loader(model, loader, cfg, args, accumulators, counters, scene_index)
            del loader, dataset; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    baseline, rows = summarize(accumulators, counters, args.ratio_values)
    save_outputs(Path(args.output_dir), checkpoint, args, baseline, rows)
    for ratio in args.ratio_values:
        values = {row["condition"]: row["delta_geometry"] for row in rows if row["removed_ratio_requested"] == ratio}
        print(f"ratio={ratio:.0%} delta_geometry={values} expected high > random > low", flush=True)
    print(f"Saved causal deletion evidence to {Path(args.output_dir).resolve()}", flush=True)


if __name__ == "__main__": main()
