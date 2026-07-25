"""Evaluate one checkpoint along a ten-level E_geo -> E_full continuum.

This is a strictly inference-only protocol.  For level alpha, the event tensor
actually consumed by the model is

    E_alpha = (1 - alpha) * E_geo + alpha * E_full.

The geometry tensor is used only to construct E_alpha; it is not exposed as a
teacher or loss target during inference.  Each level uses one predeclared fixed
depth scale, linearly interpolated from 2.3 (geo) to 2.2 (full).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import finetune_event as fe
from ablation.eag3r_metrics_eval import move_views_to_device, stack_output
from eventvggt.datasets.my_event_dataset import (
    event_multiview_collate,
    get_combined_dataset,
)
from event_branch_ablation.evaluate_event_contribution import (
    ConditionAccumulator,
    _update_condition,
)
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import (
    build_model,
)


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
CONDITIONS = ("coarse_hdr_like", "final_event_refined")


def _cfg_value(branch, name, default):
    return getattr(branch, name, default) if branch is not None else default


def build_decomposition_loader(cfg, args):
    """Build a loader that returns both full and controlled-geometry voxels."""
    dataset = get_combined_dataset(
        root=args.root or str(cfg.data.root),
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=int(_cfg_value(cfg.data, "fps", 120)),
        seed=int(_cfg_value(cfg, "seed", 0)),
        scene_names=args.scene_names,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.active_scene_count,
        split="test",
        test_frame_count=args.test_frame_count,
        ldr_event_id=args.ldr_event_id,
        event_spatial_transform=str(
            _cfg_value(cfg.data, "event_spatial_transform", "auto")
        ),
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_resize_bins,
        event_source_mode="decomposition_full",
        decomposition_supervision=True,
        decomposition_event_root=str(
            _cfg_value(cfg.data, "decomposition_event_root", "events_additive")
        ),
        decomposition_geo_branch="geometry_motion",
        decomposition_full_branch=str(
            _cfg_value(cfg.data, "decomposition_full_branch", "full")
        ),
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    indices = list(range(0, len(dataset), max(args.window_stride, 1)))
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        collate_fn=event_multiview_collate,
    )
    return dataset, loader


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="exp_f/cur_event_refiner_first_1k_then_joint_gpu4/"
                "checkpoint-adapter-best.pth",
    )
    p.add_argument(
        "--output-dir",
        default="exp_f/cur_event_refiner_first_1k_then_joint_gpu4/"
                "test_geo_to_full_10level",
    )
    p.add_argument("--root", default=None)
    p.add_argument("--scene-names", nargs="+", default=list(SCENES))
    p.add_argument("--exposures", default="0,1,2,5,10")
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--geo-depth-scale", type=float, default=2.3)
    p.add_argument("--full-depth-scale", type=float, default=2.2)
    p.add_argument("--test-frame-count", type=int, default=120)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    p.add_argument("--device", default="cuda")
    p.add_argument("--visualize-every", type=int, default=1)
    p.add_argument("--save-every-view", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _normal_rgb(normal, valid):
    image = ((normal.detach().float().cpu() + 1.0) * 0.5).clamp(0, 1)
    return image * valid.detach().float().cpu().unsqueeze(-1)


def _output_map(output, key, view_index):
    item = output.ress[view_index]
    value = item.get(key)
    if value is None:
        return None
    value = value[0].detach().float().cpu()
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3:
        value = value.mean(0)
    return value


def save_visuals(root, scene, exposure, level, alpha, scale, batch_index,
                 views, output, depth_gt, valid, intrinsics, geo_events,
                 full_events, mixed_events, every_view):
    coarse_all = stack_output(output, "depth_coarse").float()
    final_all = stack_output(output, "depth").float()
    view_indices = range(len(views)) if every_view else (0,)
    for view_index in view_indices:
        coarse = coarse_all[0, view_index]
        final = final_all[0, view_index]
        gt = depth_gt[0, view_index]
        mask = valid[0, view_index]
        if not mask.any():
            continue
        intr = intrinsics[:, view_index:view_index + 1]
        coarse_n = fe.depth_to_normals(
            coarse[None, None], intr
        )[0, 0]
        final_n = fe.depth_to_normals(final[None, None], intr)[0, 0]
        gt_n = fe.depth_to_normals(gt[None, None], intr)[0, 0]
        normal_valid = fe.normal_stencil_valid_mask(
            mask[None, None], final[None, None], eps=1e-6
        )[0, 0]
        rgb = (
            views[view_index]["img"][0].detach().float()
            .permute(1, 2, 0).cpu().clamp(0, 1)
        )
        geo = geo_events[view_index][0].detach().float().abs().sum(0).cpu()
        full = full_events[view_index][0].detach().float().abs().sum(0).cpu()
        mixed = mixed_events[view_index][0].detach().float().abs().sum(0).cpu()
        values = torch.cat((coarse[mask], final[mask], gt[mask]))
        vmin, vmax = float(values.min()), float(values.max())
        error = (final - gt).abs() * mask
        error_max = float(torch.quantile(error[mask], 0.995).clamp_min(1e-6))
        c_fusion = _output_map(output, "event_contribution", view_index)
        c_refine = _output_map(output, "normal_fusion_gate", view_index)
        panels = [
            (rgb, "LDR RGB", None, None, None),
            (geo, "|E_geo|", "gray", None, None),
            (full, "|E_full|", "gray", None, None),
            (mixed, f"|E_mix| alpha={alpha:.4f}", "gray", None, None),
            (coarse.cpu() * mask.cpu(), "coarse depth", "viridis", vmin, vmax),
            (final.cpu() * mask.cpu(), "final depth", "viridis", vmin, vmax),
            (gt.cpu() * mask.cpu(), "GT depth", "viridis", vmin, vmax),
            (error.cpu(), "|final-GT|", "magma", 0, error_max),
            (_normal_rgb(coarse_n, normal_valid), "coarse normal", None, None, None),
            (_normal_rgb(final_n, normal_valid), "final normal", None, None, None),
            (_normal_rgb(gt_n, normal_valid), "GT normal", None, None, None),
        ]
        if c_fusion is not None:
            panels.append((c_fusion, "C_fusion", "magma", 0, 1))
        if c_refine is not None:
            panels.append((c_refine, "C_refine", "magma", 0, 1))
        columns = 4
        rows = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(20, 5 * rows))
        axes = np.asarray(axes).reshape(-1)
        for axis in axes:
            axis.axis("off")
        for axis, (image, title, cmap, lo, hi) in zip(axes, panels):
            shown = axis.imshow(
                image.numpy() if torch.is_tensor(image) else image,
                cmap=cmap, vmin=lo, vmax=hi,
            )
            axis.set_title(title)
            axis.axis("off")
            if cmap is not None:
                fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
        raw_instance = views[view_index].get("instance", f"batch_{batch_index:06d}")
        if isinstance(raw_instance, (list, tuple)):
            raw_instance = raw_instance[0]
        fig.suptitle(
            f"{scene} | {exposure} | level={level:02d} | "
            f"alpha={alpha:.4f} | fixed scale={scale:.4f}"
        )
        path = (
            root / "visualizations" / _safe_name(scene) / exposure
            / f"{_safe_name(raw_instance)}_b{batch_index:05d}_v{view_index:02d}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def _mix_events(views, alpha):
    geo_values, full_values, mixed_values = [], [], []
    for view_index, view in enumerate(views):
        full = view.get("event_voxel")
        geo = view.get("geometry_event_voxel")
        if full is None or geo is None:
            raise RuntimeError(
                f"view {view_index} lacks geo/full tensors: "
                f"event_voxel={full is not None}, geometry_event_voxel={geo is not None}. "
                "Use decomposition_full with geometry_motion supervision."
            )
        if full.shape != geo.shape:
            raise RuntimeError(
                f"geo/full shape mismatch in view {view_index}: "
                f"geo={tuple(geo.shape)} full={tuple(full.shape)}"
            )
        mixed = torch.lerp(geo, full, float(alpha))
        if not torch.isfinite(mixed).all():
            raise FloatingPointError(f"non-finite mixed event tensor at alpha={alpha}")
        geo_values.append(geo)
        full_values.append(full)
        mixed_values.append(mixed)
        view["event_voxel"] = mixed
        # Do not expose an oracle geometry teacher to the inference graph.
        view.pop("geometry_event_voxel", None)
        view.pop("contribution_target", None)
        view.pop("decomposition_valid", None)
    return geo_values, full_values, mixed_values


@torch.inference_mode()
def evaluate_loader(model, loader, args, device, accumulators, scene, exposure,
                    level_dir, level, alpha, scale):
    batches = 0
    for batch_index, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        depth_gt = fe.stack_view_field(views, "depthmap").float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
        poses = fe.stack_view_field(views, "camera_pose").float()
        valid = fe.build_valid_mask(views, depth_gt, depth_min=1e-6, depth_max=None)
        geo, full, mixed = _mix_events(views, alpha)
        enabled = args.amp != "none" and device.type == "cuda"
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            output = model(views)
        depths = {
            "coarse_hdr_like": stack_output(output, "depth_coarse"),
            "final_event_refined": stack_output(output, "depth"),
        }
        for condition, depth in depths.items():
            if depth is None:
                raise RuntimeError(f"model output lacks depth for {condition}")
            for accumulator in accumulators[condition]:
                _update_condition(
                    accumulator, condition, output, depth, depth_gt,
                    intrinsics, poses, valid,
                )
        if args.visualize_every > 0 and batch_index % args.visualize_every == 0:
            save_visuals(
                level_dir, scene, exposure, level, alpha, scale, batch_index,
                views, output, depth_gt, valid, intrinsics, geo, full, mixed,
                args.save_every_view,
            )
        batches += 1
    return batches


def rows_for(scope, scene, exposure, accumulators, batches, level, alpha, scale):
    rows, metrics = [], {}
    for condition in CONDITIONS:
        value = accumulators[condition].compute()
        metrics[condition] = value
        rows.append({
            "level": level, "alpha_full": alpha, "geo_weight": 1.0 - alpha,
            "full_weight": alpha, "depth_scale": scale, "scope": scope,
            "scene": scene, "exposure": exposure, "condition": condition,
            "evaluated_batches": batches, **value,
        })
    return rows, metrics


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_level(level_dir, checkpoint, args, level, alpha, scale, nested,
                aggregates, overall_metrics, rows, complete):
    payload = {
        "checkpoint": str(checkpoint), "training": False,
        "level": level, "levels": args.levels, "alpha_full": alpha,
        "geo_weight": 1.0 - alpha, "full_weight": alpha,
        "event_definition": "E_mix=(1-alpha)*E_geo+alpha*E_full",
        "depth_scale": scale,
        "depth_scale_protocol": "predeclared linear 2.3 geo -> 2.2 full",
        "scenes": list(args.scene_names), "exposures": args.exposures,
        "results": nested, "all_scenes_pixel_weighted": aggregates,
        "overall_pixel_weighted": overall_metrics, "complete": complete,
    }
    level_dir.mkdir(parents=True, exist_ok=True)
    (level_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(level_dir / "metrics.csv", rows)


def plot_all_metrics(out, aggregate_rows):
    plot_root = out / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    excluded = {
        "level", "alpha_full", "geo_weight", "full_weight", "depth_scale",
        "scope", "scene", "exposure", "condition", "evaluated_batches",
        "depth_pixels", "normal_pixels",
    }
    metrics = sorted({
        key for row in aggregate_rows for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float))
    })
    exposures = ("ALL", "ev_0", "ev_1", "ev_2", "ev_5", "ev_10")
    for metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
        any_finite = False
        for axis, exposure in zip(axes.reshape(-1), exposures):
            for condition, style in zip(CONDITIONS, ("--o", "-o")):
                selected = sorted(
                    (row for row in aggregate_rows
                     if row["exposure"] == exposure
                     and row["condition"] == condition),
                    key=lambda row: row["alpha_full"],
                )
                x = [row["alpha_full"] for row in selected]
                y = [row.get(metric, float("nan")) for row in selected]
                if any(np.isfinite(y)):
                    any_finite = True
                    axis.plot(x, y, style, label=condition)
            axis.set_title(exposure)
            axis.set_xlabel("full-event mixture alpha (0=geo, 1=full)")
            axis.set_ylabel(metric)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        if any_finite:
            fig.suptitle(f"{metric}: E_geo to E_full")
            fig.tight_layout()
            fig.savefig(plot_root / f"{_safe_name(metric)}.png", dpi=150)
        plt.close(fig)


def main():
    args = arguments()
    if args.levels < 2:
        raise ValueError("--levels must be >=2 so both geo and full endpoints exist")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    # Initial construction scale is overwritten before every level.
    model, cfg = build_model(checkpoint, device, args.geo_depth_scale)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.event_source_mode = "decomposition_full"
    cfg.data.decomposition_supervision = True
    cfg.data.decomposition_geo_branch = "geometry_motion"
    exposures = [
        f"ev_{value.strip().removeprefix('ev_')}"
        for value in args.exposures.split(",") if value.strip()
    ]
    all_rows = []
    aggregate_rows = []
    all_summaries = []
    for level in range(args.levels):
        alpha = level / float(args.levels - 1)
        scale = (
            (1.0 - alpha) * args.geo_depth_scale
            + alpha * args.full_depth_scale
        )
        if level == 0 and (alpha != 0.0 or scale != args.geo_depth_scale):
            raise AssertionError("geo endpoint is not exact")
        if level == args.levels - 1 and (
            alpha != 1.0 or scale != args.full_depth_scale
        ):
            raise AssertionError("full endpoint is not exact")
        model.fixed_eval_depth_scale = float(scale)
        model.eval()
        level_dir = out / (
            f"level_{level:02d}_alpha_{alpha:.4f}_scale_{scale:.4f}"
        )
        print(
            f"[geo->full] level={level:02d}/{args.levels - 1:02d} "
            f"alpha={alpha:.6f} scale={scale:.6f}", flush=True
        )
        totals = {
            exposure: {name: ConditionAccumulator() for name in CONDITIONS}
            for exposure in exposures
        }
        overall = {name: ConditionAccumulator() for name in CONDITIONS}
        exposure_batches = {exposure: 0 for exposure in exposures}
        level_rows, nested = [], {}
        level_batches = 0
        for scene in args.scene_names:
            nested[scene] = {}
            for exposure in exposures:
                print(
                    f"  [test] scene={scene} exposure={exposure}", flush=True
                )
                ns = SimpleNamespace(
                    root=args.root, num_views=args.num_views,
                    resolution=args.resolution, scene_names=[scene],
                    initial_scene_idx=0, active_scene_count=1,
                    test_frame_count=args.test_frame_count,
                    ldr_event_id=exposure,
                    event_resize_method="voxel_linear_time",
                    event_resize_bins=5, window_stride=args.window_stride,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=False, max_batches=args.max_batches,
                )
                dataset, loader = build_decomposition_loader(cfg, ns)
                active = list(dataset.get_active_scenes())
                if active != [scene]:
                    raise RuntimeError(
                        f"requested scene={scene!r}, loader selected {active!r}"
                    )
                local = {name: ConditionAccumulator() for name in CONDITIONS}
                fanout = {
                    name: (local[name], totals[exposure][name], overall[name])
                    for name in CONDITIONS
                }
                batches = evaluate_loader(
                    model, loader, args, device, fanout, scene, exposure,
                    level_dir, level, alpha, scale,
                )
                rows, metrics = rows_for(
                    "scene", scene, exposure, local, batches,
                    level, alpha, scale,
                )
                level_rows.extend(rows)
                nested[scene][exposure] = metrics
                exposure_batches[exposure] += batches
                level_batches += batches
                write_level(
                    level_dir, checkpoint, args, level, alpha, scale, nested,
                    {}, {}, level_rows, complete=False,
                )
                final = metrics["final_event_refined"]
                print(
                    f"    final AbsRel={final['abs_rel']:.6f} "
                    f"RMSElog={final['rmse_log']:.6f} "
                    f"d1={final['delta1']:.6f} "
                    f"Nmean={final['normal_mean_deg']:.3f}", flush=True
                )
                del fanout, local, loader, dataset
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        aggregates = {}
        for exposure in exposures:
            rows, metrics = rows_for(
                "all_scenes_pixel_weighted", "ALL", exposure,
                totals[exposure], exposure_batches[exposure],
                level, alpha, scale,
            )
            level_rows.extend(rows)
            aggregate_rows.extend(rows)
            aggregates[exposure] = metrics
        rows, overall_metrics = rows_for(
            "all_pixel_weighted", "ALL", "ALL", overall, level_batches,
            level, alpha, scale,
        )
        level_rows.extend(rows)
        aggregate_rows.extend(rows)
        all_rows.extend(level_rows)
        write_level(
            level_dir, checkpoint, args, level, alpha, scale, nested,
            aggregates, overall_metrics, level_rows, complete=True,
        )
        all_summaries.append({
            "level": level, "alpha_full": alpha, "depth_scale": scale,
            "all_scenes_pixel_weighted": aggregates,
            "overall_pixel_weighted": overall_metrics,
        })
        write_csv(out / "all_levels_metrics.csv", all_rows)
        (out / "all_levels_summary.json").write_text(
            json.dumps({
                "checkpoint": str(checkpoint), "training": False,
                "event_definition": "E_mix=(1-alpha)*E_geo+alpha*E_full",
                "levels": all_summaries, "complete": False,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    plot_all_metrics(out, aggregate_rows)
    (out / "all_levels_summary.json").write_text(
        json.dumps({
            "checkpoint": str(checkpoint), "training": False,
            "event_definition": "E_mix=(1-alpha)*E_geo+alpha*E_full",
            "levels": all_summaries, "complete": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved ten tables, visualizations and plots to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
