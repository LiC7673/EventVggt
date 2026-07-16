"""Streaming four-scene, all-EV evaluation for the fixed detail-first model."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

import finetune_event as fe
from ablation.eag3r_metrics_eval import (
    cfg_from_checkpoint, move_views_to_device, stack_output,
)
from event_branch_ablation.evaluate_event_contribution import (
    ConditionAccumulator, _update_condition,
)
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_alternating_detail_first_fixed_model import (
    AlternatingDetailFirstFixedModel,
)
import real_reliability_stage.evaluate_stage2_heldout as protocol


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
CONDITIONS = ("coarse_hdr_like", "final_event_refined")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--scene-names", nargs="+", default=list(SCENES))
    p.add_argument("--exposures", default="0,1,2,5,10")
    p.add_argument("--test-frame-count", type=int, default=120)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--event-resize-method", default="voxel_linear_time")
    p.add_argument("--event-resize-bins", type=int, default=5)
    p.add_argument(
        "--event-source-mode",
        choices=("decomposition_full", "cur_event", "cur_best", "current"),
        default="decomposition_full",
        help=("event file used as the model's inference event input; cur_event "
              "strictly reads cur_event/events.h5"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    p.add_argument("--depth-scale", type=float, default=2.0,
                   help="fixed prediction scale; never estimated from test GT")
    p.add_argument("--visualize-every", type=int, default=1)
    p.add_argument("--max-visuals-per-condition", type=int, default=0,
                   help="0 means save every selected batch without a cap")
    return p.parse_args()


def build_model(checkpoint, device, depth_scale):
    raw = torch_load(checkpoint)
    expected = AlternatingDetailFirstFixedModel.checkpoint_schema
    if raw.get("schema") != expected:
        raise ValueError(f"checkpoint schema={raw.get('schema')!r}, expected={expected!r}")
    cfg = cfg_from_checkpoint(raw, None); m = cfg.model
    model = AlternatingDetailFirstFixedModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        event_count_cmax=float(getattr(m, "event_count_cmax", 3.0)),
        pixel_refiner_hidden=int(getattr(m, "pixel_refiner_hidden", 64)),
        pixel_refine_log_limit=float(getattr(m, "pixel_refine_log_limit", .30)),
        pixel_refiner_delay=int(getattr(m, "pixel_refiner_delay", 500)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .0015)),
        alignment_confidence_tau=.10, hdr_token_bottleneck=256,
        hdr_warmup_steps=0, normal_refine_iterations=1, normal_refine_step_limit=.05,
        c_delay_steps=int(getattr(m, "c_delay_steps", 1000)),
        c_transition_steps=int(getattr(m, "c_transition_steps", 1000)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(raw)), strict=True)
    runtime = raw.get("runtime_state", {})
    model._dual_alignment_step = int(runtime.get("dual_alignment_step", 0))
    # A legacy/incomplete checkpoint may lack the Python-only counter. Do not
    # silently evaluate with the pixel refiner disabled in that case.
    if model._dual_alignment_step <= 500:
        print(
            f"[eval warning] dual_alignment_step={model._dual_alignment_step}; "
            "forcing fully deployed pixel refiner for final-model evaluation",
            flush=True,
        )
        model._dual_alignment_step = 1500
    model.set_confidence_stage("full")
    model.fixed_eval_depth_scale = float(depth_scale)
    return model.to(device).eval(), cfg


def _normal_rgb(normal, valid):
    image = ((normal.detach().float().cpu() + 1.0) * .5).clamp(0, 1)
    return image * valid.detach().float().cpu().unsqueeze(-1)


def save_visual(root, scene, exposure, index, views, output, depth_gt, valid,
                intrinsics, event_source_mode):
    coarse = stack_output(output, "depth_coarse")[0, 0].float()
    final = stack_output(output, "depth")[0, 0].float()
    gt = depth_gt[0, 0].float(); mask = valid[0, 0]
    coarse_n = fe.depth_to_normals(coarse[None, None], intrinsics[:1, :1])[0, 0]
    final_n = fe.depth_to_normals(final[None, None], intrinsics[:1, :1])[0, 0]
    gt_n = fe.depth_to_normals(gt[None, None], intrinsics[:1, :1])[0, 0]
    rgb = views[0]["img"][0].detach().float().permute(1, 2, 0).cpu().clamp(0, 1)
    event = views[0]["event_voxel"][0].detach().float().abs().sum(0).cpu()
    c_fusion = output.ress[0]["event_contribution"][0].detach().float().cpu()
    c_refine = output.ress[0]["normal_fusion_gate"][0].detach().float().cpu()
    valid_values = torch.cat((coarse[mask], final[mask], gt[mask]))
    vmin, vmax = float(valid_values.min()), float(valid_values.max())
    final_error = (final - gt).abs() * mask
    error_max = float(final_error.max().clamp_min(1e-6))
    panels = (
        (rgb, "LDR RGB", None, None, None),
        (event, f"event input ({event_source_mode})", "gray", None, None),
        (coarse.cpu() * mask.cpu(), "HDR-like coarse depth", "viridis", vmin, vmax),
        (final.cpu() * mask.cpu(), "event-refined final depth", "viridis", vmin, vmax),
        (gt.cpu() * mask.cpu(), "GT depth", "viridis", vmin, vmax),
        (final_error.cpu(), "|final depth - GT depth|", "magma", 0, error_max),
        (_normal_rgb(coarse_n, mask), "coarse normal", None, None, None),
        (_normal_rgb(final_n, mask), "final normal", None, None, None),
        (_normal_rgb(gt_n, mask), "GT normal", None, None, None),
        (c_fusion, "C_fusion", "magma", 0, 1),
        (c_refine, "C_refine", "magma", 0, 1),
    )
    fig, axes = plt.subplots(3, 4, figsize=(20, 15)); axes = axes.reshape(-1)
    for axis in axes: axis.axis("off")
    for axis, (image, title, cmap, lo, hi) in zip(axes, panels):
        shown = axis.imshow(image.numpy() if torch.is_tensor(image) else image,
                            cmap=cmap, vmin=lo, vmax=hi)
        axis.set_title(title); axis.axis("off")
        if cmap is not None: fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
    fig.suptitle(f"scene={scene} exposure={exposure} sample={index}")
    path = root / "visualizations" / scene / exposure / f"sample_{index:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


@torch.inference_mode()
def evaluate_loader(model, loader, cfg, args, device, accumulators, scene, exposure, out):
    count = visuals = 0
    for batch_index, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches: break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        depth_gt = fe.stack_view_field(views, "depthmap").float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
        poses = fe.stack_view_field(views, "camera_pose").float()
        valid = fe.build_valid_mask(views, depth_gt, depth_min=1e-6, depth_max=None)
        enabled = args.amp != "none" and device.type == "cuda"
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            output = model(views)
        values = {"coarse_hdr_like": stack_output(output, "depth_coarse"),
                  "final_event_refined": stack_output(output, "depth")}
        for name, depth in values.items():
            for accumulator in accumulators[name]:
                _update_condition(accumulator, name, output, depth, depth_gt,
                                  intrinsics, poses, valid)
        visual_budget = (args.max_visuals_per_condition <= 0
                         or visuals < args.max_visuals_per_condition)
        if (args.visualize_every > 0 and batch_index % args.visualize_every == 0
                and visual_budget):
            save_visual(out, scene, exposure, batch_index, views, output,
                        depth_gt, valid, intrinsics, args.event_source_mode)
            visuals += 1
        count += 1
    return count


def rows_for(scope, scene, exposure, values, batches):
    rows, computed = [], {}
    for name in CONDITIONS:
        metric = values[name].compute(); computed[name] = metric
        rows.append(dict(scope=scope, scene=scene, exposure=exposure,
                         condition=name, evaluated_batches=batches, **metric))
    return rows, computed


def write_progress(out, checkpoint, args, exposures, nested, aggregates,
                   overall_metrics, rows, complete=False):
    payload = dict(
        checkpoint=str(checkpoint), scenes=list(args.scene_names),
        exposures=exposures, results=nested,
        event_source_mode=args.event_source_mode,
        depth_alignment=f"fixed scale={args.depth_scale}; no test-GT alignment",
        all_scenes_pixel_weighted=aggregates,
        overall_pixel_weighted=overall_metrics,
        complete=bool(complete),
    )
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0].keys()), extrasaction="ignore"
            )
            writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args(); checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute(): checkpoint = ROOT / checkpoint
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, cfg = build_model(checkpoint, device, args.depth_scale)
    # Evaluation input selection must not silently inherit the training source
    # from the checkpoint.  In particular, cur_best is a strict isolated path
    # and never falls back to events_additive/full.
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.event_source_mode = args.event_source_mode
    print(f"[test protocol] event_source_mode={args.event_source_mode}", flush=True)
    exposures = [f"ev_{x.strip().removeprefix('ev_')}" for x in args.exposures.split(",") if x.strip()]
    totals = {e: {n: ConditionAccumulator() for n in CONDITIONS} for e in exposures}
    overall = {n: ConditionAccumulator() for n in CONDITIONS}; rows=[]; nested={}
    exposure_batches = {e: 0 for e in exposures}; all_batches=0
    for scene in args.scene_names:
        nested[scene] = {}
        for exposure in exposures:
            print(f"[test] scene={scene} exposure={exposure}", flush=True)
            ns = SimpleNamespace(root=args.root, num_views=args.num_views,
                resolution=args.resolution, scene_names=[scene], initial_scene_idx=0,
                active_scene_count=1, test_frame_count=args.test_frame_count,
                ldr_event_id=exposure, event_resize_method=args.event_resize_method,
                event_resize_bins=args.event_resize_bins, window_stride=args.window_stride,
                batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False,
                max_batches=args.max_batches)
            dataset, loader = protocol.build_loader(cfg, ns)
            active = list(dataset.get_active_scenes())
            if active != [scene]:
                raise RuntimeError(f"requested scene={scene!r}, loader selected {active!r}")
            local = {n: ConditionAccumulator() for n in CONDITIONS}
            fanout = {n: (local[n], totals[exposure][n], overall[n]) for n in CONDITIONS}
            batches = evaluate_loader(model, loader, cfg, args, device, fanout,
                                      scene, exposure, out)
            current_rows, metrics = rows_for("scene", scene, exposure, local, batches)
            rows += current_rows; nested[scene][exposure] = metrics
            exposure_batches[exposure] += batches; all_batches += batches
            # Persist after every scene/exposure. Long all-frame evaluation is
            # inspectable while running and remains usable after interruption.
            write_progress(
                out, checkpoint, args, exposures, nested, {}, {}, rows,
                complete=False,
            )
            print(f"  final MAE={metrics['final_event_refined']['mae']:.6f} "
                  f"AbsRel={metrics['final_event_refined']['abs_rel']:.6f} "
                  f"Nmean={metrics['final_event_refined']['normal_mean_deg']:.3f}", flush=True)
            del fanout, local, loader, dataset; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    aggregates={}
    for exposure in exposures:
        new_rows, metrics = rows_for("all_scenes_pixel_weighted", "ALL", exposure,
                                     totals[exposure], exposure_batches[exposure])
        rows += new_rows; aggregates[exposure] = metrics
    new_rows, all_metrics = rows_for("all_pixel_weighted", "ALL", "ALL", overall, all_batches)
    rows += new_rows
    write_progress(
        out, checkpoint, args, exposures, nested, aggregates, all_metrics, rows,
        complete=True,
    )
    print(f"Saved metrics and visualizations to {out.resolve()}", flush=True)


if __name__ == "__main__": main()
