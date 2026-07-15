"""Stream four named scenes/exposures through the C-confidence checkpoint.

Only one scene/exposure dataset exists at a time.  The model is loaded once.
Reported normals are the paper-protocol depth-derived normals; this model's
``normal`` output is produced from the same final depth and is therefore the
same geometric quantity.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import finetune_event as fe
from ablation.eag3r_metrics_eval import move_views_to_device, stack_output
from event_branch_ablation.evaluate_event_contribution import (
    ConditionAccumulator,
    _update_condition,
)
from paired_token_reliability.evaluate_linear_voxel_conditioned_confidence_refine import (
    build_model,
)
import real_reliability_stage.evaluate_stage2_heldout as protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)
CONDITIONS = ("coarse_rgb", "full_event")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--scene-names", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--exposures", default="0,1,2,5,10")
    parser.add_argument("--test-frame-count", type=int, default=120)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--event-resize-method", default="voxel_linear_time")
    parser.add_argument("--event-resize-bins", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--print-freq", type=int, default=20)
    return parser.parse_args()


def _loss_value(cfg, name, default):
    branch = getattr(cfg, "loss", None)
    return getattr(branch, name, default) if branch is not None else default


@torch.inference_mode()
def evaluate_loader(model, loader, cfg, args, device, accumulators):
    evaluated = 0
    start = time.time()
    for batch_index, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        depth_gt = fe.stack_view_field(views, "depthmap").float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
        gt_pose = fe.stack_view_field(views, "camera_pose").float()
        valid = fe.build_valid_mask(
            views,
            depth_gt,
            depth_min=float(_loss_value(cfg, "depth_min", 1.0e-6)),
            depth_max=_loss_value(cfg, "depth_max", None),
        )
        use_amp = args.amp != "none" and device.type == "cuda"
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(views)
        final_depth = stack_output(output, "depth")
        coarse_depth = stack_output(output, "depth_coarse")
        if final_depth is None or coarse_depth is None:
            raise RuntimeError("checkpoint must output both depth and depth_coarse")
        values = {"coarse_rgb": coarse_depth, "full_event": final_depth}
        for condition, depth in values.items():
            for accumulator in accumulators[condition]:
                _update_condition(
                    accumulator, condition, output, depth,
                    depth_gt, intrinsics, gt_pose, valid,
                )
        evaluated += 1
        if args.print_freq > 0 and evaluated % args.print_freq == 0:
            print(
                f"  [stream] batches={evaluated}/{len(loader)} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return evaluated


def metric_rows(scope, scene, exposure, accumulators, evaluated):
    rows = []
    metrics = {}
    for condition in CONDITIONS:
        current = accumulators[condition].compute()
        metrics[condition] = current
        rows.append({
            "scope": scope,
            "scene": scene,
            "exposure": exposure,
            "condition": condition,
            "evaluated_batches": evaluated,
            **current,
        })
    return rows, metrics


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, cfg = build_model(checkpoint, None, device)
    exposures = [
        f"ev_{item.strip().removeprefix('ev_')}"
        for item in args.exposures.split(",") if item.strip()
    ]
    exposure_totals = {
        exposure: {name: ConditionAccumulator() for name in CONDITIONS}
        for exposure in exposures
    }
    overall = {name: ConditionAccumulator() for name in CONDITIONS}
    exposure_batches = {exposure: 0 for exposure in exposures}
    overall_batches = 0
    rows, nested = [], {}

    for scene in args.scene_names:
        nested[scene] = {}
        for exposure in exposures:
            print(f"\n[scene={scene}] [exposure={exposure}] building loader", flush=True)
            loader_args = SimpleNamespace(
                root=args.root,
                num_views=args.num_views,
                resolution=args.resolution,
                scene_names=[scene],
                initial_scene_idx=0,
                active_scene_count=1,
                test_frame_count=args.test_frame_count,
                ldr_event_id=exposure,
                event_resize_method=args.event_resize_method,
                event_resize_bins=args.event_resize_bins,
                window_stride=args.window_stride,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=False,
                max_batches=args.max_batches,
            )
            dataset, loader = protocol.build_loader(cfg, loader_args)
            active = list(dataset.get_active_scenes())
            if active != [scene]:
                raise RuntimeError(f"requested scene {scene!r}, loader selected {active!r}")
            print(
                f"  windows={len(loader.dataset)} batches={len(loader)} active={active}",
                flush=True,
            )
            local = {name: ConditionAccumulator() for name in CONDITIONS}

            fanout = {
                name: (local[name], exposure_totals[exposure][name], overall[name])
                for name in CONDITIONS
            }
            evaluated = evaluate_loader(model, loader, cfg, args, device, fanout)
            current_rows, current_metrics = metric_rows(
                "scene", scene, exposure, local, evaluated
            )
            rows.extend(current_rows)
            nested[scene][exposure] = {
                "evaluated_batches": evaluated,
                "conditions": current_metrics,
            }
            exposure_batches[exposure] += evaluated
            overall_batches += evaluated
            print(
                f"  final MAE={current_metrics['full_event']['mae']:.6f} "
                f"AbsRel={current_metrics['full_event']['abs_rel']:.6f} "
                f"Nmean={current_metrics['full_event']['normal_mean_deg']:.4f}",
                flush=True,
            )
            del fanout, local, loader, dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate = {}
    for exposure in exposures:
        current_rows, current_metrics = metric_rows(
            "all_scenes_pixel_weighted", "ALL", exposure,
            exposure_totals[exposure], exposure_batches[exposure],
        )
        rows.extend(current_rows)
        aggregate[exposure] = current_metrics
    overall_rows, overall_metrics = metric_rows(
        "all_scenes_exposures_pixel_weighted", "ALL", "ALL",
        overall, overall_batches,
    )
    rows.extend(overall_rows)

    summary = {
        "checkpoint": str(checkpoint),
        "streaming_unit": "one scene and one exposure",
        "scenes": list(args.scene_names),
        "exposures": exposures,
        "event_resize_method": args.event_resize_method,
        "event_resize_bins": args.event_resize_bins,
        "results": nested,
        "all_scenes_pixel_weighted": aggregate,
        "overall_pixel_weighted": overall_metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        fieldnames = list(rows[0].keys())
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved streaming four-scene metrics to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
