"""Evaluate Stage-2 reliability refinement on four scene-disjoint sequences.

The same checkpoint is tested under coarse-RGB, zero-event, real-event,
reverse-time, and swapped-polarity conditions. Full-vs-zero is the causal event
gain; final-vs-coarse also contains the RGB/depth capacity of the refiner.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from ablation.eag3r_metrics_eval import (  # noqa: E402
    cfg_from_checkpoint,
    move_views_to_device,
    stack_output,
    strip_module_prefix,
    torch_load,
)
from event_branch_ablation.evaluate_event_contribution import (  # noqa: E402
    ConditionAccumulator,
    _comparison,
    _paired_ci,
    _sample_metrics,
    _update_condition,
)
from eventvggt.datasets.my_event_dataset import (  # noqa: E402
    event_multiview_collate,
    get_combined_dataset,
)
from eventvggt.models.streamvggt_pretrained_reliability_detail import StreamVGGT  # noqa: E402


CONDITIONS = ("coarse_rgb", "zero_event", "full_event", "reverse_time", "swap_polarity")


def _cfg_value(branch, name: str, default):
    return getattr(branch, name, default)


def build_model(checkpoint: Path, reliability_checkpoint: str | None, device: torch.device):
    raw_checkpoint = torch_load(checkpoint)
    cfg = cfg_from_checkpoint(raw_checkpoint, None)
    OmegaConf.set_struct(cfg, False)
    for name in ("model", "data", "loss"):
        if hasattr(cfg, name):
            OmegaConf.set_struct(getattr(cfg, name), False)

    model_cfg = cfg.model
    reliability_source = Path(reliability_checkpoint).expanduser() if reliability_checkpoint else checkpoint
    if not reliability_source.is_absolute():
        reliability_source = ROOT_DIR / reliability_source
    model = StreamVGGT(
        img_size=int(_cfg_value(model_cfg, "img_size", 518)),
        patch_size=int(_cfg_value(model_cfg, "patch_size", 14)),
        embed_dim=int(_cfg_value(model_cfg, "embed_dim", 1024)),
        event_hidden_dim=int(_cfg_value(model_cfg, "event_hidden_dim", 16)),
        head_frames_chunk_size=int(_cfg_value(model_cfg, "head_frames_chunk_size", 2)),
        event_num_bins=int(_cfg_value(model_cfg, "event_num_bins", 10)),
        event_count_cmax=float(_cfg_value(model_cfg, "event_count_cmax", 3.0)),
        residual_scale=float(_cfg_value(model_cfg, "refiner_residual_scale", 0.035)),
        residual_highpass_kernel=int(_cfg_value(model_cfg, "event_delta_highpass_kernel", 9)),
        residual_patch_zero_mean=bool(_cfg_value(model_cfg, "event_delta_patch_zero_mean", True)),
        residual_patch_size=int(_cfg_value(model_cfg, "event_delta_patch_size", 14)),
        residual_abs_limit=float(_cfg_value(model_cfg, "event_delta_abs_limit", 0.025)),
        refine_points=bool(_cfg_value(model_cfg, "refiner_refine_points", True)),
        use_checkpoint=False,
        reliability_checkpoint=str(reliability_source),
        reliability_base_channels=int(_cfg_value(model_cfg, "reliability_base_channels", 32)),
        reliability_gate_floor=float(_cfg_value(model_cfg, "reliability_gate_floor", 0.10)),
        reliability_frame_chunk_size=int(_cfg_value(model_cfg, "reliability_frame_chunk_size", 1)),
        reliability_rgb_input_range=str(
            _cfg_value(model_cfg, "reliability_rgb_input_range", "minus_one_one")
        ),
        residual_postfilter_kernel=int(_cfg_value(model_cfg, "residual_postfilter_kernel", 3)),
        residual_postfilter_strength=float(
            _cfg_value(model_cfg, "residual_postfilter_strength", 0.75)
        ),
    )
    state = strip_module_prefix(fe.unwrap_state_dict(raw_checkpoint))
    message = model.load_state_dict(state, strict=False)
    print(
        f"[load] missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)} "
        f"checkpoint={checkpoint}"
    )
    if message.missing_keys:
        print(f"[load] missing sample: {message.missing_keys[:8]}")
    if message.unexpected_keys:
        print(f"[load] unexpected sample: {message.unexpected_keys[:8]}")
    model.to(device).eval()
    return model, cfg


def build_loader(cfg, args):
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
        event_spatial_transform=str(_cfg_value(cfg.data, "event_spatial_transform", "auto")),
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_resize_bins,
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    indices = list(range(0, len(dataset), max(args.window_stride, 1)))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    return dataset, loader


def condition_views(views: List[Dict], condition: str) -> List[Dict]:
    if condition in {"full_event", "coarse_rgb"}:
        return views
    conditioned = []
    for view in views:
        current = dict(view)
        voxel = view["event_voxel"]
        if condition == "zero_event":
            current["event_voxel"] = torch.zeros_like(voxel)
        else:
            bins = voxel.shape[1] // 2
            pos = voxel[:, :bins]
            neg = voxel[:, bins : 2 * bins]
            if condition == "reverse_time":
                current["event_voxel"] = torch.cat([pos.flip(1), neg.flip(1)], dim=1)
            elif condition == "swap_polarity":
                current["event_voxel"] = torch.cat([neg, pos], dim=1)
            else:
                raise ValueError(f"Unknown condition: {condition}")
        conditioned.append(current)
    return conditioned


@torch.inference_mode()
def evaluate(model, loader, cfg, args, device):
    accumulators = {name: ConditionAccumulator() for name in CONDITIONS}
    records = []
    evaluated = 0
    start = time.time()
    for batch_idx, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        depth_gt = fe.stack_view_field(views, "depthmap").float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
        gt_pose = fe.stack_view_field(views, "camera_pose").float()
        valid = fe.build_valid_mask(
            views,
            depth_gt,
            depth_min=float(_cfg_value(cfg.loss, "depth_min", 1.0e-6)),
            depth_max=_cfg_value(cfg.loss, "depth_max", None),
        )
        use_amp = args.amp != "none" and device.type == "cuda"
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        cache = {}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            full_output = model(condition_views(views, "full_event"))
        full_depth = stack_output(full_output, "depth")
        coarse_depth = stack_output(full_output, "depth_coarse")
        cache["full_event"] = (full_output, full_depth)
        cache["coarse_rgb"] = (full_output, coarse_depth)
        for condition in ("zero_event", "reverse_time", "swap_polarity"):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(condition_views(views, condition))
            cache[condition] = (output, stack_output(output, "depth"))

        for condition in CONDITIONS:
            output, depth = cache[condition]
            if depth is None:
                raise RuntimeError(f"Missing depth for condition={condition}")
            _update_condition(
                accumulators[condition], condition, output, depth, depth_gt, intrinsics, gt_pose, valid
            )
            records.append(
                {
                    "batch_index": batch_idx,
                    "condition": condition,
                    **_sample_metrics(depth, depth_gt, intrinsics, valid),
                }
            )
        evaluated += 1
        if (batch_idx + 1) % args.print_freq == 0:
            print(f"[eval] {batch_idx + 1}/{len(loader)} elapsed={time.time() - start:.1f}s")
    return {name: accumulator.compute() for name, accumulator in accumulators.items()}, records, evaluated


def write_outputs(args, checkpoint, dataset, metrics, records, evaluated):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {
        "refiner_capacity_zero_vs_coarse": ("coarse_rgb", "zero_event"),
        "full_event_net_gain": ("zero_event", "full_event"),
        "real_vs_reverse_time": ("reverse_time", "full_event"),
        "real_vs_swap_polarity": ("swap_polarity", "full_event"),
    }
    comparisons = {}
    for name, (baseline, candidate) in pairs.items():
        comparisons[name] = {
            "baseline": baseline,
            "candidate": candidate,
            **_comparison(metrics[baseline], metrics[candidate]),
            "paired_ci": _paired_ci(records, baseline, candidate),
        }
    summary = {
        "checkpoint": str(checkpoint),
        "active_scenes": dataset.get_active_scenes(),
        "initial_scene_idx": args.initial_scene_idx,
        "active_scene_count": args.active_scene_count,
        "num_views": args.num_views,
        "window_stride": args.window_stride,
        "evaluated_batches": evaluated,
        "conditions": metrics,
        "comparisons": comparisons,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    rows = [{"condition": name, **values} for name, values in metrics.items()]
    with (out_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "per_batch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print("\nHeld-out Stage-2 evaluation")
    print("condition       AbsRel   delta1  RMSElog  normal(deg)  <11.25   ATE")
    for name, values in metrics.items():
        print(
            f"{name:14s} {values['abs_rel']:8.5f} {values['delta1']:7.4f} "
            f"{values['rmse_log']:8.5f} {values['normal_mean_deg']:12.4f} "
            f"{values['normal_11_25']:8.4f} {values['ate']:8.5f}"
        )
    gain = comparisons["full_event_net_gain"]
    print(
        "\nFull-event gain over zero-event: "
        f"normal={gain['normal_mean_deg_reduction']:.4f} deg, "
        f"AbsRel={gain['abs_rel_reduction']:.6f}, "
        f"delta1={gain['delta1_increase']:.6f}"
    )
    print(f"Saved to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reliability-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--initial-scene-idx", type=int, default=12)
    parser.add_argument("--active-scene-count", type=int, default=4)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--test-frame-count", type=int, default=120)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--print-freq", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT_DIR / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, cfg = build_model(checkpoint, args.reliability_checkpoint, device)
    dataset, loader = build_loader(cfg, args)
    print(
        f"[dataset] scenes={dataset.get_active_scenes()} windows={len(loader.dataset)} "
        f"batches={len(loader)} stride={args.window_stride}"
    )
    metrics, records, evaluated = evaluate(model, loader, cfg, args, device)
    write_outputs(args, checkpoint, dataset, metrics, records, evaluated)


if __name__ == "__main__":
    main()

