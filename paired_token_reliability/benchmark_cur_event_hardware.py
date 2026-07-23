"""Benchmark parameters, CUDA memory, latency and throughput of cur-event V2."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import finetune_event as fe
from ablation.eag3r_metrics_eval import move_views_to_device
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import (
    build_model as build_v2_model,
)
from paired_token_reliability import evaluate_latest_strategy_ablation as ablation_eval
import real_reliability_stage.evaluate_stage2_heldout as protocol


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth",
    )
    parser.add_argument("--output", default="exp_f/hardware_benchmark_cur_event_v2.json")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument(
        "--scene", default="DH2_Socrates and Seneca_Car_Paint_Midnight"
    )
    parser.add_argument("--exposure", default="ev_2")
    parser.add_argument("--event-source-mode", default="cur_event")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--event-resize-bins", type=int, default=5)
    parser.add_argument("--event-resize-method", default="voxel_linear_time")
    parser.add_argument("--depth-scale", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument(
        "--variant",
        choices=("full", "noisy_event_only", "multi_ldr_only", "without_refiner_normal"),
        default="full",
        help="Configure the checkpoint's actual ablation inference route.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parameter_statistics(model):
    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(
        parameter.numel() for _, parameter in parameters if parameter.requires_grad
    )
    extension_words = (
        "event_",
        "contribution",
        "full_geo",
        "ldr_event_hdr",
        "normal_fusion",
        "normal_depth_refiner",
        "pixel_depth_refiner",
    )
    extension_ids = {
        id(parameter)
        for name, parameter in parameters
        if any(word in name for word in extension_words)
    }
    extension = sum(
        parameter.numel()
        for _, parameter in parameters
        if id(parameter) in extension_ids
    )
    return {
        "parameters": int(total),
        "parameters_million": total / 1e6,
        "trainable_flag_parameters": int(trainable),
        "trainable_flag_parameters_million": trainable / 1e6,
        "event_extension_parameters": int(extension),
        "event_extension_parameters_million": extension / 1e6,
        "fp32_parameter_size_mb": total * 4 / 1024**2,
    }


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    alpha = position - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


@torch.inference_mode()
def profile_forward_flops(model, views, device, amp):
    """Best-effort PyTorch operator FLOP count for one forward pass."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    enabled = amp != "none" and device.type == "cuda"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=enabled
        ):
            output = model(views)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del output
    flops = sum(float(event.flops or 0) for event in profiler.key_averages())
    return {
        "profiled_flops": int(flops),
        "profiled_tflops_per_forward": flops / 1e12,
        "flops_note": (
            "PyTorch operator-profiler estimate; unsupported custom operators "
            "may be omitted. Use the same profiler for every compared method."
        ),
    }


@torch.inference_mode()
def benchmark(model, views, device, *, warmup, repeats, amp):
    if warmup < 1 or repeats < 1:
        raise ValueError("--warmup and --repeats must be positive")
    enabled = amp != "none" and device.type == "cuda"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            output = model(views)
        del output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        baseline_allocated = 0

    timings_ms = []
    for _ in range(repeats):
        if device.type == "cuda":
            start.record()
        else:
            begin = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            output = model(views)
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize(device)
            timings_ms.append(float(start.elapsed_time(end)))
        else:
            timings_ms.append((time.perf_counter() - begin) * 1000.0)
        del output

    mean_ms = statistics.fmean(timings_ms)
    result = {
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "latency_mean_ms": mean_ms,
        "latency_std_ms": (
            statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
        ),
        "latency_p50_ms": percentile(timings_ms, 0.50),
        "latency_p95_ms": percentile(timings_ms, 0.95),
        "samples_per_second": 1000.0 * len(views[0]["img"]) / mean_ms,
        "views_per_second": (
            1000.0 * len(views) * len(views[0]["img"]) / mean_ms
        ),
    }
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        result.update(
            {
                "baseline_allocated_gb": baseline_allocated / 1024**3,
                "peak_allocated_gb": peak_allocated / 1024**3,
                "incremental_inference_peak_gb": (
                    max(0, peak_allocated - baseline_allocated) / 1024**3
                ),
                "peak_reserved_gb": (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                ),
            }
        )
    return result


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if args.variant == "full":
        model, cfg = build_v2_model(checkpoint, device, args.depth_scale)
        model.set_confidence_stage("full")
        model.disable_pixel_refiner = False
    else:
        ablation_eval._VARIANT = args.variant
        model, cfg = ablation_eval.build_model(
            checkpoint, device, args.depth_scale
        )
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.event_source_mode = args.event_source_mode
    namespace = SimpleNamespace(
        root=args.root,
        num_views=args.num_views,
        resolution=args.resolution,
        scene_names=[args.scene],
        initial_scene_idx=0,
        active_scene_count=1,
        test_frame_count=max(args.num_views, 8),
        ldr_event_id=(
            args.exposure
            if str(args.exposure).startswith("ev_")
            else f"ev_{args.exposure}"
        ),
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_resize_bins,
        window_stride=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        max_batches=1,
    )
    dataset, loader = protocol.build_loader(cfg, namespace)
    active = list(dataset.get_active_scenes())
    if active != [args.scene]:
        raise RuntimeError(f"requested {args.scene!r}, loader selected {active!r}")
    cpu_views = next(iter(loader))
    views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)

    hardware = {
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        hardware.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_gb": (
                    torch.cuda.get_device_properties(device).total_memory / 1024**3
                ),
            }
        )
    payload = {
        "checkpoint": str(checkpoint),
        "variant": args.variant,
        "input": {
            "scene": args.scene,
            "exposure": namespace.ldr_event_id,
            "batch_size": args.batch_size,
            "num_views": args.num_views,
            "resolution_wh": list(args.resolution),
            "event_bins": args.event_resize_bins,
            "precision": args.amp,
            "model_latency_excludes_dataloader": True,
        },
        "hardware": hardware,
        "model": parameter_statistics(model),
        "benchmark": benchmark(
            model,
            views,
            device,
            warmup=args.warmup,
            repeats=args.repeats,
            amp=args.amp,
        ),
    }
    payload["benchmark"].update(profile_forward_flops(model, views, device, args.amp))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved benchmark to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
