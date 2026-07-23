"""Hardware benchmark for the untouched pure-RGB StreamVGGT baseline."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

import finetune_no_event as rgb
from fine_rgb.evaluate_rgb_pretrained_vs_finetuned import (
    build_model,
    make_loader,
    move_views,
)
from paired_token_reliability.benchmark_cur_event_hardware import (
    benchmark,
    parameter_statistics,
    profile_forward_flops,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ckpt/model.pt")
    parser.add_argument("--config", default="config/finetune_no_event.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument(
        "--scene", default="DH2_Socrates and Seneca_Car_Paint_Midnight"
    )
    parser.add_argument("--exposure", default="ev_2")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    config = Path(args.config)
    model, _ = build_model(config, checkpoint, device)

    # make_loader expects the common pure-RGB evaluator argument namespace.
    args.data_root = args.root
    args.scenes = [args.scene]
    args.width, args.height = args.resolution
    args.fps = 120
    args.seed = 0
    args.test_frame_count = max(8, args.num_views)
    args.pin_memory = False
    loader = make_loader(
        args,
        args.scene,
        args.exposure if args.exposure.startswith("ev_") else f"ev_{args.exposure}",
    )
    cpu_views = next(iter(loader))
    views = move_views(cpu_views, device)
    # Assert that this benchmark cannot accidentally consume events.
    views = rgb.drop_event_fields(views)

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
    measured = benchmark(
        model,
        views,
        device,
        warmup=args.warmup,
        repeats=args.repeats,
        amp=args.amp,
    )
    measured.update(profile_forward_flops(model, views, device, args.amp))
    payload = {
        "variant": "rgb_pretrained",
        "checkpoint": str(checkpoint),
        "input": {
            "scene": args.scene,
            "exposure": args.exposure,
            "batch_size": args.batch_size,
            "num_views": args.num_views,
            "resolution_wh": list(args.resolution),
            "precision": args.amp,
            "model_latency_excludes_dataloader": True,
            "event_input": False,
        },
        "hardware": hardware,
        "model": parameter_statistics(model),
        "benchmark": measured,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved benchmark to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
