"""Evaluate one unified checkpoint on ev_0/1/2/5/10 with full visual output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import real_reliability_stage.evaluate_stage2_heldout as protocol
from paired_token_reliability.evaluate_unified_geometry_contribution import build_model


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exposures", default="0,1,2,5,10")
    parser.add_argument("--root", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=12)
    parser.add_argument("--active-scene-count", type=int, default=4)
    parser.add_argument("--test-frame-count", type=int, default=120)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--visualize-every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, cfg = build_model(checkpoint, None, device)
    exposures = [f"ev_{value.strip().removeprefix('ev_')}" for value in args.exposures.split(",") if value.strip()]
    combined = {
        "checkpoint": str(checkpoint),
        "exposures": exposures,
        "initial_scene_idx": args.initial_scene_idx,
        "active_scene_count": args.active_scene_count,
        "results": {},
    }
    for exposure in exposures:
        exposure_dir = output_root / exposure
        current = SimpleNamespace(
            checkpoint=str(checkpoint), reliability_checkpoint=None,
            output_dir=str(exposure_dir), root=args.root,
            initial_scene_idx=args.initial_scene_idx,
            active_scene_count=args.active_scene_count,
            scene_names=None, test_frame_count=args.test_frame_count,
            window_stride=args.window_stride, num_views=args.num_views,
            resolution=args.resolution, ldr_event_id=exposure,
            event_resize_method=args.event_resize_method,
            event_resize_bins=args.event_resize_bins,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, max_batches=args.max_batches,
            device=args.device, amp="none", print_freq=args.print_freq,
            visualize_all=True, visualize_every=args.visualize_every,
        )
        dataset, loader = protocol.build_loader(cfg, current)
        print(
            f"\n[{exposure}] scenes={dataset.get_active_scenes()} "
            f"windows={len(loader.dataset)} batches={len(loader)}",
            flush=True,
        )
        metrics, records, evaluated, diagnostics = protocol.evaluate(
            model, loader, cfg, current, device
        )
        protocol.write_outputs(
            current, checkpoint, dataset, metrics, records, evaluated, diagnostics
        )
        combined["results"][exposure] = {
            "active_scenes": dataset.get_active_scenes(),
            "evaluated_batches": evaluated,
            "conditions": metrics,
            "diagnostics": diagnostics,
        }
    (output_root / "all_exposures_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nAll-exposure results saved to {output_root.resolve()}")


if __name__ == "__main__":
    main()

