"""Visualize temporal event bins from a trained event-branch checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from ablation.eag3r_metrics_eval import move_views_to_device, stack_output  # noqa: E402
from event_branch_ablation.evaluate_event_contribution import (  # noqa: E402
    build_heldout_loader,
    build_model,
    _condition_views,
)
from event_branch_ablation.visualization import save_event_bin_visuals  # noqa: E402


def temporal_bin_stats(voxel: torch.Tensor) -> dict:
    voxel = voxel.detach().float().cpu().clamp_min(0)
    source_bins = int(voxel.shape[0]) // 2
    positive = voxel[:source_bins]
    negative = voxel[source_bins : 2 * source_bins]
    activity = positive + negative
    flat = activity.flatten(1)
    energy = flat.sum(dim=1)
    nonzero = (flat > 0).float().mean(dim=1)
    height, width = activity.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    denominator = activity.sum(dim=(-2, -1)).clamp_min(1e-8)
    centroid_x = (activity * xx).sum(dim=(-2, -1)) / denominator
    centroid_y = (activity * yy).sum(dim=(-2, -1)) / denominator
    adjacent_cosine = []
    adjacent_iou = []
    exact_duplicate = []
    for index in range(max(source_bins - 1, 0)):
        left = flat[index]
        right = flat[index + 1]
        cosine = torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-8)
        left_mask = left > 0
        right_mask = right > 0
        union = (left_mask | right_mask).sum().clamp_min(1)
        iou = (left_mask & right_mask).sum().float() / union.float()
        adjacent_cosine.append(float(cosine))
        adjacent_iou.append(float(iou))
        exact_duplicate.append(float(torch.equal(left, right)))
    return {
        "num_bins": source_bins,
        "energy_per_bin": [float(value) for value in energy],
        "nonzero_ratio_per_bin": [float(value) for value in nonzero],
        "centroid_x_per_bin": [float(value) for value in centroid_x],
        "centroid_y_per_bin": [float(value) for value in centroid_y],
        "adjacent_cosine_mean": float(np.mean(adjacent_cosine)) if adjacent_cosine else float("nan"),
        "adjacent_iou_mean": float(np.mean(adjacent_iou)) if adjacent_iou else float("nan"),
        "exact_duplicate_ratio": float(np.mean(exact_duplicate)) if exact_duplicate else float("nan"),
    }


def configure_visualization(cfg, args) -> None:
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg, "vis"):
        cfg.vis = OmegaConf.create({})
    OmegaConf.set_struct(cfg.vis, False)
    cfg.output_dir = str(Path(args.output_dir))
    cfg.vis.event_bins_enabled = True
    cfg.vis.event_bins_count = int(args.event_bins_count)
    cfg.vis.event_bins_num_views = int(args.num_views)
    cfg.vis.event_bin_panel_width = int(args.panel_width)
    cfg.vis.sample_index = 0


@torch.inference_mode()
def visualize(model, cfg, loader, args, device: torch.device) -> None:
    saved = 0
    records = []
    for batch_idx, cpu_views in enumerate(loader):
        if saved >= args.max_samples:
            break
        cpu_views = fe.maybe_denormalize_views(cpu_views)
        views = move_views_to_device(cpu_views, device)
        use_amp = args.amp != "none" and device.type == "cuda"
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(_condition_views(views, "full_event"))

        aux = {}
        for name in ("geometry", "material", "noise"):
            value = stack_output(output, f"pred_event_{name}_token")
            if value is not None:
                aux[f"pred_event_{name}_token"] = value
        for frame_idx, view in enumerate(views):
            for label, key in (
                ("input", "event_voxel"),
                ("gt_geometry", "event_geometry_voxel"),
                ("gt_material", "event_material_voxel"),
                ("gt_noise", "event_noise_voxel"),
            ):
                value = view.get(key)
                if torch.is_tensor(value) and value.ndim == 4:
                    records.append(
                        {
                            "sample": batch_idx,
                            "view": frame_idx,
                            "stream": label,
                            **temporal_bin_stats(value[0]),
                        }
                    )
            for label, key in (
                ("pred_geometry", "pred_event_geometry_token"),
                ("pred_material", "pred_event_material_token"),
                ("pred_noise", "pred_event_noise_token"),
            ):
                value = aux.get(key)
                if torch.is_tensor(value) and value.ndim == 5:
                    records.append(
                        {
                            "sample": batch_idx,
                            "view": frame_idx,
                            "stream": label,
                            **temporal_bin_stats(value[0, frame_idx]),
                        }
                    )
        save_event_bin_visuals(
            fe,
            cfg,
            views,
            aux,
            global_step=batch_idx,
            sample_idx=0,
            vis_subdir="event_bin_visualization",
            force=True,
            filename_prefix=f"sample_{batch_idx:03d}_",
        )
        saved += 1
        print(f"[visualize] sample {saved}/{args.max_samples}")
    stats_path = Path(args.output_dir) / "temporal_bin_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    input_records = [record for record in records if record["stream"] == "input"]
    if input_records:
        print(
            "[temporal-audit] input "
            f"adjacent_cosine={np.mean([r['adjacent_cosine_mean'] for r in input_records]):.4f} "
            f"adjacent_iou={np.mean([r['adjacent_iou_mean'] for r in input_records]):.4f} "
            f"exact_duplicate={np.mean([r['exact_duplicate_ratio'] for r in input_records]):.4f}"
        )
    print(f"[temporal-audit] {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize event bins from an event-branch checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-kind", choices=["auto", "geometry", "decomposition"], default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--initial-scene-idx", type=int, default=12)
    parser.add_argument("--active-scene-count", type=int, default=1)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392])
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--additive-event-root", default="events_additive")
    parser.add_argument("--mask-dilate-kernel", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--event-bins-count", type=int, default=10)
    parser.add_argument("--panel-width", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT_DIR / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, cfg, kind = build_model(checkpoint, args, device)
    configure_visualization(cfg, args)
    dataset, loader = build_heldout_loader(cfg, args)
    print(
        f"[visualize] kind={kind} scenes={dataset.get_active_scenes()} "
        f"samples={len(dataset)} checkpoint={checkpoint}"
    )
    visualize(model, cfg, loader, args, device)
    print(f"[saved] {Path(args.output_dir) / 'event_bin_visualization' / 'event_bins'}")


if __name__ == "__main__":
    main()
