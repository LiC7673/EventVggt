"""Unified 5-method, 5-exposure, scene-disjoint ablation evaluation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import finetune_event as fe
from ab_st1_st2 import METHODS
from ab_st1_st2.model import AblationStreamVGGT
from ablation.eag3r_metrics_eval import cfg_from_checkpoint, move_views_to_device, stack_output, strip_module_prefix, torch_load
from event_branch_ablation.evaluate_event_contribution import ConditionAccumulator, _update_condition
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset


ROOT = Path(__file__).resolve().parents[1]


class ContributionStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.map_count = 0
        self.spatial_std_total = 0.0

    def update(self, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1)
        if not flat.numel():
            return
        self.count += int(flat.numel())
        self.total += float(flat.sum().cpu())
        self.square_total += float(flat.square().sum().cpu())
        self.minimum = min(self.minimum, float(flat.min().cpu()))
        self.maximum = max(self.maximum, float(flat.max().cpu()))
        maps = value.detach().float().reshape(-1, value.shape[-2] * value.shape[-1])
        self.map_count += int(maps.shape[0])
        self.spatial_std_total += float(maps.std(dim=1, unbiased=False).sum().cpu())

    def compute(self):
        if self.count == 0:
            return {key: float("nan") for key in ("mean", "std", "min", "max")}
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return {
            "mean": mean,
            # C_std is explicitly spatial: a constant-per-frame map cannot
            # evade collapse reporting merely by changing its scalar between frames.
            "std": self.spatial_std_total / max(self.map_count, 1),
            "global_std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def _model_kwargs(cfg, stage1_checkpoint, method):
    model_cfg = cfg.model
    return dict(
        img_size=int(getattr(model_cfg, "img_size", 518)),
        patch_size=int(getattr(model_cfg, "patch_size", 14)),
        embed_dim=int(getattr(model_cfg, "embed_dim", 1024)),
        event_hidden_dim=int(getattr(model_cfg, "adapter_event_hidden_dim", 48)),
        head_frames_chunk_size=int(getattr(model_cfg, "head_frames_chunk_size", 2)),
        event_num_bins=int(getattr(model_cfg, "event_num_bins", 10)),
        event_count_cmax=float(getattr(model_cfg, "event_count_cmax", 3.0)),
        stage1_checkpoint=str(Path(stage1_checkpoint).resolve()),
        event_pyramid_channels=int(getattr(model_cfg, "adapter_event_pyramid_channels", 64)),
        adapter_hidden_channels=int(getattr(model_cfg, "adapter_hidden_channels", 128)),
        ablation_method=method,
        saturation_threshold=float(getattr(model_cfg, "ablation_saturation_threshold", 0.98)),
    )


def build_model(args, device):
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    fallback = str(ROOT / "config" / "finetune_event.yaml")
    raw = torch_load(checkpoint)
    cfg = cfg_from_checkpoint(raw, fallback)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    model = AblationStreamVGGT(**_model_kwargs(cfg, args.stage1_checkpoint, args.method))
    state = strip_module_prefix(fe.unwrap_state_dict(raw))
    message = model.load_state_dict(state, strict=False)
    if args.method != "rgb_only":
        missing = [key for key in message.missing_keys if "event_encoder" in key or "geometry_adapters" in key]
        if missing:
            raise RuntimeError(f"Stage-2 checkpoint lacks event adapter weights: {missing[:10]}")
    print(
        f"[load] method={args.method} missing={len(message.missing_keys)} "
        f"unexpected={len(message.unexpected_keys)} checkpoint={checkpoint}",
        flush=True,
    )
    return model.to(device).eval(), cfg, checkpoint


def build_scene_loader(args, cfg, scene_index: int, exposure: str):
    dataset = get_combined_dataset(
        root=args.root or str(cfg.data.root),
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=int(getattr(cfg.data, "fps", 120)),
        seed=int(getattr(cfg, "seed", 0)),
        scene_names=None,
        initial_scene_idx=scene_index,
        active_scene_count=1,
        split="test",
        test_frame_count=args.test_frame_count,
        min_train_start_id=0,
        ldr_event_id=exposure,
        event_spatial_transform=str(getattr(cfg.data, "event_spatial_transform", "auto")),
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_resize_bins,
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    indices = list(range(0, len(dataset), max(args.window_stride, 1)))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=event_multiview_collate,
    )
    return dataset, loader


def _map_image(value, cmap="viridis"):
    array = value.detach().float().cpu().numpy()
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape + (3,), dtype=np.float32)
    lo, hi = np.percentile(array[finite], [2, 98])
    normalized = np.clip((array - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    return plt.get_cmap(cmap)(normalized)[..., :3]


def _unit_image(value, cmap="magma"):
    array = value.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return plt.get_cmap(cmap)(array)[..., :3]


def save_visual(path: Path, views, output, depth_gt):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = views[0]["img"][0].detach().float().permute(1, 2, 0).cpu().numpy().clip(0, 1)
    event = views[0]["event_voxel"][0].detach().float().abs().sum(dim=0)
    result = output.ress[0]
    contribution = result["event_contribution"][0]
    selected = result["selected_event_mass"][0]
    pred = result["depth"][0].squeeze(-1)
    gt = depth_gt[0, 0]
    panels = (
        (rgb, "RGB input"),
        (_map_image(event, "gray"), "event projection"),
        (_unit_image(contribution, "magma"), "contribution [0,1]"),
        (_map_image(selected, "gray"), "selected event"),
        (_map_image(pred, "viridis"), "predicted depth"),
        (_map_image(gt, "viridis"), "GT depth"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def evaluate(args):
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, cfg, checkpoint = build_model(args, device)
    overall = ConditionAccumulator()
    contribution_stats = ContributionStats()
    per_exposure = {}
    scene_names = []
    total_batches = 0
    elapsed_model = 0.0
    output_root = Path(args.output_dir)

    for exposure in args.exposures:
        exposure_accumulator = ConditionAccumulator()
        for offset in range(args.test_scene_count):
            scene_index = args.test_initial_scene_idx + offset
            dataset, loader = build_scene_loader(args, cfg, scene_index, exposure)
            active_scene = dataset.get_active_scenes()[0]
            if active_scene not in scene_names:
                scene_names.append(active_scene)
            for batch_index, cpu_views in enumerate(loader):
                if args.max_batches_per_scene > 0 and batch_index >= args.max_batches_per_scene:
                    break
                views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
                depth_gt = fe.stack_view_field(views, "depthmap").float()
                intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
                gt_pose = fe.stack_view_field(views, "camera_pose").float()
                valid = fe.build_valid_mask(views, depth_gt)
                use_amp = args.amp != "none" and device.type == "cuda"
                amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output = model(views)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_model += time.perf_counter() - start
                depth = stack_output(output, "depth")
                if depth is None:
                    raise RuntimeError("Model did not return depth")
                _update_condition(overall, args.method, output, depth, depth_gt, intrinsics, gt_pose, valid)
                _update_condition(
                    exposure_accumulator, args.method, output, depth, depth_gt, intrinsics, gt_pose, valid
                )
                contribution = stack_output(output, "event_contribution")
                if contribution is None:
                    raise RuntimeError("Model did not expose event_contribution")
                contribution_stats.update(contribution)
                if batch_index == 0:
                    save_visual(
                        output_root / "visualizations" / active_scene / f"{exposure}.png",
                        views,
                        output,
                        depth_gt,
                    )
                total_batches += 1
        per_exposure[exposure] = exposure_accumulator.compute()

    c_stats = contribution_stats.compute()
    collapse = bool(np.isfinite(c_stats["std"]) and c_stats["std"] <= args.collapse_threshold)
    result = {
        "method": args.method,
        "checkpoint": str(checkpoint.resolve()),
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
        "train_scene_count": 20,
        "test_scene_count": args.test_scene_count,
        "test_scenes": scene_names,
        "exposures": args.exposures,
        "metrics": overall.compute(),
        "per_exposure": per_exposure,
        "contribution": {**c_stats, "collapse": collapse, "threshold": args.collapse_threshold},
        "runtime_seconds_per_batch": elapsed_model / max(total_batches, 1),
        "evaluated_batches": total_batches,
        "reference_exposure_used_at_test": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if collapse:
        print(f"[COLLAPSE] {args.method}: C_std={c_stats['std']:.8f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--test-initial-scene-idx", type=int, default=20)
    parser.add_argument("--test-scene-count", type=int, default=5)
    parser.add_argument("--test-frame-count", type=int, default=120)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--max-batches-per-scene", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", nargs=2, type=int, default=[518, 392])
    parser.add_argument("--exposures", nargs="+", default=["ev_0", "ev_1", "ev_2", "ev_5", "ev_10"])
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--collapse-threshold", type=float, default=1.0e-3)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
