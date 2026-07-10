from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate
from multildr_token_exp.common import _build_token_model, _dataset
from paired_token_reliability.common import (
    as_uint8,
    build_reliability_target,
    move_views_to_device,
    strip_module_prefix,
    torch_load,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export paired-token geometry reliability labels.")
    parser.add_argument("--teacher", required=True, help="paired_token_full checkpoint-last.pth")
    parser.add_argument("--output", default="abl_event_exp/paired_token_reliability/labels")
    parser.add_argument("--ldr-ids", nargs="+", default=["ev_1", "ev_2", "ev_5", "ev_10"])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-scenes", type=int, default=2)
    parser.add_argument("--token-cosine-floor", type=float, default=0.80)
    parser.add_argument("--token-source", choices=("raw", "aligned"), default="raw")
    parser.add_argument(
        "--token-agreement-mode",
        choices=("fixed_floor", "robust_quantile"),
        default="robust_quantile",
    )
    parser.add_argument("--token-quantile-low", type=float, default=0.05)
    parser.add_argument("--token-quantile-high", type=float, default=0.95)
    parser.add_argument("--dilate-kernel", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument(
        "--min-start-id",
        type=int,
        default=0,
        help="Skip initial synthetic windows whose first event intervals are empty.",
    )
    return parser.parse_args()


def _teacher(checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict) or "cfg" not in checkpoint:
        raise RuntimeError("Teacher checkpoint must contain the resolved training cfg.")
    cfg = OmegaConf.create(checkpoint["cfg"])
    model = _build_token_model(cfg)
    raw_state = checkpoint.get("model", checkpoint)
    state = strip_module_prefix(raw_state)
    message = model.load_state_dict(state, strict=False)
    if any("exposure_token_adapter" in key for key in message.missing_keys):
        raise RuntimeError(f"Teacher is missing exposure token adapter weights: {message}")
    model.to(device).eval().requires_grad_(False)
    return model, cfg


@torch.no_grad()
def _tokens(model, views, source):
    images = fe.stack_view_field(views, "img")
    model.aggregator(images)
    if model._last_exposure_tokens is None:
        raise RuntimeError("Teacher aggregator did not expose paired tokens.")
    tokens = (
        model._last_raw_exposure_tokens
        if source == "raw"
        else model._last_exposure_tokens
    )
    if tokens is None:
        raise RuntimeError(f"Teacher did not expose {source} paired tokens.")
    return tokens.detach()


def _save_preview(path: Path, rgb, target, components):
    from PIL import Image

    rgb = rgb.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    panels = [(rgb * 255.0).astype(np.uint8)]
    for value in (components["event_support"], components["geometry"], components["token_agreement"], target):
        gray = as_uint8(value)
        panels.append(np.repeat(gray[..., None], 3, axis=-1))
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


def main():
    args = parse_args()
    output = Path(args.output)
    target_dir = output / "targets"
    preview_dir = output / "previews"
    target_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher, cfg = _teacher(Path(args.teacher), device)
    dataset = _dataset(cfg, "train", "random")
    scenes = list(dataset.get_active_scenes())
    val_scenes = set(scenes[-max(args.val_scenes, 0) :]) if args.val_scenes else set()
    pairs = list(itertools.combinations(args.ldr_ids, 2))
    records = []
    agreement_means = []
    agreement_stds = []
    cosine_means = []
    selected = [
        index
        for index in range(0, len(dataset), max(args.stride, 1))
        if int(dataset.start_img_ids[index][1]) >= int(args.min_start_id)
    ]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    for ordinal, dataset_index in enumerate(selected):
        scene = dataset.start_img_ids[dataset_index][0]
        ldr_a, ldr_b = pairs[ordinal % len(pairs)]
        sample_a = dataset[(dataset_index, 0, int(cfg.data.num_views), ldr_a)]
        sample_b = dataset[(dataset_index, 0, int(cfg.data.num_views), ldr_b)]
        views_a = move_views_to_device(fe.maybe_denormalize_views(event_multiview_collate([sample_a])), device)
        views_b = move_views_to_device(fe.maybe_denormalize_views(event_multiview_collate([sample_b])), device)
        event_a = fe.stack_view_field(views_a, "event_voxel")
        event_b = fe.stack_view_field(views_b, "event_voxel")
        if not torch.equal(event_a, event_b):
            raise RuntimeError(f"Paired exposures changed event voxels at dataset index {dataset_index}.")
        rgb_a = fe.stack_view_field(views_a, "img")
        rgb_b = fe.stack_view_field(views_b, "img")
        rgb_difference = float((rgb_a - rgb_b).abs().mean())
        if rgb_difference < 1.0e-5:
            raise RuntimeError(
                f"Paired LDR images are effectively identical at dataset index {dataset_index}: "
                f"{ldr_a} vs {ldr_b}, mean_abs_diff={rgb_difference:.3e}"
            )
        token_a = _tokens(teacher, views_a, args.token_source)
        token_b = _tokens(teacher, views_b, args.token_source)
        raw_cosine = torch.nn.functional.cosine_similarity(token_a.float(), token_b.float(), dim=-1)
        depth = fe.stack_view_field(views_a, "depthmap")
        intrinsics = fe.stack_view_field(views_a, "camera_intrinsics")
        target, weight, components = build_reliability_target(
            event_a,
            depth,
            intrinsics,
            token_a,
            token_b,
            token_cosine_floor=args.token_cosine_floor,
            token_agreement_mode=args.token_agreement_mode,
            token_quantile_low=args.token_quantile_low,
            token_quantile_high=args.token_quantile_high,
            dilate_kernel=args.dilate_kernel,
        )
        agreement = components["token_agreement"]
        agreement_means.append(float(agreement.mean()))
        agreement_stds.append(float(agreement.std()))
        cosine_means.append(float(raw_cosine.mean()))
        target_path = target_dir / f"{ordinal:06d}.npz"
        np.savez_compressed(
            target_path,
            target=as_uint8(target[0]),
            weight=as_uint8(weight[0]),
            event_support=as_uint8(components["event_support"][0]),
            geometry=as_uint8(components["geometry"][0]),
            token_agreement=as_uint8(components["token_agreement"][0]),
        )
        records.append(
            {
                "dataset_index": dataset_index,
                "scene": scene,
                "ldr_a": ldr_a,
                "ldr_b": ldr_b,
                "rgb_mean_abs_difference": rgb_difference,
                "target": str(target_path.relative_to(output)),
                "split": "val" if scene in val_scenes else "train",
            }
        )
        if ordinal < args.preview_count:
            _save_preview(
                preview_dir / f"{ordinal:04d}_{scene}.png",
                views_a[0]["img"][0],
                target[0, 0],
                {key: value[0, 0] for key, value in components.items()},
            )
        if ordinal % 25 == 0:
            print(f"exported {ordinal + 1}/{len(selected)} scene={scene} pair={ldr_a},{ldr_b}", flush=True)

    agreement_mean = float(np.mean(agreement_means))
    agreement_std = float(np.mean(agreement_stds))
    if agreement_std < 0.03 or agreement_mean > 0.95 or agreement_mean < 0.05:
        raise RuntimeError(
            "Degenerate token-agreement labels: "
            f"mean={agreement_mean:.4f}, mean_spatial_std={agreement_std:.4f}. "
            "Refusing to publish a misleading manifest."
        )
    manifest = {
        "label_schema_version": 2,
        "teacher": str(Path(args.teacher).resolve()),
        "dataset_root": str(cfg.data.root),
        "scene_names": scenes,
        "num_views": int(cfg.data.num_views),
        "resolution": list(cfg.data.resolution),
        "fps": int(cfg.data.fps),
        "event_resize_bins": int(cfg.data.event_resize_bins),
        "event_resize_method": str(cfg.data.event_resize_method),
        "event_y_flip": str(getattr(cfg.data, "event_y_flip", "auto")),
        "event_spatial_transform": str(getattr(cfg.data, "event_spatial_transform", "auto")),
        "test_frame_count": int(cfg.data.test_frame_count),
        "min_start_id": int(args.min_start_id),
        "token_source": args.token_source,
        "token_agreement_mode": args.token_agreement_mode,
        "token_quantile_low": float(args.token_quantile_low),
        "token_quantile_high": float(args.token_quantile_high),
        "token_agreement_mean": agreement_mean,
        "token_agreement_mean_spatial_std": agreement_std,
        "raw_token_cosine_mean": float(np.mean(cosine_means)),
        "records": records,
    }
    write_json(output / "manifest.json", manifest)
    print(f"done: records={len(records)} manifest={output / 'manifest.json'}")


if __name__ == "__main__":
    main()
