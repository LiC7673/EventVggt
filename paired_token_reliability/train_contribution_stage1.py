"""Train Stage-1 Multi-LDR event contribution and selected-event refinement.

Example:

    python -m paired_token_reliability.train_contribution_stage1 \
      --config config/finetune_event.yaml \
      --pretrained ckpt/model.pt \
      --output abl_event_exp/event_contribution_stage1 \
      data.root=/data/reflective_raw data.active_scene_count=12

The frozen RGB model is never placed in the optimizer.  Coarse depth, normals,
and patch features are detached, so the supervision target cannot drift while
Stage 1 is being trained.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_event as fe
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.models.aggregator import Aggregator
from paired_token_reliability.common import (
    infer_patch_grid,
    move_views_to_device,
    strip_module_prefix,
    torch_load,
)
from paired_token_reliability.contribution_dataset import (
    MultiLdrContributionDataset,
    contribution_pair_collate,
    generate_ordered_pairs,
    parse_exposure_sequence,
)
from paired_token_reliability.contribution_stage1 import (
    MultiLdrEventContributionModel,
    build_bridge_masks,
    geometry_emphasis_weight,
    orient_exposure_pair,
    stage1_contribution_loss,
)


ROOT = Path(__file__).resolve().parents[1]


class FrozenRGBGeometryExtractor(nn.Module):
    """The two pretrained modules needed by Stage 1, without unused event/pose heads."""

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg.model
        self.head_frames_chunk_size = int(getattr(model_cfg, "head_frames_chunk_size", 2))
        self.aggregator = Aggregator(
            img_size=int(model_cfg.img_size),
            patch_size=int(model_cfg.patch_size),
            embed_dim=int(model_cfg.embed_dim),
        )
        self.depth_head = DPTHead(
            dim_in=2 * int(model_cfg.embed_dim),
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "finetune_event.yaml"))
    parser.add_argument("--pretrained", default=None, help="Frozen RGB-VGGT checkpoint")
    parser.add_argument("--output", default="abl_event_exp/event_contribution_stage1")
    parser.add_argument(
        "--exposures",
        default="0,1,2,5,10",
        help="Comma-separated exposure levels in strictly increasing order",
    )
    parser.add_argument(
        "--pair-mode",
        choices=("all", "adjacent", "anchor", "explicit"),
        default="all",
        help="How to generate ordered lower-to-higher exposure pairs",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="Repeat for explicit mode, for example --pair 'ev_0->ev_5'",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "proxy", "contribution", "joint"),
        default="all",
        help="Run the full A/B/C schedule or one isolated phase",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Legacy override for the selected phase")
    parser.add_argument("--epochs-proxy", type=int, default=5)
    parser.add_argument("--epochs-contribution", type=int, default=15)
    parser.add_argument("--epochs-joint", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lr-proxy", type=float, default=None)
    parser.add_argument("--lr-contribution", type=float, default=None)
    parser.add_argument("--joint-refiner-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--contribution-channels", type=int, default=32)
    parser.add_argument("--refiner-channels", type=int, default=32)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)
    parser.add_argument("--reference-gradient-threshold", type=float, default=0.02)
    parser.add_argument("--minimum-bridge-area", type=float, default=0.002)
    parser.add_argument("--minimum-saturation-gap", type=float, default=0.02)
    parser.add_argument(
        "--supervision-region",
        choices=("bridge", "event_support"),
        default="bridge",
        help="event_support is the required no-bridge ablation, not the full method",
    )
    parser.add_argument("--geometry-alpha", type=float, default=2.0)
    parser.add_argument("--geometry-depth-gradient-weight", type=float, default=0.5)
    parser.add_argument("--normal-weight", type=float, default=0.25)
    parser.add_argument("--budget-weight", type=float, default=0.05)
    parser.add_argument("--budget-ratio", type=float, default=0.50)
    parser.add_argument("--max-log-depth-delta", type=float, default=0.20)
    parser.add_argument("--max-normal-delta", type=float, default=0.50)
    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint interval in epochs")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--proxy-checkpoint", default=None)
    parser.add_argument("--contribution-checkpoint", default=None)
    parser.add_argument("--collapse-std-threshold", type=float, default=0.01)
    return parser


def load_config(path: str, overrides: Iterable[str]):
    cfg = OmegaConf.load(path)
    overrides = list(overrides)
    invalid = [value for value in overrides if "=" not in value]
    if invalid:
        raise ValueError(f"Unrecognized arguments (expected key=value config overrides): {invalid}")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def make_dataset(cfg, split: str, pairs):
    data = cfg.data
    return MultiLdrContributionDataset(
        root=str(data.root),
        split=split,
        num_views=int(data.num_views),
        resolution=tuple(data.resolution),
        fps=int(data.fps),
        seed=int(cfg.seed),
        ordered_pairs=pairs,
        scene_names=list(data.scene_names) if getattr(data, "scene_names", None) else None,
        initial_scene_idx=int(getattr(data, "initial_scene_idx", 0)),
        active_scene_count=int(getattr(data, "active_scene_count", 3)),
        test_frame_count=int(getattr(data, "test_frame_count", 10)),
        min_train_start_id=int(getattr(data, "min_train_start_id", 0)),
        event_y_flip=getattr(data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(data, "event_spatial_transform", "auto"),
        event_resize_method=str(getattr(data, "event_resize_method", "voxel_antialias")),
        event_resize_bins=int(getattr(data, "event_resize_bins", 10)),
    )


def make_loader(dataset, *, batch_size: int, num_workers: int, train: bool):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=contribution_pair_collate,
    )


def build_frozen_rgb_model(cfg, checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    model = FrozenRGBGeometryExtractor(cfg)
    checkpoint = strip_module_prefix(fe.unwrap_state_dict(torch_load(checkpoint_path)))
    result = model.load_state_dict(checkpoint, strict=False)
    print(
        f"[RGB checkpoint] missing={len(result.missing_keys)} "
        f"unused={len(result.unexpected_keys)}"
    )
    required_missing = [
        key for key in result.missing_keys if key.startswith(("aggregator.", "depth_head."))
    ]
    if required_missing:
        raise RuntimeError(
            "The RGB checkpoint is missing required coarse-geometry weights: "
            f"{required_missing[:10]}"
        )
    model.requires_grad_(False).eval().to(device=device, dtype=dtype)
    return model


@torch.no_grad()
def frozen_rgb_geometry(
    model: FrozenRGBGeometryExtractor,
    bad_rgb: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run only the frozen RGB aggregator/depth head and expose patch features."""
    model_dtype = next(model.parameters()).dtype
    images = bad_rgb.to(dtype=model_dtype)
    tokens_list, patch_start_index = model.aggregator(images)
    depth, _ = model.depth_head(
        tokens_list,
        images=images,
        patch_start_idx=patch_start_index,
        frames_chunk_size=model.head_frames_chunk_size,
    )
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.float().clamp_min(1.0e-6)
    normals = fe.depth_to_normals(depth, intrinsics.float())

    patch_tokens = tokens_list[-1][:, :, patch_start_index:].detach()
    grid_height, grid_width = infer_patch_grid(
        patch_tokens.shape[2], bad_rgb.shape[-2], bad_rgb.shape[-1]
    )
    features = patch_tokens.reshape(
        patch_tokens.shape[0], patch_tokens.shape[1], grid_height, grid_width, patch_tokens.shape[-1]
    ).permute(0, 1, 4, 2, 3)
    return {"depth": depth, "normals": normals, "features": features}


def _stack(views, key: str) -> torch.Tensor:
    return fe.stack_view_field(views, key)


def prepare_pair(batch, device: torch.device, args):
    views_a = fe.maybe_denormalize_views(move_views_to_device(batch["views_a"], device))
    views_b = fe.maybe_denormalize_views(move_views_to_device(batch["views_b"], device))
    rgb_a = _stack(views_a, "img").float().clamp(0.0, 1.0)
    rgb_b = _stack(views_b, "img").float().clamp(0.0, 1.0)
    rgb_reference, rgb_bad, ref_is_a, sat_ref, sat_bad = orient_exposure_pair(
        rgb_a,
        rgb_b,
        saturation_threshold=args.saturation_threshold,
    )
    event_a = _stack(views_a, "event_voxel").float()
    event_b = _stack(views_b, "event_voxel").float()
    if event_a.shape != event_b.shape or not torch.equal(event_a, event_b):
        raise RuntimeError(
            "Paired exposures do not contain exactly the same event voxel. "
            "Stage-1 contribution supervision would be invalid."
        )
    depth_gt = _stack(views_a, "depthmap").float()
    depth_b = _stack(views_b, "depthmap").float()
    if depth_gt.shape != depth_b.shape or not torch.allclose(depth_gt, depth_b, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("Paired exposures do not share identical GT depth.")
    intrinsics = _stack(views_a, "camera_intrinsics").float()
    valid_mask = fe.build_valid_mask(views_a, depth_gt)
    normal_gt = fe.depth_to_normals(depth_gt, intrinsics)
    bridge = build_bridge_masks(
        rgb_reference,
        rgb_bad,
        event_a,
        saturation_threshold=args.saturation_threshold,
        reference_gradient_threshold=args.reference_gradient_threshold,
    )
    return {
        "views_a": views_a,
        "views_b": views_b,
        "rgb_reference": rgb_reference,
        "rgb_bad": rgb_bad,
        "ref_is_a": ref_is_a,
        "saturation_reference": sat_ref,
        "saturation_bad": sat_bad,
        "saturation_gap": sat_bad - sat_ref,
        "event": event_a,
        "depth_gt": depth_gt,
        "normal_gt": normal_gt,
        "intrinsics": intrinsics,
        "valid_mask": valid_mask,
        "bridge": bridge,
        "pair_labels": [
            f"{left}->{right}" for left, right in zip(batch["ldr_a"], batch["ldr_b"])
        ],
    }


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision != "none"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def configure_phase(model: MultiLdrEventContributionModel, phase: str, training: bool = True) -> None:
    """Make the A/B/C identifiability constraint explicit in module state."""
    if phase not in {"proxy", "contribution", "joint"}:
        raise ValueError(f"Unknown Stage-1 phase: {phase!r}")
    model.requires_grad_(False)
    if phase in {"contribution", "joint"}:
        model.contribution_net.requires_grad_(True)
    if phase in {"proxy", "joint"}:
        model.event_refiner.requires_grad_(True)
    model.eval()
    if training:
        if phase in {"contribution", "joint"}:
            model.contribution_net.train()
        if phase in {"proxy", "joint"}:
            model.event_refiner.train()


def build_phase_optimizer(model: MultiLdrEventContributionModel, phase: str, args):
    lr_proxy = args.lr if args.lr_proxy is None else args.lr_proxy
    lr_contribution = args.lr if args.lr_contribution is None else args.lr_contribution
    if phase == "proxy":
        groups = [{"params": model.event_refiner.parameters(), "lr": lr_proxy}]
    elif phase == "contribution":
        groups = [{"params": model.contribution_net.parameters(), "lr": lr_contribution}]
    elif phase == "joint":
        groups = [
            {"params": model.contribution_net.parameters(), "lr": lr_contribution},
            {
                "params": model.event_refiner.parameters(),
                "lr": lr_contribution * args.joint_refiner_lr_scale,
            },
        ]
    else:
        raise ValueError(f"Unknown Stage-1 phase: {phase!r}")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def process_batch(
    contribution_model,
    rgb_model,
    batch,
    device,
    args,
    phase: str,
):
    prepared = prepare_pair(batch, device, args)
    coarse = frozen_rgb_geometry(rgb_model, prepared["rgb_bad"], prepared["intrinsics"])
    geometry_weight = geometry_emphasis_weight(
        prepared["depth_gt"],
        prepared["normal_gt"],
        prepared["valid_mask"],
        alpha=args.geometry_alpha,
        depth_gradient_weight=args.geometry_depth_gradient_weight,
    )
    with autocast_context(device, args.mixed_precision):
        proxy_override = None
        bypass_contribution_net = False
        if phase == "proxy":
            proxy_override = torch.ones(
                prepared["event"].shape[:2] + prepared["event"].shape[-2:],
                device=prepared["event"].device,
                dtype=prepared["event"].dtype,
            )
            bypass_contribution_net = True
        prediction = contribution_model(
            prepared["event"],
            prepared["rgb_bad"],
            coarse["depth"],
            coarse["normals"],
            coarse["features"],
            contribution_override=proxy_override,
            bypass_contribution_net=bypass_contribution_net,
        )
    supervision_mask = (
        prepared["bridge"].bridge
        if args.supervision_region == "bridge"
        else prepared["bridge"].event_support
    )
    loss_output = stage1_contribution_loss(
        prediction["depth"],
        prediction["normals"],
        prepared["depth_gt"],
        prepared["normal_gt"],
        prepared["valid_mask"],
        supervision_mask,
        geometry_weight,
        prediction["contribution"],
        prepared["event"],
        minimum_bridge_area=args.minimum_bridge_area,
        minimum_saturation_gap=(args.minimum_saturation_gap if args.supervision_region == "bridge" else 0.0),
        saturation_gap=(prepared["saturation_gap"] if args.supervision_region == "bridge" else None),
        normal_weight=args.normal_weight,
        budget_weight=(0.0 if phase == "proxy" else args.budget_weight),
        budget_ratio=args.budget_ratio,
    )
    return prepared, coarse, prediction, loss_output


def metric_values(prepared, prediction, loss_output) -> Dict[str, float]:
    contribution = prediction["contribution"].detach().float()
    event_support = prepared["bridge"].event_support
    supported = contribution[event_support]
    if supported.numel():
        supported_mean = float(supported.mean())
        supported_std = float(supported.std(unbiased=False))
        supported_q10 = float(torch.quantile(supported, 0.10))
        supported_q90 = float(torch.quantile(supported, 0.90))
    else:
        supported_mean = supported_std = supported_q10 = supported_q90 = 0.0
    return {
        "loss": float(loss_output.loss.detach()),
        "depth": float(loss_output.depth_loss.detach()),
        "normal_cos": float(loss_output.normal_loss.detach()),
        "budget": float(loss_output.budget_loss.detach()),
        "contribution_mean": supported_mean,
        "contribution_std": supported_std,
        "contribution_q10": supported_q10,
        "contribution_q90": supported_q90,
        "bridge_area": float(prepared["bridge"].area.mean()),
        "saturation_gap": float(prepared["saturation_gap"].mean()),
        "active_samples": float(loss_output.active_samples.float().sum()),
    }


def run_epoch(
    model,
    rgb_model,
    loader,
    device,
    args,
    *,
    phase: str,
    optimizer=None,
    max_batches: int = 0,
) -> Dict[str, float]:
    training = optimizer is not None
    configure_phase(model, phase, training=training)
    totals: Dict[str, float] = {}
    count = 0
    optimized = 0
    pair_totals: Dict[str, Dict[str, float]] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prepared, _, prediction, loss_output = process_batch(
                model, rgb_model, batch, device, args, phase
            )
            if training and bool(loss_output.active_samples.any()):
                loss_output.loss.backward()
                if args.clip_grad > 0:
                    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
                optimizer.step()
                optimized += 1
        values = metric_values(prepared, prediction, loss_output)
        for sample_index, pair_label in enumerate(prepared["pair_labels"]):
            pair_record = pair_totals.setdefault(
                pair_label,
                {"seen": 0.0, "active": 0.0, "bridge_area_sum": 0.0, "saturation_gap_sum": 0.0},
            )
            pair_record["seen"] += 1.0
            pair_record["active"] += float(loss_output.active_samples[sample_index])
            pair_record["bridge_area_sum"] += float(prepared["bridge"].area[sample_index])
            pair_record["saturation_gap_sum"] += float(prepared["saturation_gap"][sample_index])
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if training and (batch_index % 20 == 0):
            print(
                f"  batch={batch_index:05d} loss={values['loss']:.5f} "
                f"D={values['depth']:.5f} Ncos={values['normal_cos']:.5f} "
                f"budget={values['budget']:.5f} bridge={values['bridge_area']:.4f} "
                f"Cstd={values['contribution_std']:.4f} "
                f"active={int(values['active_samples'])}"
            )
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["batches"] = float(count)
    result["optimized_batches"] = float(optimized)
    result["active_samples_total"] = float(totals.get("active_samples", 0.0))
    result["pair_stats"] = {
        label: {
            "seen": int(record["seen"]),
            "active": int(record["active"]),
            "active_ratio": record["active"] / max(record["seen"], 1.0),
            "bridge_area": record["bridge_area_sum"] / max(record["seen"], 1.0),
            "saturation_gap": record["saturation_gap_sum"] / max(record["seen"], 1.0),
        }
        for label, record in sorted(pair_totals.items())
    }
    return result


def checkpoint_payload(
    model, optimizer, args, cfg, phase: str, phase_epoch: int, train_metrics, val_metrics
):
    return {
        "schema": MultiLdrEventContributionModel.checkpoint_schema,
        "model": model.state_dict(),
        "contribution_net": model.contribution_net.state_dict(),
        "event_refiner": model.event_refiner.state_dict(),
        "architecture": dict(model.architecture),
        "optimizer": optimizer.state_dict(),
        "epoch": int(phase_epoch),
        "training_phase": phase,
        "phase_epoch": int(phase_epoch),
        "proxy_refiner_frozen": phase == "contribution",
        "proxy_contribution_override": 1.0 if phase == "proxy" else None,
        "budget_enabled": phase != "proxy" and args.budget_weight > 0.0,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "training_args": vars(args),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "frozen_rgb_checkpoint": str(Path(args.pretrained).resolve()),
        "ordered_pairs": list(args.pairs),
        "supervision_region": args.supervision_region,
        "legacy_target_export_used": False,
        "token_teacher_used": False,
    }


def save_checkpoint(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_config(args.config, unknown)
    cfg.seed = int(args.seed)
    args.pretrained = args.pretrained or str(getattr(cfg, "pretrained", ROOT / "ckpt" / "model.pt"))
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(f"Frozen RGB checkpoint not found: {args.pretrained}")
    exposures = parse_exposure_sequence(args.exposures)
    pairs = generate_ordered_pairs(exposures, args.pair_mode, args.pair)
    args.exposures = list(exposures)
    args.pairs = [f"{a}->{b}" for a, b in pairs]
    num_workers = int(cfg.num_workers) if args.num_workers is None else int(args.num_workers)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.mixed_precision != "none":
        print("[warning] CUDA is unavailable; disabling mixed precision.")
        args.mixed_precision = "none"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[Stage 1] output={output.resolve()}")
    print(f"[Stage 1] exposures low->high={args.exposures}")
    print(f"[Stage 1] pair mode={args.pair_mode} pairs={args.pairs}")
    print(f"[Stage 1] supervision region={args.supervision_region}")
    print("[Stage 1] token teacher/offline reliability targets: DISABLED")

    train_dataset = make_dataset(cfg, "train", pairs)
    val_dataset = make_dataset(cfg, "test", pairs)
    if len(train_dataset) == 0:
        raise RuntimeError("The Stage-1 training dataset has no valid samples.")
    if len(val_dataset) == 0:
        print("[warning] The held-out dataset is empty; best-checkpoint selection will use train loss.")
    train_loader = make_loader(
        train_dataset, batch_size=args.batch_size, num_workers=num_workers, train=True
    )
    val_loader = make_loader(
        val_dataset, batch_size=args.batch_size, num_workers=num_workers, train=False
    )

    rgb_dtype = torch.float32
    if device.type == "cuda" and args.mixed_precision == "bf16":
        rgb_dtype = torch.bfloat16
    elif device.type == "cuda" and args.mixed_precision == "fp16":
        rgb_dtype = torch.float16
    rgb_model = build_frozen_rgb_model(cfg, args.pretrained, device, rgb_dtype)
    feature_dim = 2 * int(cfg.model.embed_dim)
    model = MultiLdrEventContributionModel(
        num_bins=int(cfg.data.event_resize_bins),
        contribution_channels=args.contribution_channels,
        refiner_channels=args.refiner_channels,
        coarse_feature_dim=feature_dim,
        count_cmax=float(getattr(cfg.model, "event_count_cmax", 3.0)),
        initial_contribution=args.budget_ratio,
        max_log_depth_delta=args.max_log_depth_delta,
        max_normal_delta=args.max_normal_delta,
    ).to(device)
    if args.joint_refiner_lr_scale < 0.0:
        raise ValueError("--joint-refiner-lr-scale must be non-negative")

    phase_epochs = {
        "proxy": int(args.epochs_proxy),
        "contribution": int(args.epochs_contribution),
        "joint": int(args.epochs_joint),
    }
    if args.epochs is not None:
        legacy_target = "contribution" if args.phase == "all" else args.phase
        phase_epochs[legacy_target] = int(args.epochs)
    phases = ["proxy", "contribution"]
    if phase_epochs["joint"] > 0:
        phases.append("joint")
    if args.phase != "all":
        phases = [args.phase]
    phases = [phase for phase in phases if phase_epochs[phase] > 0]
    if not phases:
        raise ValueError("The requested Stage-1 schedule contains zero training epochs")

    def load_phase_input(path: Path, expected_phases) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Required previous-phase checkpoint not found: {path}")
        checkpoint = torch_load(path)
        if checkpoint.get("schema") != MultiLdrEventContributionModel.checkpoint_schema:
            raise ValueError(f"Refusing incompatible Stage-1 checkpoint: {path}")
        checkpoint_phase = checkpoint.get("training_phase")
        if checkpoint_phase not in expected_phases:
            raise ValueError(
                f"Checkpoint {path} is phase {checkpoint_phase!r}; expected one of {expected_phases}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"[Stage 1] loaded {checkpoint_phase} checkpoint: {path.resolve()}")

    if args.resume:
        if args.phase == "all":
            raise ValueError("--resume is only supported with an explicit --phase")
        load_phase_input(Path(args.resume), {args.phase})
        print("[Stage 1] --resume restores model weights only; each phase gets a fresh optimizer.")
    elif args.phase == "contribution":
        load_phase_input(
            Path(args.proxy_checkpoint or output / "checkpoint-proxy-best.pth"), {"proxy"}
        )
    elif args.phase == "joint":
        load_phase_input(
            Path(args.contribution_checkpoint or output / "checkpoint-contribution-best.pth"),
            {"contribution"},
        )

    history = []
    start_time = time.time()
    for phase in phases:
        if args.phase == "all" and phase == "contribution":
            load_phase_input(output / "checkpoint-proxy-best.pth", {"proxy"})
        elif args.phase == "all" and phase == "joint":
            load_phase_input(output / "checkpoint-contribution-best.pth", {"contribution"})

        configure_phase(model, phase, training=True)
        optimizer = build_phase_optimizer(model, phase, args)
        best_validation = math.inf
        print(
            f"[Stage 1-{phase}] epochs={phase_epochs[phase]} "
            f"train ContributionNet={phase in {'contribution', 'joint'}} "
            f"EventRefiner={phase in {'proxy', 'joint'}}"
        )
        for epoch in range(phase_epochs[phase]):
            print(f"[{phase}] Epoch {epoch + 1}/{phase_epochs[phase]}")
            train_metrics = run_epoch(
                model,
                rgb_model,
                train_loader,
                device,
                args,
                phase=phase,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
            )
            if train_metrics["optimized_batches"] <= 0:
                raise RuntimeError(
                    "No exposure pair produced a valid supervision region in this epoch. "
                    "Check LDR/data alignment and the saturation, gradient, area, and gap thresholds."
                )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    rgb_model,
                    val_loader,
                    device,
                    args,
                    phase=phase,
                    max_batches=args.max_val_batches,
                )
            if phase in {"contribution", "joint"} and (
                train_metrics.get("contribution_std", 0.0) < args.collapse_std_threshold
            ):
                print(
                    "[warning] Contribution map is nearly constant: "
                    f"std={train_metrics.get('contribution_std', 0.0):.6f}. "
                    "Inspect q10/q90 and causal drop-high/drop-low evaluation."
                )
            record = {
                "phase": phase,
                "phase_epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
            history.append(record)
            (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            payload = checkpoint_payload(
                model, optimizer, args, cfg, phase, epoch, train_metrics, val_metrics
            )
            validation_loss = (
                float(val_metrics.get("loss", math.inf))
                if val_metrics.get("active_samples_total", 0.0) > 0
                else float(train_metrics.get("loss", math.inf))
            )
            payload["best_validation"] = min(best_validation, validation_loss)
            save_checkpoint(output / "checkpoint-last.pth", payload)
            save_checkpoint(output / f"checkpoint-{phase}-last.pth", payload)
            if validation_loss < best_validation:
                best_validation = validation_loss
                payload["best_validation"] = best_validation
                save_checkpoint(output / f"checkpoint-{phase}-best.pth", payload)
                if phase == "contribution":
                    # This is the identifiable Stage-1 result consumed by Stage 2.
                    save_checkpoint(output / "checkpoint-best.pth", payload)
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    output / f"checkpoint-{phase}-epoch-{epoch + 1:03d}.pth", payload
                )
            print(f"  train={train_metrics}")
            print(f"  validation={val_metrics}")

    elapsed = time.time() - start_time
    print(f"Stage-1 contribution training finished in {elapsed / 60.0:.1f} minutes")
    canonical = output / "checkpoint-best.pth"
    if "contribution" in phases and canonical.is_file():
        print(f"Stage-2 input (frozen proxy + learned contribution): {canonical.resolve()}")
    else:
        print(f"Completed isolated phase; checkpoint: {(output / f'checkpoint-{phases[-1]}-best.pth').resolve()}")


if __name__ == "__main__":
    main()
