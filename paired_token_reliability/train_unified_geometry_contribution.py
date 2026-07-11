"""Train the final contribution-guided geometry model in isolated A/B/C phases."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf

import finetune_event as fe
from paired_token_reliability.common import move_views_to_device, strip_module_prefix, torch_load
from paired_token_reliability.contribution_dataset import generate_ordered_pairs, parse_exposure_sequence
from paired_token_reliability.contribution_stage1 import build_bridge_masks, orient_exposure_pair
from paired_token_reliability.contribution_dataset import MultiLdrContributionDataset
from paired_token_reliability.train_contribution_stage1 import make_loader
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss
from paired_token_reliability.unified_model import (
    UnifiedGeometryContributionModel,
    contribution_override,
)
from stage2_geometry_adapter.model import StreamVGGT


ROOT = Path(__file__).resolve().parents[1]


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device(
            args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        return False, 0, 1, device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def reduce_metrics(totals, batches, device, distributed):
    if not distributed:
        return {key: value / max(batches, 1) for key, value in totals.items()}
    keys = sorted(totals)
    packed = torch.tensor(
        [totals[key] for key in keys] + [float(batches)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(packed)
    count = max(float(packed[-1]), 1.0)
    return {key: float(packed[index]) / count for index, key in enumerate(keys)}


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(ROOT / "config" / "finetune_event.yaml"))
    value.add_argument("--pretrained", default=str(ROOT / "ckpt" / "model.pt"))
    value.add_argument("--output", default="abl_event_exp/unified_geometry_contribution")
    value.add_argument("--exposures", default="0,1,2,5,10")
    value.add_argument("--pair-mode", choices=("all", "adjacent", "anchor"), default="anchor")
    value.add_argument(
        "--epochs-a", type=int, default=2,
        help="Initial consecutive Phase-A warm-up epochs.",
    )
    value.add_argument(
        "--epochs-b", type=int, default=10,
        help="Number of alternating B->A epoch pairs after A warm-up.",
    )
    value.add_argument("--epochs-c", type=int, default=0)
    value.add_argument("--lr", type=float, default=1.0e-4)
    value.add_argument("--weight-decay", type=float, default=1.0e-4)
    value.add_argument("--batch-size", type=int, default=1)
    value.add_argument("--num-workers", type=int, default=4)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--clip-grad", type=float, default=1.0)
    value.add_argument(
        "--require-full-event-phase-b", action="store_true",
        help="Require data.event_source_mode=decomposition_full for Phase B/C.",
    )
    value.add_argument("--bridge-beta", type=float, default=2.0)
    value.add_argument("--bridge-event-dilate-kernel", type=int, default=3)
    value.add_argument(
        "--bridge-saturation-mode",
        choices=("any_channel", "all_channels", "luminance"),
        default="all_channels",
    )
    value.add_argument(
        "--bridge-require-reference-gradient", action="store_true",
        help="Restore the legacy strict Bridge definition.",
    )
    value.add_argument("--normal-weight", type=float, default=0.75)
    value.add_argument("--depth-gradient-weight", type=float, default=0.5)
    value.add_argument("--depth-curvature-weight", type=float, default=0.1)
    value.add_argument("--patch-grid-weight", type=float, default=0.25)
    value.add_argument("--grid-patch-size", type=int, default=14)
    value.add_argument("--event-normal-weight", type=float, default=0.5)
    value.add_argument("--depth-event-normal-weight", type=float, default=0.5)
    value.add_argument("--point-weight", type=float, default=1.0)
    value.add_argument("--budget-weight", type=float, default=0.05)
    value.add_argument("--pair-weight", type=float, default=0.2)
    value.add_argument("--update-weight", type=float, default=0.01)
    value.add_argument("--decomposition-weight", type=float, default=0.2)
    value.add_argument("--geometry-rank-weight", type=float, default=0.10)
    value.add_argument("--geometry-rank-margin", type=float, default=0.05)
    value.add_argument("--geometry-rank-threshold", type=float, default=0.10)
    value.add_argument("--no-pair-consistency", action="store_true")
    value.add_argument("--no-budget", action="store_true")
    value.add_argument("--no-geometry-prior", action="store_true")
    value.add_argument("--train-geometry-heads-a", action="store_true")
    value.add_argument("--max-train-batches", type=int, default=0)
    value.add_argument("--max-val-batches", type=int, default=50)
    value.add_argument("--visualize-every-batches", type=int, default=40)
    value.add_argument("--visualize-val-every-batches", type=int, default=20)
    return value


def normalize_dotlist_overrides(overrides):
    """Translate Hydra-style +/++ prefixes for plain OmegaConf dotlists.

    This trainer parses its own CLI rather than running under Hydra. Without
    normalization, ``+data.foo=value`` creates a literal ``+data`` root node
    and leaves ``data.foo`` unchanged.
    """
    normalized = []
    for item in overrides:
        if item.startswith("++"):
            item = item[2:]
        elif item.startswith("+"):
            item = item[1:]
        normalized.append(item)
    return normalized


def load_cfg(path, overrides):
    cfg = OmegaConf.load(path)
    invalid = [item for item in overrides if "=" not in item]
    if invalid:
        raise ValueError(f"Expected key=value overrides, got {invalid}")
    if overrides:
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.from_dotlist(normalize_dotlist_overrides(overrides)),
        )
    OmegaConf.set_struct(cfg, False)
    return cfg


def make_unified_dataset(cfg, split, pairs):
    """Build disjoint scene-level train/test splits when configured.

    This intentionally leaves the legacy Stage-1 dataset builder unchanged.
    """
    data = cfg.data
    is_train = split == "train"
    train_initial = int(getattr(data, "train_initial_scene_idx", 0))
    train_count = int(getattr(data, "train_scene_count", 12))
    if is_train:
        initial_scene_idx = train_initial
        active_scene_count = train_count
    else:
        initial_scene_idx = int(
            getattr(data, "test_initial_scene_idx", train_initial + train_count)
        )
        active_scene_count = int(getattr(data, "test_scene_count", 4))
    frame_count = int(getattr(
        data,
        "train_holdout_frame_count" if is_train else "heldout_test_frame_count",
        0 if is_train else 120,
    ))
    return MultiLdrContributionDataset(
        root=str(data.root),
        split=split,
        num_views=int(data.num_views),
        resolution=tuple(data.resolution),
        fps=int(data.fps),
        seed=int(cfg.seed),
        ordered_pairs=pairs,
        scene_names=list(data.scene_names) if getattr(data, "scene_names", None) else None,
        initial_scene_idx=initial_scene_idx,
        active_scene_count=active_scene_count,
        test_frame_count=frame_count,
        min_train_start_id=int(getattr(data, "min_train_start_id", 0)),
        event_y_flip=getattr(data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(data, "event_spatial_transform", "auto"),
        event_resize_method=str(getattr(data, "event_resize_method", "voxel_antialias")),
        event_resize_bins=int(getattr(data, "event_resize_bins", 10)),
        event_source_mode=str(getattr(data, "event_source_mode", "current")),
        decomposition_supervision=bool(getattr(data, "decomposition_supervision", False)),
        decomposition_event_root=str(getattr(data, "decomposition_event_root", "events_additive")),
        decomposition_geo_branch=str(getattr(data, "decomposition_geo_branch", "geometry_motion")),
        decomposition_full_branch=str(getattr(data, "decomposition_full_branch", "full")),
    )


def build_model(cfg, args, device):
    model = UnifiedGeometryContributionModel(
        img_size=int(cfg.model.img_size),
        patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        event_hidden_dim=int(getattr(cfg.model, "adapter_event_hidden_dim", 48)),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        event_num_bins=int(cfg.data.event_resize_bins),
        event_count_cmax=float(getattr(cfg.model, "event_count_cmax", 3.0)),
        event_pyramid_channels=int(getattr(cfg.model, "adapter_event_pyramid_channels", 64)),
        adapter_hidden_channels=int(getattr(cfg.model, "adapter_hidden_channels", 128)),
        contribution_channels=int(getattr(cfg.model, "contribution_channels", 32)),
        # Phase A uses an explicit C=1 override. Keep the learnable sigmoid
        # away from saturation so Phase B still receives useful gradients.
        contribution_initial_value=0.95,
        contribution_use_geometry_prior=not args.no_geometry_prior,
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    message = model.load_state_dict(state, strict=False)
    required_missing = [
        key for key in message.missing_keys
        if key.startswith(("aggregator.", "camera_head."))
    ]
    if required_missing:
        raise RuntimeError(f"Base checkpoint misses required VGGT weights: {required_missing[:10]}")
    print(
        f"[base checkpoint] missing(new modules)={len(message.missing_keys)} "
        f"unused={len(message.unexpected_keys)}",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.event_encoder.requires_grad_(True)
        model.depth_head.geometry_adapters.requires_grad_(True)
        model.point_head.geometry_adapters.requires_grad_(True)
        if train_heads_a:
            model.depth_head.requires_grad_(True)
            model.point_head.requires_grad_(True)
    elif phase == "contribution":
        model.contribution_net.requires_grad_(True)
        model.event_normal_decoder.requires_grad_(True)
    elif phase == "joint":
        model.contribution_net.requires_grad_(True)
        model.event_encoder.requires_grad_(True)
        model.event_normal_decoder.requires_grad_(True)
        model.depth_head.geometry_adapters.requires_grad_(True)
        model.point_head.geometry_adapters.requires_grad_(True)
        model.depth_head.requires_grad_(True)
        model.point_head.requires_grad_(True)
    else:
        raise ValueError(phase)
    model.camera_head.requires_grad_(False)
    model.aggregator.requires_grad_(False)
    model.train()
    model.camera_head.eval()
    model.aggregator.eval()
    if phase == "adapter":
        model.contribution_net.eval()
    if phase == "contribution":
        model.event_encoder.eval()
        model.depth_head.eval()
        model.point_head.eval()


def optimizer_for(model, phase, args):
    if phase != "joint":
        parameters = [item for item in model.parameters() if item.requires_grad]
        return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    contribution_ids = {id(item) for item in model.contribution_net.parameters()}
    adapter_ids = {
        id(item)
        for module in (
            model.event_encoder,
            model.event_normal_decoder,
            model.depth_head.geometry_adapters,
            model.point_head.geometry_adapters,
        )
        for item in module.parameters()
    }
    contribution, adapters, heads = [], [], []
    for item in model.parameters():
        if not item.requires_grad:
            continue
        if id(item) in contribution_ids:
            contribution.append(item)
        elif id(item) in adapter_ids:
            adapters.append(item)
        else:
            heads.append(item)
    return torch.optim.AdamW(
        [
            {"params": contribution, "lr": args.lr},
            {"params": adapters, "lr": 0.1 * args.lr},
            {"params": heads, "lr": 0.03 * args.lr},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )


def _select_views(views_a, views_b, choose_a):
    selected = []
    for left, right in zip(views_a, views_b):
        current = {}
        for key in left:
            lv, rv = left[key], right.get(key, left[key])
            if torch.is_tensor(lv) and torch.is_tensor(rv) and lv.ndim > 0 and lv.shape[0] == choose_a.shape[0]:
                selector = choose_a.view(choose_a.shape[0], *([1] * (lv.ndim - 1)))
                current[key] = torch.where(selector, lv, rv)
            else:
                current[key] = lv
        selected.append(current)
    return selected


def use_phase_event_source(views, phase):
    """Use oracle geometry events only in Phase A; keep full events in B/C."""
    if phase != "adapter":
        return views
    selected = []
    for view_index, view in enumerate(views):
        if "geometry_event_voxel" not in view:
            instance = view.get("instance", "unknown")
            raise RuntimeError(
                "Phase A requires geometry_event_voxel, but it is missing for "
                f"view={view_index}, instance={instance!r}. Enable decomposition "
                "supervision and provide the Blender geometry-event branch."
            )
        geometry_event = view["geometry_event_voxel"]
        full_event = view.get("event_voxel")
        if full_event is None or geometry_event.shape != full_event.shape:
            raise RuntimeError(
                "Phase-A geometry/full event shape mismatch at "
                f"view={view_index}: geometry={tuple(geometry_event.shape)}, "
                f"full={None if full_event is None else tuple(full_event.shape)}"
            )
        current = dict(view)
        current["event_voxel"] = geometry_event
        selected.append(current)
    return selected


def prepare_pair(batch, device, args, phase):
    views_a = fe.maybe_denormalize_views(move_views_to_device(batch["views_a"], device))
    views_b = fe.maybe_denormalize_views(move_views_to_device(batch["views_b"], device))
    rgb_a = fe.stack_view_field(views_a, "img").float().clamp(0, 1)
    rgb_b = fe.stack_view_field(views_b, "img").float().clamp(0, 1)
    rgb_ref, rgb_bad, ref_is_a, _, _ = orient_exposure_pair(
        rgb_a, rgb_b, saturation_mode=args.bridge_saturation_mode
    )
    target_views = _select_views(views_a, views_b, ~ref_is_a)
    reference_views = _select_views(views_a, views_b, ref_is_a)
    target_views = use_phase_event_source(target_views, phase)
    reference_views = use_phase_event_source(reference_views, phase)
    event = fe.stack_view_field(target_views, "event_voxel").float()
    bridge = build_bridge_masks(
        rgb_ref,
        rgb_bad,
        event,
        require_reference_gradient=args.bridge_require_reference_gradient,
        event_support_dilate_kernel=args.bridge_event_dilate_kernel,
        saturation_mode=args.bridge_saturation_mode,
    )
    return target_views, reference_views, event, bridge


def rho_schedule(epoch, epochs):
    progress = (epoch + 0.5) / max(epochs, 1)
    if progress < 1.0 / 3.0:
        return 1.0
    if progress < 2.0 / 3.0:
        return 0.7
    return 0.5


def build_alternating_phase_schedule(warmup_a, alternating_cycles, joint_epochs=0):
    """A warm-up, then switch phase after every epoch: A,A,B,A,B,A,..."""
    warmup_a = int(warmup_a)
    alternating_cycles = int(alternating_cycles)
    joint_epochs = int(joint_epochs)
    if warmup_a != 2:
        raise ValueError(
            "Alternating training requires exactly two initial Phase-A epochs; "
            f"got --epochs-a={warmup_a}."
        )
    if alternating_cycles < 0 or joint_epochs < 0:
        raise ValueError("Epoch counts must be non-negative.")
    schedule = ["adapter"] * warmup_a
    for _ in range(alternating_cycles):
        schedule.extend(("contribution", "adapter"))
    schedule.extend(["joint"] * joint_epochs)
    return schedule


def criterion_for(args, phase):
    return UnifiedGeometryContributionLoss(
        depth_weight=1.0,
        normal_weight=args.normal_weight,
        point_weight=args.point_weight,
        bridge_beta=(0.0 if phase == "adapter" else args.bridge_beta),
        budget_weight=(0.0 if phase == "adapter" or args.no_budget else args.budget_weight),
        pair_weight=(0.0 if phase == "adapter" or args.no_pair_consistency else args.pair_weight),
        update_weight=(0.0 if phase == "adapter" else args.update_weight),
        decomposition_weight=(0.0 if phase == "adapter" else args.decomposition_weight),
        geometry_rank_weight=(0.0 if phase == "adapter" else args.geometry_rank_weight),
        geometry_rank_margin=args.geometry_rank_margin,
        geometry_rank_threshold=args.geometry_rank_threshold,
        event_normal_weight=(0.0 if phase == "adapter" else args.event_normal_weight),
        depth_event_normal_weight=(
            0.0 if phase == "adapter" else args.depth_event_normal_weight
        ),
        depth_gradient_weight=args.depth_gradient_weight,
        depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight,
        grid_patch_size=args.grid_patch_size,
        points_loss_type="l1",
    )


def visual_map(value, fixed=False):
    array = value.detach().float().cpu().numpy()
    if fixed:
        return np.clip(array, 0, 1)
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros_like(array)
    lo, hi = np.percentile(array[valid], (2, 98))
    return np.clip((array - lo) / max(float(hi - lo), 1.0e-8), 0, 1)


def visual_normal(normal, valid):
    value = (normal.detach().float().cpu().numpy() + 1.0) * 0.5
    mask = valid.detach().bool().cpu().numpy()
    value = np.clip(value, 0.0, 1.0)
    value[~mask] = 0.0
    return value


def visual_signed(value):
    array = value.detach().float().cpu().numpy()
    finite = np.isfinite(array)
    if not finite.any():
        return np.full_like(array, 0.5)
    scale = np.percentile(np.abs(array[finite]), 98)
    return np.clip(0.5 + 0.5 * array / max(float(scale), 1.0e-8), 0.0, 1.0)


@torch.no_grad()
def save_visual(
    output_root, phase, epoch, batch_index, views, reference_views,
    event, bridge, output, aux,
):
    rgb = views[0]["img"][0].float().permute(1, 2, 0).cpu().numpy().clip(0, 1)
    reference_rgb = (
        reference_views[0]["img"][0].float().permute(1, 2, 0).cpu().numpy().clip(0, 1)
    )
    contribution = aux["contribution_spatial"][0, 0]
    pred = aux["depth_pred_live"][0, 0]
    coarse = torch.stack(
        [item["depth_coarse"] for item in output.ress], dim=1
    ).squeeze(-1)[0, 0]
    gt = fe.stack_view_field(views, "depthmap")[0, 0]
    valid = aux["valid_live"][0, 0]
    normal_valid = aux["normal_valid_live"][0, 0]
    predicted_normal = visual_normal(aux["normal_pred_live"][0, 0], normal_valid)
    target_normal = visual_normal(aux["normal_gt_live"][0, 0], normal_valid)
    event_normal = (
        visual_normal(
            aux["event_normal_live"][0, 0],
            aux["event_normal_valid_live"][0, 0],
        )
        if aux.get("event_normal_live") is not None else None
    )
    geometry = aux["geometry_score"][0, 0]
    event_channels = event[0, 0]
    event_bins = event_channels.shape[0] // 2
    diagnostic_bins = tuple(
        sorted({0, max(event_bins // 2, 0), max(event_bins - 1, 0)})
    )
    temporal_panels = tuple(
        (
            visual_map(
                event_channels[bin_index].abs()
                + event_channels[event_bins + bin_index].abs()
            ),
            f"event bin {bin_index:02d} (time ascending)",
            "gray",
        )
        for bin_index in diagnostic_bins
    )
    event_source_label = "E_geo" if phase.startswith("adapter") else "E_full"
    panels = (
        (rgb, "RGB", None),
        (reference_rgb, "reference RGB", None),
        (
            visual_map(event[0, 0].abs().sum(0)),
            f"event ({event_source_label})",
            "gray",
        ),
        *temporal_panels,
        (visual_map(contribution, True), "contribution", "magma"),
        (visual_map(bridge.bridge[0, 0].float(), True), "bridge", "gray"),
        (visual_map(geometry), "geometry score", "magma"),
        (visual_map(coarse * valid), "coarse depth", "viridis"),
        (visual_map(pred * valid), "pred depth", "viridis"),
        (visual_signed((pred - coarse) * valid), "depth update", "coolwarm"),
        (visual_map(gt * valid), "GT depth", "viridis"),
        (predicted_normal, "pred normal", None),
        (target_normal, "GT normal", None),
        *(((event_normal, "event normal", None),) if event_normal is not None else ()),
        *(
            ((visual_map(aux["decomposition_target"][0, 0], True), "decomp target", "magma"),)
            if aux.get("decomposition_target") is not None
            else ()
        ),
    )
    columns = 5
    rows = int(math.ceil(len(panels) / columns))
    # Grow the canvas with the panel count. Each panel remains at least
    # 5 inches wide/high instead of being squeezed into a fixed-size figure.
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
    instance = views[0].get("instance", "unknown")
    if isinstance(instance, (list, tuple)):
        instance = instance[0]
    time_range = views[0].get("event_time_range")
    if torch.is_tensor(time_range):
        time_range = time_range[0].detach().float().cpu().tolist()
    spatial_transform = views[0].get("event_spatial_transform", "unknown")
    if isinstance(spatial_transform, (list, tuple)):
        spatial_transform = spatial_transform[0]
    figure.suptitle(
        f"instance={instance} event_time={time_range} transform={spatial_transform}",
        fontsize=13,
    )
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run_epoch(
    model, raw_model, loader, optimizer, criterion, device, args, phase, epoch, epochs,
    training=True, distributed=False, rank=0,
):
    configure_phase(raw_model, phase, args.train_geometry_heads_a)
    model.train(training)
    raw_model.aggregator.eval()
    raw_model.camera_head.eval()
    if phase == "contribution":
        raw_model.event_encoder.eval()
        raw_model.depth_head.eval()
        raw_model.point_head.eval()
    if not training:
        raw_model.eval()
    totals, batches = {}, 0
    limit = args.max_train_batches if training else args.max_val_batches
    for batch_index, batch in enumerate(loader):
        if limit > 0 and batch_index >= limit:
            break
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            target_views, reference_views, event, bridge = prepare_pair(
                batch, device, args, phase
            )
            # Phase A consumes E_geo exactly and bypasses ContributionNet.
            # Phase B/C consume the dataset's full event and predict C.
            override = contribution_override(event, "full") if phase == "adapter" else None
            contribution_reference = None
            if phase != "adapter" and not args.no_pair_consistency:
                # The paired exposure is a stop-gradient anchor. Besides being
                # stable, this avoids a second DDP reducer pass and saves its
                # activation memory.
                with torch.no_grad():
                    contribution_reference = raw_model.predict_contribution(reference_views)
            output = model(
                target_views,
                contribution_override=override,
                decode_event_normal=(phase != "adapter"),
            )
            rho = 1.0 if phase == "adapter" else rho_schedule(epoch, epochs)
            result = criterion(
                output,
                target_views,
                bridge.bridge,
                event,
                rho=rho,
                contribution_reference=contribution_reference,
            )
            if training:
                result.loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [item for item in model.parameters() if item.requires_grad], args.clip_grad
                    )
                optimizer.step()
        values = {key: float(value.detach()) for key, value in result.details.items()}
        values["loss"] = float(result.loss.detach())
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
        train_vis = training and args.visualize_every_batches > 0 and (
            batch_index == 0 or (batch_index + 1) % args.visualize_every_batches == 0
        )
        val_vis = (not training) and args.visualize_val_every_batches > 0 and (
            batch_index == 0 or (batch_index + 1) % args.visualize_val_every_batches == 0
        )
        if rank == 0 and (train_vis or val_vis):
            split_phase = f"{phase}_{'train' if training else 'val'}"
            save_visual(
                args.output, split_phase, epoch, batch_index,
                target_views, reference_views, event, bridge, output, result.aux,
            )
        if rank == 0 and training and batch_index % 20 == 0:
            print(
                f"[{phase}] batch={batch_index:05d} loss={values['loss']:.5f} "
                f"D={values['depth']:.5f} N={values['normal']:.5f} P={values['point']:.5f} "
                f"DG={values['depth_gradient']:.5f} DC={values['depth_curvature']:.5f} "
                f"Grid={values['patch_grid']:.5f} "
                f"EN={values['event_normal']:.5f} DN={values['depth_event_normal']:.5f} "
                f"budget={values['budget']:.5f} pair={values['pair']:.5f} "
                f"decomp={values['decomposition']:.5f} "
                f"rank={values['geometry_rank']:.5f} "
                f"Cmean={values['contribution_mean']:.4f} Cstd={values['contribution_std']:.4f}",
                flush=True,
            )
    return reduce_metrics(totals, batches, device, distributed)


def save_checkpoint(path, model, optimizer, cfg, args, phase, epoch, metrics):
    payload = {
        "schema": UnifiedGeometryContributionModel.checkpoint_schema,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "phase": phase,
        "epoch": epoch,
        "metrics": metrics,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "training_args": vars(args),
        "inference_requires": ["rgb", "event_voxel"],
        "bridge_used_at_inference": False,
        "reference_used_at_inference": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_unified_phase_checkpoint(model, path):
    checkpoint = torch_load(path)
    expected = UnifiedGeometryContributionModel.checkpoint_schema
    if checkpoint.get("schema") != expected:
        raise RuntimeError(
            f"Checkpoint {path} uses schema {checkpoint.get('schema')!r}, expected "
            f"{expected!r}. The pre-resize adapter graph requires a fresh Phase-A run."
        )
    model.load_state_dict(checkpoint["model"], strict=True)


def main(argv=None):
    args, overrides = parser().parse_known_args(argv)
    if not getattr(StreamVGGT, "supports_unified_training", False):
        raise RuntimeError(
            "The loaded stage2_geometry_adapter/model.py is an old Stage-2-only version "
            "that requires a Stage-1 checkpoint. Update that file to the unified version; "
            "this trainer intentionally initializes ContributionNet from scratch."
        )
    cfg = load_cfg(args.config, overrides)
    # Phase A is defined as E_geo adapter pretraining.  Configure the default
    # decomposition source here as well as in the launchers so a direct Python
    # invocation cannot silently fall back to a non-geometry event stream.
    if args.epochs_a > 0:
        cfg.data.decomposition_supervision = True
        if not getattr(cfg.data, "decomposition_event_root", None):
            cfg.data.decomposition_event_root = "events_additive"
        if not getattr(cfg.data, "decomposition_geo_branch", None):
            cfg.data.decomposition_geo_branch = "geometry_motion"
        if not getattr(cfg.data, "decomposition_full_branch", None):
            cfg.data.decomposition_full_branch = "full"
    if (
        args.require_full_event_phase_b
        and (args.epochs_b > 0 or args.epochs_c > 0)
        and str(getattr(cfg.data, "event_source_mode", "current"))
        != "decomposition_full"
    ):
        raise RuntimeError(
            "Phase B/C require E_full, but data.event_source_mode is "
            f"{getattr(cfg.data, 'event_source_mode', 'current')!r}; expected "
            "'decomposition_full'."
        )
    distributed, rank, world_size, device = init_distributed(args)
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(args.pretrained)
    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    exposures = parse_exposure_sequence(args.exposures)
    pairs = generate_ordered_pairs(exposures, args.pair_mode)
    train_dataset = make_unified_dataset(cfg, "train", pairs)
    val_dataset = make_unified_dataset(cfg, "test", pairs)
    if rank == 0:
        print(f"DDP world_size={world_size}, per_gpu_batch={args.batch_size}", flush=True)
        print(f"train scenes={train_dataset.dataset.get_active_scenes()}", flush=True)
        print(f"test scenes={val_dataset.dataset.get_active_scenes()}", flush=True)
        print(
            "event source="
            f"{getattr(cfg.data, 'event_source_mode', 'current')} "
            "decomposition="
            f"{bool(getattr(cfg.data, 'decomposition_supervision', False))}",
            flush=True,
        )
        print(
            "phase event inputs: A=geometry_event_voxel, "
            "B/C=event_voxel"
            + (
                " (validated decomposition_full)"
                if args.require_full_event_phase_b else ""
            ),
            flush=True,
        )
        base_dataset = train_dataset.dataset
        if base_dataset.active_scenes:
            debug_scene = base_dataset.active_scenes[0]
            debug_meta = base_dataset.active_scene_data[debug_scene]
            full_timing = debug_meta.get("event_time_info") or {}
            geo_timing = debug_meta.get("geo_event_time_info") or {}
            print(
                "event temporal layout: channels=[pos bins 0..B-1, neg bins 0..B-1], "
                "bins=time-ascending; "
                f"scene={debug_scene!r} "
                f"full(origin={full_timing.get('origin')},dt={full_timing.get('dt')}) "
                f"geo(origin={geo_timing.get('origin')},dt={geo_timing.get('dt')})",
                flush=True,
            )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if distributed else None
    train_loader = make_loader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        train=True, sampler=train_sampler,
    )
    val_loader = make_loader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        train=False, sampler=val_sampler,
    )
    model = build_model(cfg, args, device)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    history = []
    schedule = build_alternating_phase_schedule(
        args.epochs_a, args.epochs_b, args.epochs_c
    )
    phase_totals = {
        phase: schedule.count(phase) for phase in ("adapter", "contribution", "joint")
    }
    phase_epochs = {phase: 0 for phase in phase_totals}
    best = {phase: math.inf for phase in phase_totals}
    optimizers = {}
    first_contribution = True
    first_joint = True
    if rank == 0:
        print(
            "epoch schedule=" + " -> ".join(
                {"adapter": "A", "contribution": "B", "joint": "C"}[item]
                for item in schedule
            ),
            flush=True,
        )
    for global_epoch, phase in enumerate(schedule):
        if phase == "contribution" and first_contribution:
            # Begin B from the best of the two initial A warm-up epochs. Later
            # A/B switches retain the current unified model and optimizer state.
            load_unified_phase_checkpoint(model, output / "checkpoint-adapter-best.pth")
            # The loaded best A weights may come from the first warm-up epoch;
            # discard momentum accumulated for a different A parameter state.
            optimizers.pop("adapter", None)
            first_contribution = False
        elif phase == "joint" and first_joint:
            load_unified_phase_checkpoint(model, output / "checkpoint-contribution-best.pth")
            first_joint = False
        configure_phase(model, phase, args.train_geometry_heads_a)
        if phase not in optimizers:
            optimizers[phase] = optimizer_for(model, phase, args)
        optimizer = optimizers[phase]
        train_model = (
            DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if distributed else model
        )
        criterion = criterion_for(args, phase)
        epoch = phase_epochs[phase]
        epochs = phase_totals[phase]
        if train_sampler is not None:
            train_sampler.set_epoch(global_epoch)
        start = time.time()
        train_metrics = run_epoch(
            train_model, model, train_loader, optimizer, criterion, device, args,
            phase, epoch, epochs, True, distributed, rank,
        )
        val_metrics = run_epoch(
            train_model, model, val_loader, optimizer, criterion, device, args,
            phase, epoch, epochs, False, distributed, rank,
        )
        record = {
            "phase": phase,
            "epoch": epoch,
            "global_epoch": global_epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        if rank == 0:
            history.append(record)
            (output / "metrics.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
        score = val_metrics.get("loss", train_metrics.get("loss", math.inf))
        if rank == 0:
            save_checkpoint(
                output / f"checkpoint-{phase}-last.pth",
                model, optimizer, cfg, args, phase, epoch, record,
            )
        if score < best[phase]:
            best[phase] = score
            if rank == 0:
                save_checkpoint(
                    output / f"checkpoint-{phase}-best.pth",
                    model, optimizer, cfg, args, phase, epoch, record,
                )
                if phase in {"contribution", "joint"}:
                    save_checkpoint(
                        output / "checkpoint-best.pth",
                        model, optimizer, cfg, args, phase, epoch, record,
                    )
        phase_epochs[phase] += 1
        if rank == 0:
            print(
                f"[{phase}] epoch={epoch+1}/{epochs} global={global_epoch+1}/{len(schedule)} "
                f"train={train_metrics} val={val_metrics} "
                f"time={(time.time()-start)/60:.1f}m"
            )
        del train_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if distributed:
            dist.barrier()
    if rank == 0:
        print(f"Unified model ready: {(output / 'checkpoint-best.pth').resolve()}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
