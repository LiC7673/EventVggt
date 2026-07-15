"""Train matched C=1 ablations using either oracle E_geo or raw E_full."""
from __future__ import annotations

import os

import finetune_event as fe
import torch

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.contribution_stage1 import build_bridge_masks
from paired_token_reliability.linear_voxel_no_contribution_source_ablation_model import (
    NoContributionSourceAblationModel,
)
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as visual_base
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as dual_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def event_source():
    source = os.environ.get("EVENT_SOURCE_ABLATION", "full").strip().lower()
    if source not in {"full", "geo"}:
        raise ValueError(f"EVENT_SOURCE_ABLATION must be full or geo, got {source!r}")
    return source


def prepare_pair(batch, device, args, phase):
    target, reference, event, bridge = dual_base.prepare_dual_alignment_pair(
        batch, device, args, phase
    )
    source = event_source()
    if source == "geo":
        selected = []
        for index, view in enumerate(target):
            geo = view.get("geometry_event_voxel")
            if not torch.is_tensor(geo):
                raise RuntimeError(f"E_geo ablation missing geometry_event_voxel at view {index}")
            current = dict(view)
            current["event_voxel"] = geo
            current["event_source_preselected"] = True
            current["event_source_label"] = "E_geo (oracle, C=1)"
            selected.append(current)
        target = selected
        event = fe.stack_view_field(target, "event_voxel").float()
        rgb_bad = fe.stack_view_field(target, "img").float().clamp(0, 1)
        rgb_ref = fe.stack_view_field(reference, "img").float().clamp(0, 1)
        bridge = build_bridge_masks(
            rgb_ref, rgb_bad, event,
            require_reference_gradient=args.bridge_require_reference_gradient,
            event_support_dilate_kernel=args.bridge_event_dilate_kernel,
            saturation_mode=args.bridge_saturation_mode,
        )
    else:
        for view in target:
            view["event_source_label"] = "E_full (C=1)"
    return target, reference, event, bridge


class NoContributionObjective:
    """Geometry + HDR alignment only; no C target, budget, or utility."""

    def __init__(self, base):
        self.base = base
        self.calls = 0

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        hdr_error = torch.stack(
            [item["hdr_token_alignment_error"] for item in output.ress], dim=1
        ).float().mean()
        result.loss = result.loss + hdr_error
        result.details["event_feature_alignment"] = hdr_error * 0.0
        result.details["hdr_token_alignment"] = hdr_error
        result.details["budget"] = result.details["budget"] * 0.0
        self.calls += int(torch.is_grad_enabled())
        if torch.is_grad_enabled() and self.calls % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[no-C-{event_source()}@{self.calls:05d}] C=1 "
                f"HDRalign={float(hdr_error.detach()):.5f}",
                flush=True,
            )
        return result


def build_model(cfg, args, device):
    m = cfg.model
    source = event_source()
    configured = str(getattr(m, "event_source_ablation", source)).strip().lower()
    if configured != source:
        raise RuntimeError(f"environment source={source}, config source={configured}")
    model = NoContributionSourceAblationModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        hdr_token_bottleneck=int(getattr(m, "hdr_token_bottleneck", 256)),
        hdr_warmup_steps=int(getattr(m, "hdr_warmup_steps", 1000)),
        normal_refine_iterations=int(getattr(m, "normal_refine_iterations", 3)),
        normal_refine_step_limit=float(getattr(m, "normal_refine_step_limit", .05)),
        point_update_scale=float(getattr(m, "point_update_scale", .10)),
        event_source=source,
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    message = model.load_state_dict(
        strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))), strict=False
    )
    required = [key for key in message.missing_keys if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(f"[event-source ablation] source=E_{source} C=1 ContributionNet=absent", flush=True)
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    if phase != "adapter":
        raise ValueError(f"event-source ablation is one-stage, got {phase}")
    model.requires_grad_(False)
    model.depth_log_scale.requires_grad_(True)
    model.event_encoder.requires_grad_(True)
    model.event_normal_decoder.requires_grad_(True)
    model.event_token_projection.requires_grad_(True)
    model.ldr_event_hdr_aligner.requires_grad_(True)
    model.normal_depth_refiner.requires_grad_(True)
    model.point_refiner.requires_grad_(True)
    model.train()
    model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(
        f"[trainable no-C {model.event_source}] event_encoder+event_normal+"
        "event_token_projection+hdr_aligner+depth_refiner+point_refiner+depth_scale",
        flush=True,
    )


def optimizer_for(model, _phase, args):
    scale_id = id(model.depth_log_scale)
    encoder_ids = {id(parameter) for parameter in model.event_encoder.parameters()}
    fast_modules = [
        model.event_normal_decoder, model.event_token_projection,
        model.ldr_event_hdr_aligner, model.normal_depth_refiner, model.point_refiner,
    ]
    fast_ids = {id(parameter) for module in fast_modules for parameter in module.parameters()}
    regular, encoder, fast, scale = [], [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) == scale_id:
            scale.append(parameter)
        elif id(parameter) in encoder_ids:
            encoder.append(parameter)
        elif id(parameter) in fast_ids:
            fast.append(parameter)
        else:
            regular.append(parameter)
    groups = []
    if regular: groups.append({"params": regular, "lr": args.lr})
    if encoder: groups.append({"params": encoder, "lr": 2.0 * args.lr})
    if fast: groups.append({"params": fast, "lr": 5.0 * args.lr})
    if scale: groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def criterion_for(args, _phase):
    base = UnifiedGeometryContributionLoss(
        depth_weight=1.0, normal_weight=args.normal_weight,
        point_weight=args.point_weight, bridge_beta=args.bridge_beta,
        budget_weight=0.0, pair_weight=0.0, update_weight=args.update_weight,
        decomposition_weight=0.0, geometry_rank_weight=0.0,
        event_normal_weight=args.event_normal_weight,
        depth_event_normal_weight=args.depth_event_normal_weight,
        depth_gradient_weight=args.depth_gradient_weight,
        depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight,
        grid_patch_size=args.grid_patch_size, points_loss_type="l1",
    )
    return NoContributionObjective(base)


def main(argv=None):
    pipeline.prepare_pair = prepare_pair
    pipeline.build_alternating_phase_schedule = dual_base.one_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = visual_base.save_visual
    pipeline.capture_runtime_state = dual_base.capture_runtime_state
    pipeline.restore_runtime_state = dual_base.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = NoContributionSourceAblationModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
