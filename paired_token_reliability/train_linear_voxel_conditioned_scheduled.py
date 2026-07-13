"""Train coarse-conditioned event detail with iteration schedules."""
from __future__ import annotations

import torch
import finetune_event as fe

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_conditioned_scheduled_model import (
    ConditionedScheduledLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_scheduled_diagnostic as schedule_base
from paired_token_reliability import train_linear_voxel_detail_residual as detail_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def build_model(cfg, args, device):
    model = ConditionedScheduledLinearVoxelModel(
        img_size=int(cfg.model.img_size), patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(cfg.model, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(cfg.model, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(cfg.model, "depth_update_scale", 1.0)),
        event_decay_tau=float(getattr(cfg.model, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(cfg.model, "depth_log_scale_limit", 2.0)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    message = model.load_state_dict(
        strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))), strict=False
    )
    required = [k for k in message.missing_keys if k.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(
        f"[conditioned scheduled base] scale={float(model.metric_depth_scale):.4f} "
        f"new={len(message.missing_keys)} unused={len(message.unexpected_keys)}",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    detail_base.configure_phase(model, phase, _train_heads_a)
    # The historical event-only depth head is executed only by the inherited
    # compatibility path; it no longer contributes to final depth or losses.
    model.depth_local_head.requires_grad_(False)
    model.conditioned_depth_head.requires_grad_(phase in {"adapter", "joint"})
    if phase in {"adapter", "joint"}:
        assert any(p.requires_grad for p in model.conditioned_depth_head.parameters())
        signature = (phase, sum(p.numel() for p in model.conditioned_depth_head.parameters() if p.requires_grad))
        if getattr(model, "_conditioned_trainable_audit", None) != signature:
            model._conditioned_trainable_audit = signature
            print(f"[conditioned audit/{phase}] depth_head={signature[1]}", flush=True)


def optimizer_for(model, phase, args):
    scale_id = id(model.depth_log_scale)
    conditioned_ids = {id(p) for p in model.conditioned_depth_head.parameters()}
    encoder_ids = {id(p) for p in model.event_encoder.parameters()}
    scale, conditioned, encoder, regular = [], [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        pid = id(parameter)
        if pid == scale_id:
            scale.append(parameter)
        elif pid in conditioned_ids:
            conditioned.append(parameter)
        elif pid in encoder_ids:
            encoder.append(parameter)
        else:
            regular.append(parameter)
    groups = []
    if regular:
        groups.append({"params": regular, "lr": args.lr})
    if encoder:
        groups.append({"params": encoder, "lr": 2.0 * args.lr})
    if conditioned:
        groups.append({"params": conditioned, "lr": 5.0 * args.lr})
    if scale:
        groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = schedule_base.criterion_for
    pipeline.save_visual = schedule_base.base.save_visual
    pipeline.UnifiedGeometryContributionModel = ConditionedScheduledLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
