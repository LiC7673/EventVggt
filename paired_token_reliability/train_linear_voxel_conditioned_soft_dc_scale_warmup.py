"""Train Soft-DC after a genuine 1k-step metric-scale warmup."""
from __future__ import annotations

import math
import os

import finetune_event as fe
import torch

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_conditioned_soft_dc_scale_warmup_model import (
    ScaleWarmupConditionedSoftDCLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_conditioned_soft_dc as base
from paired_token_reliability import train_linear_voxel_scheduled_diagnostic as diagnostic
from paired_token_reliability import train_unified_geometry_contribution as pipeline


SCALE_WARMUP_STEPS = 1000


def _cosine_ramp(step, start_step, ramp_steps, final_value):
    if step <= start_step:
        return 0.0
    progress = min(max((step - start_step) / max(ramp_steps, 1), 0.0), 1.0)
    return float(final_value) * (0.5 - 0.5 * math.cos(math.pi * progress))


def scale_warmup_schedule_values(step, args):
    # First align metric scale.  Then learn event geometry; only after the
    # event-normal derivative has had time to form do we make final depth
    # follow it.
    detail = _cosine_ramp(step, SCALE_WARMUP_STEPS, 2000, 5.0)
    event_normal = _cosine_ramp(
        step, SCALE_WARMUP_STEPS, 2000, args.event_normal_weight
    )
    depth_normal = _cosine_ramp(
        step, SCALE_WARMUP_STEPS + 1500, 1500, args.depth_event_normal_weight
    )
    return detail, event_normal, depth_normal


class ScaleWarmupSoftDCObjective(base.SoftDCScheduledObjective):
    def __call__(self, output, views, *args, **kwargs):
        next_step = base._GLOBAL_SOFT_DC_STEP + (
            1 if torch.is_grad_enabled() else 0
        )
        # The DC target is also event geometry.  Do not let it compete with
        # scale calibration during warmup; ramp it together with detail.
        dc_weight = _cosine_ramp(next_step, SCALE_WARMUP_STEPS, 2000, 1.0)
        old_direct, old_excess = base.DC_DIRECT_WEIGHT, base.DC_EXCESS_WEIGHT
        base.DC_DIRECT_WEIGHT = dc_weight
        base.DC_EXCESS_WEIGHT = 0.01 * dc_weight
        try:
            result = super().__call__(output, views, *args, **kwargs)
        finally:
            base.DC_DIRECT_WEIGHT, base.DC_EXCESS_WEIGHT = old_direct, old_excess

        result.details["scale_warmup_active"] = result.loss.new_tensor(
            float(next_step <= SCALE_WARMUP_STEPS)
        )
        result.details["event_dc_schedule_weight"] = result.loss.new_tensor(dc_weight)
        if (
            torch.is_grad_enabled()
            and next_step % 100 == 0
            and int(os.environ.get("RANK", "0")) == 0
        ):
            print(
                f"[scale-warmup@{next_step:05d}] "
                f"active={int(next_step <= SCALE_WARMUP_STEPS)} "
                f"dc_weight={dc_weight:.4f}",
                flush=True,
            )
        return result


def build_model(cfg, args, device):
    model = ScaleWarmupConditionedSoftDCLinearVoxelModel(
        img_size=int(cfg.model.img_size), patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(cfg.model, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(cfg.model, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(cfg.model, "depth_update_scale", 1.0)),
        event_decay_tau=float(getattr(cfg.model, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(cfg.model, "depth_log_scale_limit", 2.0)),
        event_dc_limit=float(getattr(cfg.model, "event_dc_limit", .03)),
        event_residual_target_limit=float(
            getattr(cfg.model, "event_residual_target_limit", .25)
        ),
        scale_warmup_steps=int(
            getattr(cfg.model, "scale_warmup_steps", SCALE_WARMUP_STEPS)
        ),
        event_min_pixel_mass=float(
            getattr(cfg.model, "event_min_pixel_mass", .10)
        ),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    message = model.load_state_dict(
        strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))), strict=False
    )
    required = [
        key for key in message.missing_keys
        if key.startswith(("aggregator.", "camera_head."))
    ]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(
        f"[soft-DC scale-warmup base] scale={float(model.metric_depth_scale):.4f} "
        f"warmup={model.scale_warmup_steps} dc_limit={model.event_dc_limit:.3f} "
        f"target_limit={model.event_residual_target_limit:.3f} "
        f"min_event_mass={model.event_min_pixel_mass:.4f} "
        f"support_kernel={model.support_dilation_kernel}",
        flush=True,
    )
    return model.to(device)


def criterion_for(args, phase):
    criterion = base.normal_base.criterion_for(args, phase)
    # The inherited linear-voxel criterion historically ignored ``phase`` and
    # therefore kept E_full/E_geo decomposition, pair and ranking objectives
    # alive during Stage A.  Stage A must be an oracle-E_geo geometry stage:
    # no target derived from E_full is allowed into its objective.
    for current in diagnostic._walk_wrappers(criterion):
        if hasattr(current, "decomposition_weight"):
            if phase == "adapter":
                current.decomposition_weight = 0.0
                current.pair_weight = 0.0
                current.budget_weight = 0.0
                current.geometry_rank_weight = 0.0
            break
    if phase not in {"adapter", "joint"}:
        return criterion
    detail_objective = diagnostic._find_wrapper(
        criterion, base.detail_module.DetailResidualObjective
    )
    normal_objective = diagnostic._find_wrapper(
        criterion, base.normal_base.NormalDerivativeObjective
    )
    return ScaleWarmupSoftDCObjective(
        criterion, args, detail_objective, normal_objective
    )


def main(argv=None):
    diagnostic.schedule_values = scale_warmup_schedule_values
    pipeline.build_model = build_model
    pipeline.configure_phase = base.conditioned_base.configure_phase
    pipeline.optimizer_for = base.conditioned_base.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = base.normal_base.save_visual
    pipeline.UnifiedGeometryContributionModel = (
        ScaleWarmupConditionedSoftDCLinearVoxelModel
    )
    pipeline.main(argv)


if __name__ == "__main__":
    main()
