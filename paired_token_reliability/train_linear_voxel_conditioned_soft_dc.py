"""Scheduled coarse-conditioned event geometry without hard mean removal."""
from __future__ import annotations

import math
import os
import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_conditioned_soft_dc_model import (
    ConditionedSoftDCLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_conditioned_scheduled as conditioned_base
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as normal_base
from paired_token_reliability import train_linear_voxel_detail_residual as detail_module
from paired_token_reliability import train_linear_voxel_scheduled_diagnostic as diagnostic
from paired_token_reliability import train_unified_geometry_contribution as pipeline


_GLOBAL_SOFT_DC_STEP = 0
DC_DIRECT_WEIGHT = 1.0
DC_EXCESS_WEIGHT = .01


def build_model(cfg, args, device):
    model = ConditionedSoftDCLinearVoxelModel(
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
        f"[conditioned soft-DC base] scale={float(model.metric_depth_scale):.4f} "
        f"dc_limit={model.event_dc_limit:.3f} "
        f"target_limit={model.event_residual_target_limit:.3f} "
        f"new={len(message.missing_keys)} unused={len(message.unexpected_keys)}",
        flush=True,
    )
    return model.to(device)


def _masked_mean(value, mask):
    weight = mask.to(value)
    return (value * weight).sum(dim=(-2, -1), keepdim=True) / weight.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1.0)


def _masked_smooth_l1(pred, target, mask):
    error = F.smooth_l1_loss(pred, target, beta=.01, reduction="none")
    weight = mask.to(error)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


class SoftDCScheduledObjective:
    def __init__(self, criterion, args, detail_objective, normal_objective):
        self.criterion = criterion
        self.args = args
        self.detail_objective = detail_objective
        self.normal_objective = normal_objective

    def __call__(self, output, views, *args, **kwargs):
        global _GLOBAL_SOFT_DC_STEP
        training = torch.is_grad_enabled()
        if training:
            _GLOBAL_SOFT_DC_STEP += 1
        step = _GLOBAL_SOFT_DC_STEP
        detail_weight, event_weight, depth_weight = diagnostic.schedule_values(step, self.args)
        # Disable the inherited hard-centered target objective.  This wrapper
        # supervises DC and local detail separately below.
        self.detail_objective.weight = 0.0
        self.normal_objective.event_weight = event_weight
        self.normal_objective.depth_weight = depth_weight

        result = self.criterion(output, views, *args, **kwargs)
        pred = torch.stack([item["depth_delta_ratio"] for item in output.ress], dim=1).float()
        coarse = torch.stack([item["depth_coarse"] for item in output.ress], dim=1)[..., 0].float()
        gt = fe.stack_view_field(views, "depthmap").to(pred).float()
        valid = result.aux["valid_live"].bool()
        support = torch.stack(
            [item["depth_update_actual_support"] for item in output.ress], dim=1
        ).bool()
        mask = valid & support & torch.isfinite(gt) & torch.isfinite(coarse) & (coarse > 1e-6)

        dc_limit = float(output.ress[0]["depth_update_dc_limit"].detach())
        target_limit = float(output.ress[0]["depth_update_target_limit"].detach())
        target = (gt / coarse.detach().clamp_min(1e-6) - 1.0).clamp(
            -target_limit, target_limit
        )
        pred_dc = _masked_mean(pred, mask)
        target_dc = _masked_mean(target, mask).clamp(-dc_limit, dc_limit)
        pred_detail = (pred - pred_dc) * mask
        target_detail = (target - _masked_mean(target, mask)) * mask
        detail_loss = _masked_smooth_l1(pred_detail, target_detail, mask)
        view_valid = mask.flatten(-2).any(dim=-1)
        dc_loss = F.smooth_l1_loss(
            pred_dc[..., 0, 0][view_valid], target_dc[..., 0, 0][view_valid],
            beta=.005, reduction="mean",
        ) if view_valid.any() else pred.sum() * 0.0
        dc_excess = output.ress[0]["depth_update_dc_excess_loss"]

        result.loss = (
            result.loss
            + detail_weight * detail_loss
            + DC_DIRECT_WEIGHT * dc_loss
            + DC_EXCESS_WEIGHT * dc_excess
        )
        result.details["detail_residual"] = detail_loss
        result.details["detail_schedule_weight"] = result.loss.new_tensor(detail_weight)
        result.details["event_dc"] = dc_loss
        result.details["event_dc_excess"] = dc_excess
        result.details["event_derivative_schedule_weight"] = result.loss.new_tensor(event_weight)
        result.details["depth_derivative_schedule_weight"] = result.loss.new_tensor(depth_weight)

        if training and step % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[soft-dc-weight@{step:05d}] detail={detail_weight:.4f} "
                f"ENd={event_weight:.4f} DNd={depth_weight:.4f} "
                f"dc={float(pred_dc.abs().mean().detach()):.6f} "
                f"target_dc={float(target_dc.abs().mean().detach()):.6f}",
                flush=True,
            )

        if training and step % 500 == 0 and int(os.environ.get("RANK", "0")) == 0:
            raw = torch.stack([x["depth_update_raw_ratio"] for x in output.ress], dim=1)
            bounded = torch.stack([x["depth_update_bounded_ratio"] for x in output.ress], dim=1)
            after_c = torch.stack([x["depth_update_contribution_ratio"] for x in output.ress], dim=1)
            after_support = torch.stack([x["depth_update_support_ratio"] for x in output.ress], dim=1)
            detail = torch.stack([x["depth_update_detail_ratio"] for x in output.ress], dim=1)
            final_abs = torch.stack([x["depth_update_final_absolute"] for x in output.ress], dim=1)
            count = mask.float().sum().clamp_min(1.0)
            mean_delta = (final_abs.abs() * mask).sum() / count
            mean_depth = (coarse.abs() * mask).sum() / count
            mean_error = ((coarse - gt).abs() * mask).sum() / count
            print(
                f"[soft-dc-stages@{step:05d}] "
                f"raw({diagnostic._masked_stats(raw, mask)}) "
                f"bounded({diagnostic._masked_stats(bounded, mask)}) "
                f"afterC({diagnostic._masked_stats(after_c, mask)}) "
                f"afterSupport({diagnostic._masked_stats(after_support, mask)}) "
                f"detail({diagnostic._masked_stats(detail, mask)}) "
                f"finalAbs({diagnostic._masked_stats(final_abs, mask)}) "
                f"update/depth={float(mean_delta/mean_depth.clamp_min(1e-6)):.6f} "
                f"update/error={float(mean_delta/mean_error.clamp_min(1e-6)):.6f}",
                flush=True,
            )
        return result


def criterion_for(args, phase):
    criterion = normal_base.criterion_for(args, phase)
    if phase not in {"adapter", "joint"}:
        return criterion
    detail_objective = diagnostic._find_wrapper(
        criterion, detail_module.DetailResidualObjective
    )
    normal_objective = diagnostic._find_wrapper(
        criterion, normal_base.NormalDerivativeObjective
    )
    return SoftDCScheduledObjective(
        criterion, args, detail_objective, normal_objective
    )


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = conditioned_base.configure_phase
    pipeline.optimizer_for = conditioned_base.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = normal_base.save_visual
    pipeline.UnifiedGeometryContributionModel = ConditionedSoftDCLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
