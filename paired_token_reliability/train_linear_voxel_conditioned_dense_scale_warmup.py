"""Dense depth-update training with 1k scale-only warmup."""
from __future__ import annotations

import math
import os

import finetune_event as fe
import torch
import torch.nn.functional as F

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_conditioned_dense_scale_warmup_model import (
    ConditionedDenseScaleWarmupLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_conditioned_soft_dc as softdc
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as normal_base
from paired_token_reliability import train_linear_voxel_detail_residual as detail_base
from paired_token_reliability import train_linear_voxel_scheduled_diagnostic as diagnostic
from paired_token_reliability import train_unified_geometry_contribution as pipeline


SCALE_WARMUP_STEPS = 1000
_GLOBAL_DENSE_STEP = 0


def _cosine_ramp(step, start, duration, final):
    if step <= start:
        return 0.0
    progress = min(max((step - start) / max(duration, 1), 0.0), 1.0)
    return float(final) * (0.5 - 0.5 * math.cos(math.pi * progress))


def schedule(step, args):
    detail = _cosine_ramp(step, SCALE_WARMUP_STEPS, 2000, 5.0)
    event_normal = _cosine_ramp(
        step, SCALE_WARMUP_STEPS, 2000, args.event_normal_weight
    )
    depth_normal = _cosine_ramp(
        step, SCALE_WARMUP_STEPS + 1500, 1500, args.depth_event_normal_weight
    )
    return detail, event_normal, depth_normal


def _weighted_smooth_l1(pred, target, weight):
    value = F.smooth_l1_loss(pred, target, beta=.01, reduction="none")
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class DenseScheduledObjective:
    def __init__(self, criterion, args, inherited_detail, normal_objective):
        self.criterion = criterion
        self.args = args
        self.inherited_detail = inherited_detail
        self.normal_objective = normal_objective

    def __call__(self, output, views, *args, **kwargs):
        global _GLOBAL_DENSE_STEP
        training = torch.is_grad_enabled()
        if training:
            _GLOBAL_DENSE_STEP += 1
        step = _GLOBAL_DENSE_STEP
        detail_weight, event_weight, depth_weight = schedule(step, self.args)
        self.inherited_detail.weight = 0.0
        self.normal_objective.event_weight = event_weight
        self.normal_objective.depth_weight = depth_weight

        result = self.criterion(output, views, *args, **kwargs)
        pred = torch.stack(
            [item["depth_delta_ratio"] for item in output.ress], dim=1
        ).float()
        coarse = torch.stack(
            [item["depth_coarse"] for item in output.ress], dim=1
        )[..., 0].float()
        gt = fe.stack_view_field(views, "depthmap").to(pred).float()
        valid = (
            result.aux["valid_live"].bool()
            & torch.isfinite(gt) & torch.isfinite(coarse) & (coarse > 1e-6)
        )
        event_support = torch.stack(
            [item["event_normal_support"] for item in output.ress], dim=1
        ).bool()
        target_limit = float(output.ress[0]["depth_update_target_limit"].detach())
        target = (gt / coarse.detach().clamp_min(1e-6) - 1.0).clamp(
            -target_limit, target_limit
        )
        # Dense supervision over the whole valid object. Events increase the
        # local priority but never mask non-event pixels out of the objective.
        weight = valid.to(pred) * (1.0 + 2.0 * event_support.to(pred))
        dense_detail = _weighted_smooth_l1(pred, target, weight)
        result.loss = result.loss + detail_weight * dense_detail
        result.details["dense_depth_residual"] = dense_detail
        result.details["detail_schedule_weight"] = result.loss.new_tensor(detail_weight)
        result.details["event_derivative_schedule_weight"] = result.loss.new_tensor(event_weight)
        result.details["depth_derivative_schedule_weight"] = result.loss.new_tensor(depth_weight)

        if training and step % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[dense-weight@{step:05d}] detail={detail_weight:.4f} "
                f"ENd={event_weight:.4f} DNd={depth_weight:.4f} "
                f"scale_warmup={int(step <= SCALE_WARMUP_STEPS)}",
                flush=True,
            )
        if training and step % 500 == 0 and int(os.environ.get("RANK", "0")) == 0:
            absolute = torch.stack(
                [item["depth_update_final_absolute"] for item in output.ress], dim=1
            )
            count = valid.float().sum().clamp_min(1.0)
            mean_update = (absolute.abs() * valid).sum() / count
            mean_depth = (coarse.abs() * valid).sum() / count
            mean_error = ((coarse - gt).abs() * valid).sum() / count
            event_count = (valid & event_support).float().sum().clamp_min(1.0)
            outside_count = (valid & ~event_support).float().sum().clamp_min(1.0)
            event_mean = (pred.abs() * (valid & event_support)).sum() / event_count
            outside_mean = (pred.abs() * (valid & ~event_support)).sum() / outside_count
            print(
                f"[dense-update@{step:05d}] "
                f"ratio_mean={float((pred.abs()*valid).sum()/count):.6f} "
                f"event_mean={float(event_mean):.6f} outside_mean={float(outside_mean):.6f} "
                f"update/depth={float(mean_update/mean_depth.clamp_min(1e-6)):.6f} "
                f"update/error={float(mean_update/mean_error.clamp_min(1e-6)):.6f}",
                flush=True,
            )
        return result


def build_model(cfg, args, device):
    m = cfg.model
    model = ConditionedDenseScaleWarmupLinearVoxelModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        event_dc_limit=float(getattr(m, "event_dc_limit", .50)),
        event_residual_target_limit=float(getattr(m, "event_residual_target_limit", .50)),
        scale_warmup_steps=int(getattr(m, "scale_warmup_steps", SCALE_WARMUP_STEPS)),
        event_min_pixel_mass=float(getattr(m, "event_min_pixel_mass", .10)),
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
        f"[dense scale-warmup base] scale={float(model.metric_depth_scale):.4f} "
        f"warmup={model.scale_warmup_steps} limit={model.depth_update_scale:.3f} "
        f"min_event_mass={model.event_min_pixel_mass:.3f} hard_support=OFF",
        flush=True,
    )
    return model.to(device)


def criterion_for(args, phase):
    criterion = normal_base.criterion_for(args, phase)
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
    inherited_detail = diagnostic._find_wrapper(
        criterion, detail_base.DetailResidualObjective
    )
    normal_objective = diagnostic._find_wrapper(
        criterion, normal_base.NormalDerivativeObjective
    )
    return DenseScheduledObjective(
        criterion, args, inherited_detail, normal_objective
    )


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = softdc.conditioned_base.configure_phase
    pipeline.optimizer_for = softdc.conditioned_base.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = normal_base.save_visual
    pipeline.UnifiedGeometryContributionModel = ConditionedDenseScaleWarmupLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
