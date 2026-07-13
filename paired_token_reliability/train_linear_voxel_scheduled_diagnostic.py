"""Iteration-scheduled event detail training with staged update diagnostics."""
from __future__ import annotations

import math
import os
import torch
import finetune_event as fe

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_scheduled_diagnostic_model import (
    ScheduledDiagnosticLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as base
from paired_token_reliability import train_linear_voxel_detail_residual as detail_module
from paired_token_reliability import train_unified_geometry_contribution as pipeline


_GLOBAL_DETAIL_STEP = 0


def _cosine_ramp(step, start_step, ramp_steps, start_value, final_value):
    if step <= start_step:
        return float(start_value)
    progress = min(max((step - start_step) / max(ramp_steps, 1), 0.0), 1.0)
    smooth = .5 - .5 * math.cos(math.pi * progress)
    return float(start_value + (final_value - start_value) * smooth)


def schedule_values(step, args):
    detail = _cosine_ramp(step, 500, 2000, .25, 5.0)
    event_normal = _cosine_ramp(step, 500, 2000, .10, args.event_normal_weight)
    depth_normal = _cosine_ramp(step, 2500, 1500, 0.0, args.depth_event_normal_weight)
    return detail, event_normal, depth_normal


def _walk_wrappers(root):
    current = root
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        yield current
        if hasattr(current, "base"):
            current = current.base
        elif hasattr(current, "criterion"):
            current = current.criterion
        else:
            return


def _find_wrapper(root, wrapper_type):
    for current in _walk_wrappers(root):
        if isinstance(current, wrapper_type):
            return current
    raise RuntimeError(f"missing criterion wrapper {wrapper_type.__name__}")


def build_model(cfg, args, device):
    model = ScheduledDiagnosticLinearVoxelModel(
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
        f"[scheduled diagnostic base] scale={float(model.metric_depth_scale):.4f} "
        f"new={len(message.missing_keys)} unused={len(message.unexpected_keys)}",
        flush=True,
    )
    return model.to(device)


def _masked_stats(value, mask):
    selected = value.detach().float().abs()[mask]
    if selected.numel() == 0:
        return "empty"
    return (
        f"mean={float(selected.mean()):.6g},"
        f"p95={float(torch.quantile(selected, .95)):.6g},"
        f"max={float(selected.max()):.6g}"
    )


def _stack_output(output, key):
    return torch.stack([item[key] for item in output.ress], dim=1)


class ScheduledDiagnosticObjective:
    def __init__(self, criterion, args):
        self.criterion = criterion
        self.args = args
        self.detail_objective = _find_wrapper(
            criterion, detail_module.DetailResidualObjective
        )
        self.normal_objective = _find_wrapper(
            criterion, base.NormalDerivativeObjective
        )

    def __call__(self, output, views, *args, **kwargs):
        global _GLOBAL_DETAIL_STEP
        training = torch.is_grad_enabled()
        if training:
            _GLOBAL_DETAIL_STEP += 1
        step = _GLOBAL_DETAIL_STEP
        detail_weight, event_weight, depth_weight = schedule_values(step, self.args)
        self.detail_objective.weight = detail_weight
        self.normal_objective.event_weight = event_weight
        self.normal_objective.depth_weight = depth_weight

        result = self.criterion(output, views, *args, **kwargs)
        reference = result.loss
        result.details["detail_schedule_weight"] = reference.new_tensor(detail_weight)
        result.details["event_derivative_schedule_weight"] = reference.new_tensor(event_weight)
        result.details["depth_derivative_schedule_weight"] = reference.new_tensor(depth_weight)

        if training and step % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[event-weight@{step:05d}] detail={detail_weight:.4f} "
                f"ENd={event_weight:.4f} DNd={depth_weight:.4f}",
                flush=True,
            )

        if training and step % 500 == 0 and int(os.environ.get("RANK", "0")) == 0:
            support = _stack_output(output, "depth_update_actual_support").bool()
            raw = _stack_output(output, "depth_update_raw_ratio")
            bounded = _stack_output(output, "depth_update_bounded_ratio")
            contribution = _stack_output(output, "depth_update_contribution_ratio")
            support_gated = _stack_output(output, "depth_update_support_ratio")
            centered = _stack_output(output, "depth_update_centered_ratio")
            final_absolute = _stack_output(output, "depth_update_final_absolute")
            coarse = _stack_output(output, "depth_coarse")[..., 0].float()
            gt = fe.stack_view_field(views, "depthmap").to(coarse).float()
            valid = result.aux["valid_live"].bool()
            metric_mask = support & valid & torch.isfinite(gt) & torch.isfinite(coarse)
            count = metric_mask.float().sum().clamp_min(1.0)
            mean_delta = (final_absolute.abs() * metric_mask).sum() / count
            mean_depth = (coarse.abs() * metric_mask).sum() / count
            mean_error = ((coarse - gt).abs() * metric_mask).sum() / count
            update_to_depth = mean_delta / mean_depth.clamp_min(1e-6)
            update_to_error = mean_delta / mean_error.clamp_min(1e-6)
            print(
                f"[update-stages@{step:05d}] "
                f"raw({_masked_stats(raw, metric_mask)}) "
                f"bounded({_masked_stats(bounded, metric_mask)}) "
                f"afterC({_masked_stats(contribution, metric_mask)}) "
                f"afterSupport({_masked_stats(support_gated, metric_mask)}) "
                f"centered({_masked_stats(centered, metric_mask)}) "
                f"finalAbs({_masked_stats(final_absolute, metric_mask)}) "
                f"update/depth={float(update_to_depth):.6f} "
                f"update/error={float(update_to_error):.6f}",
                flush=True,
            )
        return result


def criterion_for(args, phase):
    criterion = base.criterion_for(args, phase)
    if phase not in {"adapter", "joint"}:
        return criterion
    return ScheduledDiagnosticObjective(criterion, args)


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = detail_module.configure_phase
    pipeline.optimizer_for = detail_module.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = base.save_visual
    pipeline.UnifiedGeometryContributionModel = ScheduledDiagnosticLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
