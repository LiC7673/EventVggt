"""Scale-preserving constraints for event-guided depth detail refinement."""

from typing import Dict, List

import torch
import torch.nn.functional as F

import finetune_event as fe
from mul_loss_fine.event_supported_mv_loss import (
    _normal_gradient_magnitude,
    _normalize_detail_support,
    _stack_output_field,
    _weighted_mean,
)
from mul_loss_fine.image_guided_event_reliability_loss import (
    make_configured_image_guided_event_reliability_loss,
)


class ScalePreservingEventReliabilityLossMixin:
    """Keep the pretrained coarse geometry while learning local event detail."""

    def _init_scale_preserving_loss(
        self,
        *,
        scale_weight: float,
        low_frequency_weight: float,
        non_detail_guard_weight: float,
        low_frequency_kernel: int,
        non_detail_margin: float,
        detail_threshold: float,
    ) -> None:
        self.scale_preserve_weight = float(scale_weight)
        self.low_frequency_preserve_weight = float(low_frequency_weight)
        self.non_detail_guard_weight = float(non_detail_guard_weight)
        self.low_frequency_kernel = max(3, int(low_frequency_kernel))
        if self.low_frequency_kernel % 2 == 0:
            self.low_frequency_kernel += 1
        self.non_detail_margin = max(0.0, float(non_detail_margin))
        self.scale_preserve_detail_threshold = float(detail_threshold)

    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]):
        total_loss, details, aux = super().forward(model_output, views)
        depth_final = _stack_output_field(model_output, "depth")
        depth_coarse = _stack_output_field(model_output, "depth_coarse")
        if depth_final is None or depth_coarse is None:
            details.update(
                {
                    "scale_preserve_loss": 0.0,
                    "low_frequency_preserve_loss": 0.0,
                    "non_detail_guard_loss": 0.0,
                    "scale_preserving_extra_loss": 0.0,
                }
            )
            return total_loss, details, aux

        depth_final = depth_final.float()
        depth_coarse = depth_coarse.to(device=depth_final.device, dtype=depth_final.dtype).detach()
        depth_gt = fe.stack_view_field(views, "depthmap").to(
            device=depth_final.device,
            dtype=depth_final.dtype,
        )
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(
            device=depth_final.device,
            dtype=depth_final.dtype,
        )
        valid = fe.build_valid_mask(
            views,
            depth_gt,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        )
        weight = valid.to(dtype=depth_final.dtype)

        delta_log = (
            torch.log(depth_final.clamp_min(self.depth_min))
            - torch.log(depth_coarse.clamp_min(self.depth_min))
        )

        # A local detail branch should not introduce a global per-frame scale shift.
        frame_shift = (delta_log * weight).sum(dim=(-2, -1)) / weight.sum(
            dim=(-2, -1)
        ).clamp_min(1.0)
        scale_loss = frame_shift.abs().mean()

        # Suppress only the low-frequency part of the correction. High-frequency
        # geometry remains available to the event branch.
        flat_delta = delta_log.flatten(0, 1).unsqueeze(1)
        flat_weight = weight.flatten(0, 1).unsqueeze(1)
        padding = self.low_frequency_kernel // 2
        smooth_num = F.avg_pool2d(
            flat_delta * flat_weight,
            kernel_size=self.low_frequency_kernel,
            stride=1,
            padding=padding,
        )
        smooth_den = F.avg_pool2d(
            flat_weight,
            kernel_size=self.low_frequency_kernel,
            stride=1,
            padding=padding,
        )
        smooth_delta = smooth_num / smooth_den.clamp_min(1e-6)
        low_frequency_loss = _weighted_mean(smooth_delta.abs(), smooth_den.clamp(0.0, 1.0))

        # Corrections are allowed around GT geometry detail, but should not make
        # smooth regions worse than the frozen Multi-LDR coarse prediction.
        gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics)
        batch, seq_len, height, width = valid.shape
        gt_grad = _normal_gradient_magnitude(gt_normals.flatten(0, 1)).view(
            batch, seq_len, 1, height, width
        )
        detail_support = _normalize_detail_support(
            gt_grad,
            valid,
            threshold=self.scale_preserve_detail_threshold,
            power=1.0,
        ).squeeze(2).detach()
        non_detail_weight = weight * (1.0 - detail_support)
        target = aux.get("depth_gt_aligned", depth_gt).to(
            device=depth_final.device,
            dtype=depth_final.dtype,
        )
        target = target.clamp_min(self.depth_min)
        final_rel_error = (depth_final - target).abs() / target
        coarse_rel_error = ((depth_coarse - target).abs() / target).detach()
        guard_map = F.relu(final_rel_error - coarse_rel_error - self.non_detail_margin)
        non_detail_guard_loss = _weighted_mean(
            guard_map.unsqueeze(2),
            non_detail_weight.unsqueeze(2),
        )

        extra = (
            self.scale_preserve_weight * scale_loss
            + self.low_frequency_preserve_weight * low_frequency_loss
            + self.non_detail_guard_weight * non_detail_guard_loss
        )
        total_loss = total_loss + extra
        details.update(
            {
                "scale_preserve_loss": float(scale_loss.detach()),
                "low_frequency_preserve_loss": float(low_frequency_loss.detach()),
                "non_detail_guard_loss": float(non_detail_guard_loss.detach()),
                "scale_preserving_extra_loss": float(extra.detach()),
                "extra_loss_total": float(details.get("extra_loss_total", 0.0)) + float(extra.detach()),
                "total_loss_with_extra": float(total_loss.detach()),
            }
        )
        return total_loss, details, aux


def make_configured_scale_preserving_event_reliability_loss(cfg):
    configured_base = make_configured_image_guided_event_reliability_loss(cfg)

    class ConfiguredScalePreservingEventReliabilityLoss(
        ScalePreservingEventReliabilityLossMixin,
        configured_base,
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_scale_preserving_loss(
                scale_weight=float(getattr(cfg.loss, "scale_preserve_weight", 2.0)),
                low_frequency_weight=float(getattr(cfg.loss, "low_frequency_preserve_weight", 0.5)),
                non_detail_guard_weight=float(getattr(cfg.loss, "non_detail_guard_weight", 0.25)),
                low_frequency_kernel=int(getattr(cfg.loss, "low_frequency_preserve_kernel", 31)),
                non_detail_margin=float(getattr(cfg.loss, "non_detail_guard_margin", 0.002)),
                detail_threshold=float(getattr(cfg.loss, "scale_preserve_detail_threshold", 0.02)),
            )

    return ConfiguredScalePreservingEventReliabilityLoss
