"""Loss adapter that uses frozen reliability to weight event supervision."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

import finetune_event as fe
from mul_loss_fine.event_supported_mv_loss import (
    _make_event_support,
    _normal_gradient_magnitude,
    _normalize_detail_support,
    _stack_output_field,
)
from mul_loss_fine.launcher import make_configured_loss


def _masked_lowpass(value: torch.Tensor, valid: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return value
    if kernel % 2 == 0:
        kernel += 1
    flat_value = value.flatten(0, 1).unsqueeze(1)
    flat_valid = valid.flatten(0, 1).unsqueeze(1).to(dtype=value.dtype)
    numerator = F.avg_pool2d(flat_value * flat_valid, kernel, stride=1, padding=kernel // 2)
    denominator = F.avg_pool2d(flat_valid, kernel, stride=1, padding=kernel // 2).clamp_min(1.0e-4)
    return (numerator / denominator).squeeze(1).view_as(value)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _weighted_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    weight_dx = torch.minimum(weight[..., :, 1:], weight[..., :, :-1])
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    weight_dy = torch.minimum(weight[..., 1:, :], weight[..., :-1, :])
    return _weighted_mean((pred_dx - target_dx).abs(), weight_dx) + _weighted_mean(
        (pred_dy - target_dy).abs(), weight_dy
    )


class FrozenReliabilityWeightedEventLossMixin:
    """Replace event voxels used by event-aware losses with R.detach() * V."""

    def _init_stage2_residual_target_loss(
        self,
        *,
        residual_target_weight: float,
        residual_gradient_weight: float,
        target_highpass_kernel: int,
        target_abs_limit: float,
        geometry_threshold: float,
        reliability_floor: float,
    ) -> None:
        self.stage2_residual_target_weight = float(residual_target_weight)
        self.stage2_residual_gradient_weight = float(residual_gradient_weight)
        self.stage2_target_highpass_kernel = int(target_highpass_kernel)
        self.stage2_target_abs_limit = float(target_abs_limit)
        self.stage2_geometry_threshold = float(geometry_threshold)
        self.stage2_target_reliability_floor = float(reliability_floor)

    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]):
        reliability = _stack_output_field(model_output, "event_reliability")
        model_gate = _stack_output_field(model_output, "event_gate")
        model_event_support = _stack_output_field(model_output, "event_support")
        if reliability is None:
            total, details, aux = super().forward(model_output, views)
            details.update(
                {
                    "stage2_reliability_mean": 0.0,
                    "stage2_gate_mean": 0.0,
                    "stage2_reliability_positive_ratio": 0.0,
                    "stage2_weighted_event_abs_mean": 0.0,
                }
            )
            return total, details, aux

        reliability = reliability.detach()
        weighted_views = []
        weighted_event_sum = reliability.new_tensor(0.0)
        weighted_event_count = reliability.new_tensor(0.0)
        for view_idx, view in enumerate(views):
            weighted_view = dict(view)
            event_voxel = view.get("event_voxel")
            if torch.is_tensor(event_voxel) and event_voxel.numel() > 0:
                rel = reliability[:, view_idx].to(device=event_voxel.device, dtype=event_voxel.dtype)
                weighted_event = event_voxel * rel.unsqueeze(1)
                weighted_view["event_voxel"] = weighted_event
                weighted_event_sum = weighted_event_sum + weighted_event.detach().abs().sum()
                weighted_event_count = weighted_event_count + weighted_event.numel()
            weighted_views.append(weighted_view)

        total, details, aux = super().forward(model_output, weighted_views)
        predicted_delta = _stack_output_field(model_output, "event_delta_log")
        residual_target_loss = reliability.new_tensor(0.0)
        residual_gradient_loss = reliability.new_tensor(0.0)
        target_abs = reliability.new_tensor(0.0)
        predicted_abs = reliability.new_tensor(0.0)
        sign_accuracy = reliability.new_tensor(0.0)
        if predicted_delta is not None and (
            self.stage2_residual_target_weight > 0.0
            or self.stage2_residual_gradient_weight > 0.0
        ):
            final_depth = _stack_output_field(model_output, "depth").to(dtype=predicted_delta.dtype)
            coarse_depth = _stack_output_field(model_output, "depth_coarse").to(dtype=predicted_delta.dtype)
            depth_gt = fe.stack_view_field(views, "depthmap").to(
                device=predicted_delta.device, dtype=predicted_delta.dtype
            )
            intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(
                device=predicted_delta.device, dtype=predicted_delta.dtype
            )
            valid = fe.build_valid_mask(
                views,
                depth_gt,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
            )
            coarse_detached = coarse_depth.detach().clamp_min(self.depth_min)
            raw_target = torch.log(depth_gt.clamp_min(self.depth_min)) - torch.log(coarse_detached)
            target_low = _masked_lowpass(raw_target, valid, self.stage2_target_highpass_kernel)
            target_delta = (raw_target - target_low) * valid.to(dtype=raw_target.dtype)
            if self.stage2_target_abs_limit > 0.0:
                target_delta = target_delta.clamp(
                    -self.stage2_target_abs_limit, self.stage2_target_abs_limit
                )

            gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics)
            batch, seq_len, height, width = depth_gt.shape
            gt_gradient = _normal_gradient_magnitude(gt_normals.flatten(0, 1)).view(
                batch, seq_len, 1, height, width
            )
            geometry = _normalize_detail_support(
                gt_gradient,
                valid,
                threshold=self.stage2_geometry_threshold,
                power=1.0,
            ).squeeze(2).detach()
            rel_weight = self.stage2_target_reliability_floor + (
                1.0 - self.stage2_target_reliability_floor
            ) * reliability
            event_support = _make_event_support(
                views,
                height=height,
                width=width,
                device=predicted_delta.device,
                dtype=predicted_delta.dtype,
                blur_kernel=1,
                dilate_kernel=1,
                threshold=0.20,
                power=2.0,
                top_fraction=0.20,
                mode="temporal_polarity",
            ).detach()
            target_weight = (
                valid.to(dtype=predicted_delta.dtype)
                * geometry
                * rel_weight
                * event_support
            )

            residual_target_loss = _weighted_mean(
                (predicted_delta - target_delta.detach()).abs(), target_weight
            )
            residual_gradient_loss = _weighted_gradient_loss(
                predicted_delta,
                target_delta.detach(),
                target_weight,
            )
            residual_supervision = (
                self.stage2_residual_target_weight * residual_target_loss
                + self.stage2_residual_gradient_weight * residual_gradient_loss
            )
            total = total + residual_supervision
            target_abs = _weighted_mean(target_delta.detach().abs(), target_weight)
            predicted_abs = _weighted_mean(predicted_delta.detach().abs(), target_weight)
            sign_mask = target_weight * (target_delta.detach().abs() >= 1.0e-4).to(target_weight.dtype)
            sign_accuracy = _weighted_mean(
                ((predicted_delta.detach() * target_delta.detach()) > 0).to(target_weight.dtype),
                sign_mask,
            )
            details["extra_loss_total"] = float(details.get("extra_loss_total", 0.0)) + float(
                residual_supervision.detach()
            )
            details["total_loss_with_extra"] = float(total.detach())

        details.update(
            {
                "stage2_reliability_mean": float(reliability.mean()),
                "stage2_gate_mean": float(model_gate.detach().mean()) if model_gate is not None else 0.0,
                "stage2_event_support_mean": float(model_event_support.detach().mean())
                if model_event_support is not None
                else 0.0,
                "stage2_reliability_positive_ratio": float((reliability >= 0.5).float().mean()),
                "stage2_weighted_event_abs_mean": float(
                    (weighted_event_sum / weighted_event_count.clamp_min(1.0)).detach()
                ),
                "stage2_residual_target_loss": float(residual_target_loss.detach()),
                "stage2_residual_gradient_loss": float(residual_gradient_loss.detach()),
                "stage2_target_delta_abs": float(target_abs.detach()),
                "stage2_predicted_delta_abs": float(predicted_abs.detach()),
                "stage2_delta_sign_accuracy": float(sign_accuracy.detach()),
            }
        )
        aux["event_reliability"] = reliability
        if model_gate is not None:
            aux["event_gate"] = model_gate.detach()
        if model_event_support is not None:
            aux["event_motion_density"] = model_event_support.detach()
        return total, details, aux


def make_stage2_reliability_weighted_loss(cfg):
    configured_base = make_configured_loss(cfg)

    class ConfiguredStage2ReliabilityLoss(
        FrozenReliabilityWeightedEventLossMixin,
        configured_base,
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_stage2_residual_target_loss(
                residual_target_weight=float(
                    getattr(cfg.loss, "stage2_residual_target_weight", 1.0)
                ),
                residual_gradient_weight=float(
                    getattr(cfg.loss, "stage2_residual_gradient_weight", 2.0)
                ),
                target_highpass_kernel=int(
                    getattr(cfg.loss, "stage2_target_highpass_kernel", 9)
                ),
                target_abs_limit=float(getattr(cfg.loss, "stage2_target_abs_limit", 0.025)),
                geometry_threshold=float(
                    getattr(cfg.loss, "stage2_target_geometry_threshold", 0.02)
                ),
                reliability_floor=float(
                    getattr(cfg.loss, "stage2_target_reliability_floor", 0.25)
                ),
            )

    return ConfiguredStage2ReliabilityLoss


__all__ = ["make_stage2_reliability_weighted_loss"]
