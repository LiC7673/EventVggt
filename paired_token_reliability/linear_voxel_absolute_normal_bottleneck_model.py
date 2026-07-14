"""Absolute event normals followed by a strict normal-to-depth bottleneck."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_conditioned_dense_scale_warmup_model import (
    ConditionedDenseScaleWarmupLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class AbsoluteNormalBottleneckLinearVoxelModel(
    ConditionedDenseScaleWarmupLinearVoxelModel
):
    checkpoint_schema = "linear_time_voxel_absolute_normal_bottleneck_v1"

    def __init__(self, *args, pixel_hidden=32, normal_refine_iterations=3,
                 normal_refine_step_limit=.05,
                 normal_bottleneck_warmup_steps=2000, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        geometry_channels = hidden + 4
        self.normal_confidence_head = nn.Sequential(
            nn.Conv2d(geometry_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # event/coarse normal residual (3), current normal (3), confidence (1),
        # and the accumulated log-depth ratio (1). Raw event features are absent.
        self.normal_bottleneck_refiner = nn.Sequential(
            nn.Conv2d(8, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        for head in (self.normal_confidence_head, self.normal_bottleneck_refiner):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.normal_refine_iterations = max(1, int(normal_refine_iterations))
        self.normal_refine_step_limit = float(normal_refine_step_limit)
        self.normal_bottleneck_warmup_steps = max(
            int(self.scale_warmup_steps), int(normal_bottleneck_warmup_steps)
        )

    def _zero_dependency(self, modules):
        value = self.depth_log_scale * 0.0
        for module in modules:
            for parameter in module.parameters():
                value = value + parameter.sum() * 0.0
        return value

    @staticmethod
    def _write_depth(views, output, coarse, final, ratio, regularizer):
        for index, (view, item) in enumerate(zip(views, output.ress)):
            final_i, ratio_i = final[:, index], ratio[:, index]
            item["depth"] = final_i.unsqueeze(-1)
            item["normal"] = depth_to_normals(
                final_i.float(), view["camera_intrinsics"].to(final_i).float()
            )
            item["depth_delta_ratio"] = ratio_i
            item["depth_pixel_update"] = coarse[:, index] * ratio_i
            item["depth_total_update"] = item["depth_pixel_update"]
            item["depth_update_final_absolute"] = item["depth_pixel_update"]
            item["depth_update_centered_ratio"] = ratio_i
            item["depth_update_detail_ratio"] = ratio_i
            item["adapter_update_loss"] = regularizer
            item["depth_update_tv"] = regularizer

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        if not output.ress:
            return output
        feature = self._captured_event_feature
        if feature is None:
            raise RuntimeError("event encoder feature was not captured")
        b, v, hidden, height, width = feature.shape
        coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], dim=1
        ).float()
        intrinsics = torch.stack(
            [view["camera_intrinsics"] for view in views], dim=1
        ).to(coarse).float()
        coarse_normal = depth_to_normals(coarse, intrinsics)
        log_coarse = torch.log(coarse.clamp_min(1e-6))

        # Keep the successful old behavior: the event decoder predicts a full
        # absolute normal directly from the multiscale event feature.
        absolute_event_normal = torch.stack(
            [item["event_normal"] for item in output.ress], dim=1
        ).float()
        absolute_event_normal = F.normalize(
            absolute_event_normal, dim=-1, eps=1e-6
        )
        geometry_input = torch.cat((
            feature.float(), log_coarse.unsqueeze(2),
            coarse_normal.movedim(-1, 2),
        ), dim=2).reshape(b * v, hidden + 4, height, width)
        learned_confidence = torch.sigmoid(self.normal_confidence_head(geometry_input))
        contribution = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        ).float().reshape(b * v, 1, height, width)
        confidence = learned_confidence * contribution

        coarse_n = coarse_normal.movedim(-1, 2).reshape(b * v, 3, height, width)
        event_n = absolute_event_normal.movedim(-1, 2).reshape(
            b * v, 3, height, width
        )
        fused_normal = F.normalize(
            coarse_n + confidence * (event_n - coarse_n), dim=1, eps=1e-6
        )

        # Loss support is local and real; it never masks the predicted map or
        # the dense normal-to-depth convolutional output.
        filtered = self._last_filtered_event_support
        if filtered is None:
            raise RuntimeError("filtered event support was not captured")
        support = filtered.float().reshape(b * v, 1, height, width)
        kernel = max(1, int(self.support_dilation_kernel))
        if kernel > 1:
            support = F.max_pool2d(support, kernel, 1, kernel // 2)
        support = support[:, 0].reshape(b, v, height, width) > 0

        bottleneck_active = not (
            self.training
            and self._scale_warmup_forward_step <= self.normal_bottleneck_warmup_steps
        )
        if not bottleneck_active:
            zero = self._zero_dependency((self.normal_bottleneck_refiner,))
            final = coarse + zero
            ratio = final * 0.0
            self._write_depth(views, output, coarse, final, ratio, ratio.mean())
            iteration_updates = None
        else:
            log_coarse_flat = log_coarse.reshape(b * v, 1, height, width)
            # Never inherit the parent's direct event-depth prediction.
            log_depth = log_coarse_flat.clone()
            intrinsics_flat = intrinsics.reshape(b * v, 3, 3)
            step_limit = max(self.normal_refine_step_limit, 1e-6)
            updates = []
            for _ in range(self.normal_refine_iterations):
                current_depth = torch.exp(log_depth[:, 0])
                current_normal = depth_to_normals(
                    current_depth.unsqueeze(1), intrinsics_flat.unsqueeze(1)
                )[:, 0].movedim(-1, 1)
                residual = fused_normal - current_normal
                log_ratio = log_depth - log_coarse_flat
                refine_input = torch.cat(
                    (residual, current_normal, confidence, log_ratio), dim=1
                )
                baseline_input = torch.cat((
                    torch.zeros_like(residual), current_normal,
                    confidence, log_ratio,
                ), dim=1)
                raw = (
                    self.normal_bottleneck_refiner(refine_input)
                    - self.normal_bottleneck_refiner(baseline_input)
                )
                step = confidence * step_limit * torch.tanh(raw / step_limit)
                log_depth = log_depth + step
                updates.append(step)
            total_limit = min(max(float(self.depth_update_scale), 1e-6), .999)
            ratio_log = (log_depth - log_coarse_flat).clamp(
                min=math.log(1.0 - total_limit),
                max=math.log(1.0 + total_limit),
            )
            final = torch.exp(log_coarse_flat + ratio_log)[:, 0].reshape(
                b, v, height, width
            )
            ratio = final / coarse.clamp_min(1e-6) - 1.0
            dx = ratio[..., :, 1:] - ratio[..., :, :-1]
            dy = ratio[..., 1:, :] - ratio[..., :-1, :]
            regularizer = .5 * (dx.abs().mean() + dy.abs().mean())
            self._write_depth(views, output, coarse, final, ratio, regularizer)
            iteration_updates = torch.cat(updates, dim=1).reshape(
                b, v, self.normal_refine_iterations, height, width
            )

        learned_bv = learned_confidence.reshape(b, v, height, width)
        confidence_bv = confidence.reshape(b, v, height, width)
        fused_bv = fused_normal.reshape(b, v, 3, height, width).movedim(2, -1)
        for index, item in enumerate(output.ress):
            item["event_normal"] = absolute_event_normal[:, index]
            item["event_normal_absolute"] = absolute_event_normal[:, index]
            item["event_normal_support"] = support[:, index]
            item["learned_normal_confidence"] = learned_bv[:, index]
            item["normal_confidence"] = confidence_bv[:, index]
            item["normal_refine_target"] = fused_bv[:, index]
            item["normal_bottleneck_active"] = item["depth"].new_tensor(
                float(bottleneck_active)
            )
            if iteration_updates is not None:
                item["normal_refine_iteration_updates"] = iteration_updates[:, index]
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["AbsoluteNormalBottleneckLinearVoxelModel"]
