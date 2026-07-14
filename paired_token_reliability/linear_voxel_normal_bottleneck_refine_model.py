"""Event-normal bottleneck: depth can only use event evidence through normals."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_conditioned_dense_scale_warmup_model import (
    ConditionedDenseScaleWarmupLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class NormalBottleneckRefineLinearVoxelModel(
    ConditionedDenseScaleWarmupLinearVoxelModel
):
    checkpoint_schema = "linear_time_voxel_normal_bottleneck_refine_v1"

    def __init__(self, *args, pixel_hidden=32, normal_refine_iterations=3,
                 normal_refine_step_limit=.05,
                 normal_bottleneck_warmup_steps=2000, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        geometry_channels = hidden + 4
        self.event_normal_delta_head = nn.Sequential(
            nn.Conv2d(geometry_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        self.normal_confidence_head = nn.Sequential(
            nn.Conv2d(geometry_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # No raw event feature and no legacy event-depth prediction are inputs.
        # residual normal (3), current normal (3), confidence (1), log ratio (1).
        self.normal_bottleneck_refiner = nn.Sequential(
            nn.Conv2d(8, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        for head in (
            self.event_normal_delta_head,
            self.normal_confidence_head,
            self.normal_bottleneck_refiner,
        ):
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

    def _write_zero_depth_update(self, views, output, coarse, zero):
        for index, (view, item) in enumerate(zip(views, output.ress)):
            final = coarse[:, index] + zero
            ratio = final * 0.0
            item["depth"] = final.unsqueeze(-1)
            item["normal"] = depth_to_normals(
                final.float(), view["camera_intrinsics"].to(final).float()
            )
            item["depth_delta_ratio"] = ratio
            item["depth_pixel_update"] = ratio
            item["depth_total_update"] = ratio
            item["depth_update_final_absolute"] = ratio
            item["depth_update_centered_ratio"] = ratio
            item["depth_update_detail_ratio"] = ratio
            item["adapter_update_loss"] = ratio.mean() * 0.0

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        if not output.ress:
            return output

        coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], dim=1
        ).float()
        scale_warmup = bool(float(output.ress[0]["scale_warmup_active"].detach()))
        if scale_warmup:
            zero = self._zero_dependency((
                self.event_normal_delta_head,
                self.normal_confidence_head,
                self.normal_bottleneck_refiner,
            ))
            self._write_zero_depth_update(views, output, coarse, zero)
            for item in output.ress:
                item["normal_confidence"] = item["depth"][..., 0] * 0.0
                item["learned_normal_confidence"] = item["normal_confidence"]
                item["event_normal_delta"] = item["normal"].detach() * 0.0
                item["normal_refine_target"] = item["normal"]
                item["event_normal"] = item["normal"]
                item["normal_bottleneck_active"] = item["depth"].new_tensor(0.0)
            return output

        feature = self._captured_event_feature
        if feature is None:
            raise RuntimeError("event encoder feature was not captured")
        b, v, hidden, height, width = feature.shape
        intrinsics = torch.stack(
            [view["camera_intrinsics"] for view in views], dim=1
        ).to(coarse).float()
        coarse_normal = depth_to_normals(coarse, intrinsics)
        log_coarse = torch.log(coarse.clamp_min(1e-6))
        geometry_input = torch.cat((
            feature.float(), log_coarse.unsqueeze(2),
            coarse_normal.movedim(-1, 2),
        ), dim=2).reshape(b * v, hidden + 4, height, width)

        delta_normal = torch.tanh(self.event_normal_delta_head(geometry_input))
        learned_confidence = torch.sigmoid(self.normal_confidence_head(geometry_input))
        contribution = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        ).float().reshape(b * v, 1, height, width)
        confidence = learned_confidence * contribution
        coarse_n = coarse_normal.movedim(-1, 2).reshape(b * v, 3, height, width)
        target_normal = F.normalize(
            coarse_n + confidence * delta_normal, dim=1, eps=1e-6
        )

        # Stage N: after scale alignment, learn event normals for another 1k
        # steps before granting the normal-to-depth module any correction.
        normal_only = self.training and (
            self._scale_warmup_forward_step <= self.normal_bottleneck_warmup_steps
        )
        if normal_only:
            zero = self._zero_dependency((self.normal_bottleneck_refiner,))
            self._write_zero_depth_update(views, output, coarse, zero)
            iteration_updates = None
        else:
            log_coarse_flat = log_coarse.reshape(b * v, 1, height, width)
            # Crucial: start from coarse, not the parent's direct event-depth result.
            log_depth = log_coarse_flat.clone()
            intrinsics_flat = intrinsics.reshape(b * v, 3, 3)
            step_limit = max(self.normal_refine_step_limit, 1e-6)
            updates = []
            for _ in range(self.normal_refine_iterations):
                current_depth = torch.exp(log_depth[:, 0])
                current_normal = depth_to_normals(
                    current_depth.unsqueeze(1), intrinsics_flat.unsqueeze(1)
                )[:, 0].movedim(-1, 1)
                residual = target_normal - current_normal
                log_ratio = log_depth - log_coarse_flat
                refine_input = torch.cat(
                    (residual, current_normal, confidence, log_ratio), dim=1
                )
                # Context-only output is subtracted: zero normal residual must
                # produce exactly zero depth update.
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
            tv = .5 * (dx.abs().mean() + dy.abs().mean())
            iteration_updates = torch.cat(updates, dim=1).reshape(
                b, v, self.normal_refine_iterations, height, width
            )
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
                item["adapter_update_loss"] = tv
                item["depth_update_tv"] = tv

        delta_bv = delta_normal.reshape(b, v, 3, height, width).movedim(2, -1)
        learned_bv = learned_confidence.reshape(b, v, height, width)
        confidence_bv = confidence.reshape(b, v, height, width)
        target_bv = target_normal.reshape(b, v, 3, height, width).movedim(2, -1)
        for index, item in enumerate(output.ress):
            item["event_normal_delta"] = delta_bv[:, index]
            item["learned_normal_confidence"] = learned_bv[:, index]
            item["normal_confidence"] = confidence_bv[:, index]
            item["normal_refine_target"] = target_bv[:, index]
            item["event_normal"] = target_bv[:, index]
            item["normal_bottleneck_active"] = item["depth"].new_tensor(
                float(not normal_only)
            )
            if iteration_updates is not None:
                item["normal_refine_iteration_updates"] = iteration_updates[:, index]
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["NormalBottleneckRefineLinearVoxelModel"]
