"""C-connected event-normal confidence with iterative dense depth refinement."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_conditioned_dense_scale_warmup_model import (
    ConditionedDenseScaleWarmupLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class ConditionedConfidenceNormalRefineLinearVoxelModel(
    ConditionedDenseScaleWarmupLinearVoxelModel
):
    checkpoint_schema = "linear_time_voxel_c_confidence_normal_refine_v1"

    def __init__(self, *args, pixel_hidden=32, normal_refine_iterations=3,
                 normal_refine_step_limit=.05, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        geometry_channels = hidden + 4  # event feature, log coarse, coarse normal
        self.event_normal_delta_head = nn.Sequential(
            nn.Conv2d(geometry_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        self.normal_confidence_head = nn.Sequential(
            nn.Conv2d(geometry_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # event feature + current/coarse log depth + current/target normal + confidence
        self.normal_depth_refiner = nn.Sequential(
            nn.Conv2d(hidden + 9, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.event_normal_delta_head[-1].weight)
        nn.init.zeros_(self.event_normal_delta_head[-1].bias)
        nn.init.zeros_(self.normal_depth_refiner[-1].weight)
        nn.init.zeros_(self.normal_depth_refiner[-1].bias)
        nn.init.zeros_(self.normal_confidence_head[-1].weight)
        nn.init.zeros_(self.normal_confidence_head[-1].bias)
        self.normal_refine_iterations = max(1, int(normal_refine_iterations))
        self.normal_refine_step_limit = float(normal_refine_step_limit)

    def _new_head_zero_dependency(self):
        value = self.depth_log_scale * 0.0
        for module in (
            self.event_normal_delta_head,
            self.normal_confidence_head,
            self.normal_depth_refiner,
        ):
            for parameter in module.parameters():
                value = value + parameter.sum() * 0.0
        return value

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        if not output.ress:
            return output
        warmup = bool(float(output.ress[0]["scale_warmup_active"].detach()))
        if warmup:
            # Keep new heads in the DDP graph while scale alone determines the
            # actual prediction during the first 1000 training forwards.
            zero = self._new_head_zero_dependency()
            for item in output.ress:
                item["depth"] = item["depth"] + zero
                item["normal_confidence"] = item["depth"][..., 0] * 0.0
                item["learned_normal_confidence"] = item["normal_confidence"]
                item["event_normal_delta"] = item["normal"].detach() * 0.0
                item["normal_refine_target"] = item["normal"]
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
        geometry_input = torch.cat(
            (feature.float(), log_coarse.unsqueeze(2), coarse_normal.movedim(-1, 2)),
            dim=2,
        ).reshape(b * v, hidden + 4, height, width)

        delta_normal = torch.tanh(self.event_normal_delta_head(geometry_input))
        learned_confidence = torch.sigmoid(self.normal_confidence_head(geometry_input))
        contribution = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        ).float().reshape(b * v, 1, height, width)
        # C and normal confidence have distinct meanings and are connected
        # multiplicatively. This remains a soft confidence, never a hard mask.
        normal_confidence = learned_confidence * contribution
        coarse_n = coarse_normal.movedim(-1, 2).reshape(b * v, 3, height, width)
        target_normal = F.normalize(
            coarse_n + normal_confidence * delta_normal, dim=1, eps=1e-6
        )

        initial_depth = torch.stack(
            [item["depth"][..., 0] for item in output.ress], dim=1
        ).float()
        log_depth = torch.log(initial_depth.clamp_min(1e-6)).reshape(
            b * v, 1, height, width
        )
        log_coarse_flat = log_coarse.reshape(b * v, 1, height, width)
        feature_flat = feature.reshape(b * v, hidden, height, width).float()
        intrinsics_flat = intrinsics.reshape(b * v, 3, 3)
        step_limit = max(self.normal_refine_step_limit, 1e-6)
        iteration_updates = []
        for _ in range(self.normal_refine_iterations):
            current_depth = torch.exp(log_depth[:, 0])
            current_normal = depth_to_normals(
                current_depth.unsqueeze(1), intrinsics_flat.unsqueeze(1)
            )[:, 0].movedim(-1, 1)
            refine_input = torch.cat(
                (
                    feature_flat, log_depth, log_coarse_flat,
                    current_normal, target_normal, normal_confidence,
                ),
                dim=1,
            )
            raw_step = self.normal_depth_refiner(refine_input)
            step = step_limit * torch.tanh(raw_step / step_limit)
            log_depth = log_depth + step
            iteration_updates.append(step)

        # Preserve the experiment's +/-50% total relative-depth permission.
        total_limit = min(max(float(self.depth_update_scale), 1e-6), .999)
        log_ratio = (log_depth - log_coarse_flat).clamp(
            min=math.log(1.0 - total_limit), max=math.log(1.0 + total_limit)
        )
        final = (torch.exp(log_coarse_flat + log_ratio)[:, 0]).reshape(
            b, v, height, width
        )
        ratio = final / coarse.clamp_min(1e-6) - 1.0
        dx = ratio[..., :, 1:] - ratio[..., :, :-1]
        dy = ratio[..., 1:, :] - ratio[..., :-1, :]
        tv = .5 * (dx.abs().mean() + dy.abs().mean())

        delta_normal_bv = delta_normal.reshape(b, v, 3, height, width).movedim(2, -1)
        learned_bv = learned_confidence.reshape(b, v, height, width)
        confidence_bv = normal_confidence.reshape(b, v, height, width)
        target_bv = target_normal.reshape(b, v, 3, height, width).movedim(2, -1)
        for index, (view, item) in enumerate(zip(views, output.ress)):
            final_i = final[:, index]
            ratio_i = ratio[:, index]
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
            item["event_normal_delta"] = delta_normal_bv[:, index]
            item["learned_normal_confidence"] = learned_bv[:, index]
            item["normal_confidence"] = confidence_bv[:, index]
            item["normal_refine_target"] = target_bv[:, index]
            # Existing ENd/DNd losses now operate on the fused coarse+event
            # target rather than an unrelated absolute-normal decoder.
            item["event_normal"] = target_bv[:, index]
            item["normal_refine_iteration_updates"] = torch.cat(
                iteration_updates, dim=1
            ).reshape(b, v, self.normal_refine_iterations, height, width)[:, index]

        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["ConditionedConfidenceNormalRefineLinearVoxelModel"]
