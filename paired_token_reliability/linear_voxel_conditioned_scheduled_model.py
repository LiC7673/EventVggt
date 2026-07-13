"""Coarse-conditioned event detail refinement with staged diagnostics."""
from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_detail_normal_derivative_model import (
    DetailNormalDerivativeLinearVoxelModel,
)
from paired_token_reliability.linear_voxel_detail_residual_model import support_center
from paired_token_reliability.signed_multiscale_model import signed_support
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class ConditionedScheduledLinearVoxelModel(DetailNormalDerivativeLinearVoxelModel):
    """Event locates detail; calibrated RGB geometry determines its correction."""
    checkpoint_schema = "linear_time_voxel_conditioned_scheduled_v1"

    def __init__(self, *args, pixel_hidden=32, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        self.conditioned_depth_head = nn.Sequential(
            nn.Conv2d(hidden + 4, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.conditioned_depth_head[-1].weight)
        nn.init.zeros_(self.conditioned_depth_head[-1].bias)
        self._captured_event_feature = None
        self._event_feature_hook = self.event_encoder.register_forward_hook(
            self._capture_event_feature
        )

    def _capture_event_feature(self, _module, _inputs, output):
        self._captured_event_feature = output

    def forward(self, views, *args, **kwargs):
        self._captured_event_feature = None
        output = super().forward(views, *args, **kwargs)
        # The inherited audit describes the obsolete event-only depth head.
        # Keep its counter away from 500-step boundaries; the scheduled
        # objective prints the conditioned head's exact stages instead.
        self._calibration_forward_count = 1
        feature = self._captured_event_feature
        if feature is None:
            raise RuntimeError("event_encoder did not produce a captured feature")

        b, v, hidden, height, width = feature.shape
        coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], dim=1
        ).float()
        intrinsics = torch.stack(
            [view["camera_intrinsics"] for view in views], dim=1
        ).to(coarse).float()
        coarse_normal = depth_to_normals(coarse, intrinsics)

        # Log depth preserves metric-scale information while compressing its
        # dynamic range.  The event feature remains the dominant channel set.
        log_depth = torch.log(coarse.clamp_min(1e-6)).unsqueeze(2)
        normal_channels = coarse_normal.movedim(-1, 2)
        conditioned_input = torch.cat((feature.float(), log_depth, normal_channels), dim=2)
        flat = conditioned_input.reshape(b * v, hidden + 4, height, width)
        raw = self.conditioned_depth_head(flat)[:, 0].reshape(b, v, height, width)
        bounded = self.depth_update_scale * torch.tanh(raw)

        contribution = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        ).float()
        signed = torch.stack([item["signed_event"] for item in output.ress], dim=1)
        support = signed_support(signed, self.support_dilation_kernel)[:, :, 0]
        contribution_gated = bounded * contribution
        support_gated = contribution_gated * support
        centered = support_center(support_gated, support.bool())

        update_tv = .5 * (
            (centered[..., :, 1:] - centered[..., :, :-1]).abs().mean()
            + (centered[..., 1:, :] - centered[..., :-1, :]).abs().mean()
        )
        update_regularizer = centered.abs().mean() + .5 * update_tv

        for index, (view, item) in enumerate(zip(views, output.ress)):
            final = coarse[:, index] * (1.0 + centered[:, index])
            item["depth_delta_ratio"] = centered[:, index]
            item["depth"] = final.unsqueeze(-1)
            item["normal"] = depth_to_normals(
                final.float(), view["camera_intrinsics"].to(final).float()
            )
            item["depth_pixel_update"] = coarse[:, index] * centered[:, index]
            item["depth_total_update"] = item["depth_pixel_update"]
            item["adapter_update_loss"] = update_regularizer
            item["depth_update_tv"] = update_tv

            item["depth_update_raw_ratio"] = raw[:, index]
            item["depth_update_bounded_ratio"] = bounded[:, index]
            item["depth_update_contribution_ratio"] = contribution_gated[:, index]
            item["depth_update_support_ratio"] = support_gated[:, index]
            item["depth_update_centered_ratio"] = centered[:, index]
            item["depth_update_final_absolute"] = item["depth_pixel_update"]
            item["depth_update_actual_support"] = support[:, index].bool()
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["ConditionedScheduledLinearVoxelModel"]
