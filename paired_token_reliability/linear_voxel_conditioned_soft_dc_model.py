"""Conditioned event geometry with soft, bounded DC preservation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_conditioned_scheduled_model import (
    ConditionedScheduledLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class ConditionedSoftDCLinearVoxelModel(ConditionedScheduledLinearVoxelModel):
    checkpoint_schema = "linear_time_voxel_conditioned_soft_dc_v1"

    def __init__(self, *args, event_dc_limit=.03,
                 event_residual_target_limit=.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_dc_limit = float(event_dc_limit)
        self.event_residual_target_limit = float(event_residual_target_limit)

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        support_gated = torch.stack(
            [item["depth_update_support_ratio"] for item in output.ress], dim=1
        )
        support = torch.stack(
            [item["depth_update_actual_support"] for item in output.ress], dim=1
        ).to(support_gated)
        count = support.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        dc_raw = (support_gated * support).sum(dim=(-2, -1), keepdim=True) / count
        detail = (support_gated - dc_raw) * support
        dc = dc_raw.clamp(-self.event_dc_limit, self.event_dc_limit)
        delta = detail + dc * support

        detail_tv = .5 * (
            (detail[..., :, 1:] - detail[..., :, :-1]).abs().mean()
            + (detail[..., 1:, :] - detail[..., :-1, :]).abs().mean()
        )
        # Magnitude/TV regularization protects local geometry from noise but
        # deliberately does not suppress the allowed per-view DC correction.
        detail_regularizer = detail.abs().mean() + .5 * detail_tv
        dc_excess = F.relu(dc_raw.abs() - self.event_dc_limit).square().mean()

        for index, (view, item) in enumerate(zip(views, output.ress)):
            coarse = item["depth_coarse"][..., 0]
            final = coarse * (1.0 + delta[:, index])
            item["depth_delta_ratio"] = delta[:, index]
            item["depth"] = final.unsqueeze(-1)
            item["normal"] = depth_to_normals(
                final.float(), view["camera_intrinsics"].to(final).float()
            )
            item["depth_pixel_update"] = coarse * delta[:, index]
            item["depth_total_update"] = item["depth_pixel_update"]
            item["adapter_update_loss"] = detail_regularizer
            item["depth_update_tv"] = detail_tv
            item["depth_update_detail_ratio"] = detail[:, index]
            item["depth_update_dc_raw_ratio"] = dc_raw[:, index, 0, 0]
            item["depth_update_dc_ratio"] = dc[:, index, 0, 0]
            item["depth_update_dc_excess_loss"] = dc_excess
            item["depth_update_dc_limit"] = final.new_tensor(self.event_dc_limit)
            item["depth_update_target_limit"] = final.new_tensor(
                self.event_residual_target_limit
            )
            item["depth_update_centered_ratio"] = delta[:, index]
            item["depth_update_final_absolute"] = item["depth_pixel_update"]
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["ConditionedSoftDCLinearVoxelModel"]
