"""Calibrated RGB depth plus explicitly supervised event detail residual."""
from __future__ import annotations

import torch

from paired_token_reliability.linear_voxel_calibrated_model import (
    CalibratedLinearVoxelMultiscalePixelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


def support_center(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Remove the per-view DC component without touching unsupported pixels."""
    weight = support.to(value)
    mean = (value * weight).sum(dim=(-2, -1), keepdim=True) / weight.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1.0)
    return (value - mean) * weight


class DetailResidualLinearVoxelModel(CalibratedLinearVoxelMultiscalePixelModel):
    checkpoint_schema = "linear_time_voxel_detail_residual_v1"

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        for view, item in zip(views, output.ress):
            coarse = item["depth_coarse"][..., 0]
            ratio = torch.where(
                coarse.abs() > 1e-6,
                item["depth_pixel_update"] / coarse.clamp_min(1e-6),
                torch.zeros_like(coarse),
            )
            support = item["event_normal_reliability"] > 0
            detail_ratio = support_center(ratio, support)
            final = coarse * (1.0 + detail_ratio)
            intrinsics = view["camera_intrinsics"].to(final).float()

            item["depth_delta_ratio"] = detail_ratio
            item["depth"] = final.unsqueeze(-1)
            item["normal"] = depth_to_normals(final.float(), intrinsics)
            item["depth_pixel_update"] = coarse * detail_ratio
            item["depth_total_update"] = item["depth_pixel_update"]
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["DetailResidualLinearVoxelModel", "support_center"]
