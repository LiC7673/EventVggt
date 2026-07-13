"""Linear-voxel pixel model with explicit global depth-scale calibration.

The RGB backbone still predicts the coarse shape.  A single positive scalar
corrects a dataset-level metric-scale mismatch, while the event branch remains
responsible only for the spatially varying residual.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_multiscale_model import (
    LinearVoxelMultiscalePixelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class CalibratedLinearVoxelMultiscalePixelModel(LinearVoxelMultiscalePixelModel):
    checkpoint_schema = "linear_time_voxel_calibrated_pixel_v1"

    def __init__(self, *args, depth_log_scale_limit=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        # exp(0) == 1: loading the RGB checkpoint initially reproduces the
        # uncalibrated model exactly.  The clamp prevents degenerate scales.
        self.depth_log_scale = nn.Parameter(torch.zeros(()))
        self.depth_log_scale_limit = float(depth_log_scale_limit)
        self._calibration_forward_count = 0

    @property
    def metric_depth_scale(self):
        return torch.exp(self.depth_log_scale.clamp(
            -self.depth_log_scale_limit, self.depth_log_scale_limit
        ))

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        scale = self.metric_depth_scale
        local_ratios = []
        total_ratios = []
        support_masks = []

        for view, item in zip(views, output.ress):
            raw_coarse = item["depth_coarse"][..., 0]
            old_local_update = item["depth_pixel_update"]
            local_ratio = torch.where(
                raw_coarse.abs() > 1e-6,
                old_local_update / raw_coarse.clamp_min(1e-6),
                torch.zeros_like(old_local_update),
            )
            calibrated_base = raw_coarse * scale
            final_map = calibrated_base * (1.0 + local_ratio)
            intrinsics = view["camera_intrinsics"].to(final_map).float()

            item["depth"] = final_map.unsqueeze(-1)
            item["normal"] = depth_to_normals(final_map.float(), intrinsics)
            item["depth_calibrated_base"] = calibrated_base.unsqueeze(-1)
            # Keep this panel local: a uniform global scale should not hide the
            # event branch's spatial correction.
            item["depth_pixel_update"] = calibrated_base * local_ratio
            item["depth_total_update"] = final_map - raw_coarse
            item["metric_depth_scale"] = scale

            local_ratios.append(local_ratio.detach().abs())
            total_ratios.append((final_map.detach() / raw_coarse.detach().clamp_min(1e-6) - 1.0).abs())
            support_masks.append(item["event_normal_reliability"].detach() > 0)

        # Training-side utilization audit.  This is deliberately in the new
        # route, so existing trainers/models remain byte-for-byte untouched.
        if self.training:
            self._calibration_forward_count += 1
            if self._calibration_forward_count % 500 == 0 and int(os.environ.get("RANK", "0")) == 0:
                local = torch.cat([x.reshape(-1) for x in local_ratios])
                total = torch.cat([x.reshape(-1) for x in total_ratios])
                support = torch.cat([x.reshape(-1) for x in support_masks]).float()
                print(
                    f"[depth-update@{self._calibration_forward_count:05d}] "
                    f"scale={float(scale.detach()):.5f} "
                    f"local_abs(mean={float(local.mean()):.6f},"
                    f"p95={float(torch.quantile(local, .95)):.6f},max={float(local.max()):.6f}) "
                    f"total_abs(mean={float(total.mean()):.6f},p95={float(torch.quantile(total, .95)):.6f}) "
                    f"support={float(support.mean()):.4f}",
                    flush=True,
                )
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["CalibratedLinearVoxelMultiscalePixelModel"]
