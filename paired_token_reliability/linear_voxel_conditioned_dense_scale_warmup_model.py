"""Dense event-conditioned depth refinement after metric-scale warmup.

Event support is deliberately *not* an output mask.  Sparse event evidence is
encoded together with calibrated coarse geometry and the correction head may
produce a dense residual.  Event support remains available only for losses and
diagnostics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_conditioned_soft_dc_scale_warmup_model import (
    ScaleWarmupConditionedSoftDCLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class ConditionedDenseScaleWarmupLinearVoxelModel(
    ScaleWarmupConditionedSoftDCLinearVoxelModel
):
    checkpoint_schema = "linear_time_voxel_conditioned_dense_scale_warmup_v1"

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        if not output.ress:
            return output

        warmup = bool(float(output.ress[0]["scale_warmup_active"].detach()))
        if warmup:
            # The parent has already made final==calibrated coarse and removed
            # every event correction during the first scale-only 1k steps.
            return output

        raw = torch.stack(
            [item["depth_update_raw_ratio"] for item in output.ress], dim=1
        ).float()
        contribution = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        ).float()

        # A 50% limit is a ceiling, not a 0.5x small-signal multiplier.
        # d/draw [limit*tanh(raw/limit)] at raw=0 is exactly one.
        limit = max(float(self.depth_update_scale), 1.0e-6)
        bounded = limit * torch.tanh(raw / limit)
        dense_delta = bounded * contribution

        dx = dense_delta[..., :, 1:] - dense_delta[..., :, :-1]
        dy = dense_delta[..., 1:, :] - dense_delta[..., :-1, :]
        tv = 0.5 * (dx.abs().mean() + dy.abs().mean())
        dxx = dx[..., :, 1:] - dx[..., :, :-1]
        dyy = dy[..., 1:, :] - dy[..., :-1, :]
        curvature = 0.5 * (dxx.abs().mean() + dyy.abs().mean())
        # No magnitude penalty: it previously encouraged the dense correction
        # to collapse. Smoothness controls isolated dents without forcing zero.
        dense_regularizer = tv + 0.25 * curvature

        for index, (view, item) in enumerate(zip(views, output.ress)):
            coarse = item["depth_coarse"][..., 0]
            delta = dense_delta[:, index]
            final = coarse * (1.0 + delta)
            item["depth_delta_ratio"] = delta
            item["depth"] = final.unsqueeze(-1)
            item["normal"] = depth_to_normals(
                final.float(), view["camera_intrinsics"].to(final).float()
            )
            item["depth_pixel_update"] = coarse * delta
            item["depth_total_update"] = item["depth_pixel_update"]
            item["adapter_update_loss"] = dense_regularizer
            item["depth_update_tv"] = tv
            item["depth_update_curvature"] = curvature
            item["depth_update_bounded_ratio"] = bounded[:, index]
            item["depth_update_contribution_ratio"] = delta
            # Compatibility fields now carry the same dense tensor. There is
            # intentionally no after-support operation in this model.
            item["depth_update_support_ratio"] = delta
            item["depth_update_detail_ratio"] = delta
            item["depth_update_centered_ratio"] = delta
            item["depth_update_final_absolute"] = item["depth_pixel_update"]
            item["depth_update_is_dense"] = final.new_tensor(1.0)

        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["ConditionedDenseScaleWarmupLinearVoxelModel"]
