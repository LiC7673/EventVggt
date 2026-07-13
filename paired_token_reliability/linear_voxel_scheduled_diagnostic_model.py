"""Scheduled experiment model exposing every event-depth update stage."""
from __future__ import annotations

import torch

from paired_token_reliability.linear_voxel_detail_normal_derivative_model import (
    DetailNormalDerivativeLinearVoxelModel,
)
from paired_token_reliability.signed_multiscale_model import signed_support
from stage2_geometry_adapter.model import GeometryAdapterOutput


class ScheduledDiagnosticLinearVoxelModel(DetailNormalDerivativeLinearVoxelModel):
    checkpoint_schema = "linear_time_voxel_scheduled_diagnostic_v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._captured_raw_depth_update = None
        self._raw_update_hook = self.depth_local_head.register_forward_hook(
            self._capture_raw_depth_update
        )

    def _capture_raw_depth_update(self, _module, _inputs, output):
        self._captured_raw_depth_update = output[:, 0]

    def forward(self, views, *args, **kwargs):
        self._captured_raw_depth_update = None
        output = super().forward(views, *args, **kwargs)
        if self._captured_raw_depth_update is None:
            raise RuntimeError("depth_local_head did not produce a captured raw update")

        b = output.ress[0]["depth"].shape[0]
        v = len(output.ress)
        height, width = output.ress[0]["depth"].shape[1:3]
        raw = self._captured_raw_depth_update.reshape(b, v, height, width)
        bounded = self.depth_update_scale * torch.tanh(raw)
        contribution = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        )
        signed = torch.stack([item["signed_event"] for item in output.ress], dim=1)
        support = signed_support(signed, self.support_dilation_kernel)[:, :, 0]
        contribution_gated = bounded * contribution
        support_gated = contribution_gated * support

        for index, item in enumerate(output.ress):
            centered = item["depth_delta_ratio"]
            coarse = item["depth_coarse"][..., 0]
            item["depth_update_raw_ratio"] = raw[:, index]
            item["depth_update_bounded_ratio"] = bounded[:, index]
            item["depth_update_contribution_ratio"] = contribution_gated[:, index]
            item["depth_update_support_ratio"] = support_gated[:, index]
            item["depth_update_centered_ratio"] = centered
            item["depth_update_final_absolute"] = coarse * centered
            item["depth_update_actual_support"] = support[:, index].bool()
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["ScheduledDiagnosticLinearVoxelModel"]
