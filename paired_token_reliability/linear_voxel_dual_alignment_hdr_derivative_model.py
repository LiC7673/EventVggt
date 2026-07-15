"""V10 variant whose event geometry head predicts dN/dx,dN/dy directly."""
from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_dual_alignment_hdr_model import (
    DualAlignmentHDRLinearVoxelModel,
)


class _ZeroNormalGate(nn.Module):
    def forward(self, value):
        return value[:, :1] * 0.0


class EventNormalDerivativeV10Model(DualAlignmentHDRLinearVoxelModel):
    checkpoint_schema = "linear_time_voxel_dual_alignment_hdr_event_normal_derivative_v10"

    def __init__(self, *args, pixel_hidden=32, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        self.event_normal_decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 6, 1),
        )
        nn.init.zeros_(self.event_normal_decoder[-1].weight)
        nn.init.zeros_(self.event_normal_decoder[-1].bias)
        # No absolute event normal is allowed to steer depth in this route.
        self.normal_fusion_gate = _ZeroNormalGate()
        self.disable_normal_depth_refiner = True
        self._normal_derivative_predictions = []

    def _decode_event_normal(self, feature):
        b, v, channels, height, width = feature.shape
        raw = self.event_normal_decoder(
            feature.reshape(b * v, channels, height, width)
        )
        derivative = .25 * torch.tanh(raw / .25).reshape(
            b, v, 2, 3, height, width
        ).permute(0, 1, 4, 5, 2, 3).contiguous()
        self._normal_derivative_predictions.append(derivative)
        # Compatibility-only neutral vector. The zero gate above guarantees
        # that it never modifies HDR base normals or final depth.
        proxy = derivative.new_zeros(b, v, height, width, 3)
        proxy[..., 2] = 1.0
        return proxy

    def forward(self, views, *args, **kwargs):
        self._normal_derivative_predictions = []
        output = super().forward(views, *args, **kwargs)
        # Normal decode order in the common V10 path: aligned full, then geo.
        full = self._normal_derivative_predictions[0]
        geo = self._normal_derivative_predictions[1] if len(self._normal_derivative_predictions) > 1 else None
        for index, item in enumerate(output.ress):
            item["event_normal_derivative"] = full[:, index]
            item["event_normal_derivative_full"] = full[:, index]
            item["event_normal_derivative_geo"] = (
                geo[:, index] if geo is not None else full[:, index].detach()
            )
        return output
