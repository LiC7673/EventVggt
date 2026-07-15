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
    checkpoint_schema = "split_signed_linear_voxel_dual_alignment_hdr_event_normal_derivative_v11"

    def __init__(self, *args, pixel_hidden=32, event_count_cmax=3.0, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        self.event_count_cmax = max(float(event_count_cmax), 1.0)
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

    def _decayed_signed(self, views, split_event):
        """Keep 2B polarity channels and preserve their signed accumulated mass.

        Layout for five bins is [P0..P4, N0..N4]. Positive channels are
        nonnegative and negative channels are nonpositive.  There is no
        thresholding and no P-N collapse, so simultaneous polarities cannot
        cancel before reaching the encoder.
        """
        bins = int(self.voxel_bins)
        if split_event.shape[2] != 2 * bins:
            raise ValueError(
                f"expected {2 * bins} polarity voxel channels, got {split_event.shape[2]}"
            )
        ceiling = self.event_count_cmax
        denominator = torch.log1p(split_event.new_tensor(ceiling, dtype=torch.float32))
        mass = torch.log1p(split_event.float().abs().clamp_max(ceiling)) / denominator
        polarity = mass.new_ones(2 * bins)
        polarity[bins:] = -1.0
        voxel = mass * polarity.view(1, 1, 2 * bins, 1, 1)

        ranges = []
        for view in views:
            value = view.get("event_time_range")
            if not torch.is_tensor(value):
                raise KeyError("event_time_range is required for temporal decay")
            ranges.append(value.to(device=voxel.device, dtype=voxel.dtype))
        time_range = torch.stack(ranges, dim=1)
        t0, current = time_range[..., 0], time_range[..., 1]
        fraction = (
            torch.arange(bins, device=voxel.device, dtype=voxel.dtype) + .5
        ) / bins
        centers = t0.unsqueeze(-1) + (current - t0).unsqueeze(-1) * fraction
        weights = torch.exp(
            -(current.unsqueeze(-1) - centers) / self.event_decay_tau
        ).clamp(0, 1)
        polarity_weights = torch.cat((weights, weights), dim=-1)
        return voxel * polarity_weights.unsqueeze(-1).unsqueeze(-1), weights

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
