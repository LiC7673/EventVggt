"""V10 derivative variant with a strictly pixel-space high-frequency head."""
from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_dual_alignment_hdr_derivative_model import (
    EventNormalDerivativeV10Model,
)


class PixelHighFrequencyDerivativeV10Model(EventNormalDerivativeV10Model):
    checkpoint_schema = "linear_time_voxel_dual_alignment_hdr_pixel_hf_derivative_v11"

    def __init__(self, *args, pixel_hidden=32, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        # No pooling/striding: every output derivative corresponds to one input pixel.
        # The short local residual path prevents the dilated encoder context from
        # erasing thin event edges.
        self.event_normal_decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 6, 1),
        )
        nn.init.zeros_(self.event_normal_decoder[-1].weight)
        nn.init.zeros_(self.event_normal_decoder[-1].bias)

    def _decode_event_normal(self, feature):
        b, v, channels, height, width = feature.shape
        raw = self.event_normal_decoder(feature.reshape(b * v, channels, height, width))
        # A wider bound than V10 avoids clipping strong normal discontinuities.
        derivative = .50 * torch.tanh(raw / .50).reshape(
            b, v, 2, 3, height, width
        ).permute(0, 1, 4, 5, 2, 3).contiguous()
        self._normal_derivative_predictions.append(derivative)
        proxy = derivative.new_zeros(b, v, height, width, 3)
        proxy[..., 2] = 1.0
        return proxy
