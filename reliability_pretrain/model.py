"""Small U-Net for geometry-event reliability prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReliabilityUNet(nn.Module):
    def __init__(
        self,
        *,
        event_channels: int = 10,
        image_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        in_channels = int(event_channels) + int(image_channels)
        c = int(base_channels)
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)
        self.mid = ConvBlock(4 * c, 4 * c)
        self.dec2 = ConvBlock(6 * c, 2 * c)
        self.dec1 = ConvBlock(3 * c, c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward_logits(self, event_voxel: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([event_voxel, rgb], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, 2, ceil_mode=True))
        mid = self.mid(e3)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)

    def forward(self, event_voxel: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(event_voxel, rgb))
