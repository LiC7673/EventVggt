"""Decompose full events into geometry/material/noise tokens before refinement.

At inference this model consumes only a full polarity-separated event voxel
and RGB. A dense decomposition head predicts a three-way partition for every
temporal/polarity channel. The predicted geometry token is passed to the
existing temporal-detail refiner; material and noise tokens are auxiliary
training outputs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eventvggt.models.streamvggt_temporal_detail import (
    StreamVGGT as TemporalDetailStreamVGGT,
    _ResidualBlock,
    _group_count,
)


class AdditiveEventTokenDecomposer(nn.Module):
    BRANCH_NAMES = ("geometry", "material", "noise")

    def __init__(
        self,
        *,
        num_bins: int = 10,
        hidden_dim: int = 24,
        count_cmax: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_bins = max(1, int(num_bins))
        self.event_channels = 2 * self.num_bins
        self.count_cmax = max(1.0, float(count_cmax))
        groups = _group_count(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.event_channels + 3, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            _ResidualBlock(hidden_dim, dilation=2),
            _ResidualBlock(hidden_dim, dilation=1),
        )
        self.partition_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3 * self.event_channels, kernel_size=1),
        )
        # Uniform decomposition is a neutral initialization.
        nn.init.zeros_(self.partition_head[-1].weight)
        nn.init.zeros_(self.partition_head[-1].bias)

    def _resample_time(self, voxel: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, height, width = voxel.shape
        source_bins = channels // 2
        if source_bins <= 0:
            raise ValueError("Additive decomposition requires polarity-separated event channels.")
        pos = voxel[:, :, :source_bins]
        neg = voxel[:, :, source_bins : 2 * source_bins]
        if source_bins == self.num_bins:
            return torch.cat([pos, neg], dim=2)

        def resize(value: torch.Tensor) -> torch.Tensor:
            result = F.interpolate(
                value.reshape(batch * seq_len, 1, source_bins, height, width),
                size=(self.num_bins, height, width),
                mode="trilinear",
                align_corners=False,
            )
            result = result * (float(source_bins) / float(self.num_bins))
            return result.reshape(batch, seq_len, self.num_bins, height, width)

        return torch.cat([resize(pos), resize(neg)], dim=2)

    def forward(self, full_voxel: torch.Tensor, rgb: torch.Tensor):
        full_voxel = self._resample_time(full_voxel).clamp_min(0.0)
        batch, seq_len, channels, height, width = full_voxel.shape
        flat_full = full_voxel.reshape(batch * seq_len, channels, height, width)
        flat_rgb = rgb.reshape(batch * seq_len, 3, height, width)
        norm = torch.log1p(flat_full.clamp_max(self.count_cmax)) / torch.log1p(
            flat_full.new_tensor(self.count_cmax)
        )
        features = self.encoder(torch.cat([norm, flat_rgb], dim=1))
        logits = self.partition_head(features)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        logits = logits.view(batch * seq_len, 3, channels, height, width)
        probabilities = torch.softmax(logits, dim=1)
        branch_voxels = probabilities * flat_full.unsqueeze(1)
        branch_voxels = branch_voxels.view(batch, seq_len, 3, channels, height, width)
        probabilities = probabilities.view(batch, seq_len, 3, channels, height, width)
        return branch_voxels, probabilities


class StreamVGGT(TemporalDetailStreamVGGT):
    def __init__(
        self,
        *args,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        decomposition_hidden_dim: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            **kwargs,
        )
        self.event_branch_decomposer = AdditiveEventTokenDecomposer(
            num_bins=event_num_bins,
            hidden_dim=decomposition_hidden_dim,
            count_cmax=event_count_cmax,
        )

    def forward(self, views, query_points: Optional[torch.Tensor] = None, **kwargs):
        if not all("event_voxel" in view for view in views):
            return super().forward(views, query_points=query_points, **kwargs)

        images = torch.stack([view["img"] for view in views], dim=1)
        full_voxel = torch.stack([view["event_voxel"] for view in views], dim=1).to(images.device)
        branch_voxels, branch_probabilities = self.event_branch_decomposer(
            full_voxel.to(dtype=images.dtype), images
        )
        geometry_voxel = branch_voxels[:, :, 0].to(dtype=full_voxel.dtype)

        geometry_views = []
        for view_idx, view in enumerate(views):
            geometry_view = dict(view)
            geometry_view["event_voxel"] = geometry_voxel[:, view_idx]
            geometry_views.append(geometry_view)

        output = super().forward(geometry_views, query_points=query_points, **kwargs)
        for view_idx, result in enumerate(output.ress):
            result["pred_event_geometry_token"] = branch_voxels[:, view_idx, 0]
            result["pred_event_material_token"] = branch_voxels[:, view_idx, 1]
            result["pred_event_noise_token"] = branch_voxels[:, view_idx, 2]
            result["pred_event_branch_probability"] = branch_probabilities[:, view_idx]
        output.views = geometry_views
        return output


__all__ = ["StreamVGGT", "AdditiveEventTokenDecomposer"]

