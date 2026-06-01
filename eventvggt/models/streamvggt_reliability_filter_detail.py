"""Dense temporal detail refinement with reliability-filtered event features."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eventvggt.models.streamvggt_temporal_detail import (
    StreamVGGT as TemporalDetailStreamVGGT,
    StreamVGGTOutput,
    TemporalVoxelDetailRefiner,
    _ResidualBlock,
    _group_count,
)


class ReliabilityFilteredTemporalVoxelDetailRefiner(TemporalVoxelDetailRefiner):
    """Filter event voxels by learned geometry reliability before refinement."""

    def __init__(
        self,
        *,
        num_bins: int = 10,
        hidden_dim: int = 16,
        count_cmax: float = 3.0,
        residual_scale: float = 0.03,
        gate_downsample: int = 2,
        reliability_floor: float = 0.30,
        reliability_init_bias: float = 0.5,
        refine_points: bool = True,
        use_checkpoint: bool = True,
        min_depth: float = 1e-6,
    ) -> None:
        super().__init__(
            num_bins=num_bins,
            hidden_dim=hidden_dim,
            count_cmax=count_cmax,
            residual_scale=residual_scale,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
            min_depth=min_depth,
        )
        self.gate_downsample = max(1, int(gate_downsample))
        self.reliability_floor = min(max(float(reliability_floor), 0.0), 1.0)
        self.num_temporal_stats = 8
        groups = _group_count(hidden_dim)
        self.temporal_reliability = nn.Sequential(
            nn.Conv2d(2 * self.num_bins + self.num_temporal_stats, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            _ResidualBlock(hidden_dim, dilation=2),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.temporal_reliability[-1].weight)
        nn.init.constant_(self.temporal_reliability[-1].bias, float(reliability_init_bias))

    def _prepare_voxels(self, voxel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, channels, height, width = voxel.shape
        source_bins = channels // 2
        if source_bins <= 0:
            raise ValueError("Reliability-filtered detail requires polarity-separated event voxel channels.")

        pos = voxel[:, :, :source_bins].clamp_min(0.0)
        neg = voxel[:, :, source_bins : 2 * source_bins].clamp_min(0.0)
        pos = self._resample_time(pos, self.num_bins)
        neg = self._resample_time(neg, self.num_bins)
        norm = torch.log1p(voxel.new_tensor(self.count_cmax))
        pos = torch.log1p(pos.clamp_max(self.count_cmax)) / norm
        neg = torch.log1p(neg.clamp_max(self.count_cmax)) / norm

        pos_flat = pos.reshape(batch * seq_len, self.num_bins, height, width)
        neg_flat = neg.reshape(batch * seq_len, self.num_bins, height, width)
        raw_voxels = torch.cat([pos_flat, neg_flat], dim=1)
        reliability, presence, temporal_quality = self._event_reliability(raw_voxels, output_size=(height, width))
        filter_map = self.reliability_floor + (1.0 - self.reliability_floor) * reliability

        filtered_pos = pos_flat * filter_map
        filtered_neg = neg_flat * filter_map
        voxels = torch.cat([filtered_pos, filtered_neg], dim=1)
        activity = filtered_pos + filtered_neg
        time = torch.linspace(-1.0, 1.0, self.num_bins, device=voxel.device, dtype=voxel.dtype).view(
            1, self.num_bins, 1, 1
        )
        summary = torch.cat(
            [
                filtered_pos.mean(dim=1, keepdim=True),
                filtered_neg.mean(dim=1, keepdim=True),
                activity.amax(dim=1, keepdim=True),
                ((filtered_pos - filtered_neg) * time).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )

        self.last_reliability = reliability.reshape(batch, seq_len, height, width)
        self.last_presence = presence.reshape(batch, seq_len, height, width).detach()
        self.last_temporal_quality = temporal_quality.reshape(batch, seq_len, height, width).detach()
        self.last_filter = filter_map.reshape(batch, seq_len, height, width).detach()
        return voxels, summary

    def _event_reliability(self, voxels: torch.Tensor, *, output_size):
        pooled = F.avg_pool2d(
            voxels,
            kernel_size=self.gate_downsample,
            stride=self.gate_downsample,
            ceil_mode=True,
        )
        pos = pooled[:, : self.num_bins]
        neg = pooled[:, self.num_bins :]
        activity = pos + neg
        total = activity.sum(dim=1, keepdim=True)
        active = (total > 1e-6).to(dtype=voxels.dtype)

        presence = torch.log1p(total)
        max_presence = presence.flatten(2).amax(dim=-1, keepdim=True).view(-1, 1, 1, 1)
        presence = presence / max_presence.clamp_min(1e-6)
        presence = F.avg_pool2d(presence, kernel_size=3, stride=1, padding=1)

        probability = activity / total.clamp_min(1e-6)
        entropy = -(probability * torch.log(probability.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        entropy = entropy / max(math.log(float(max(self.num_bins, 2))), 1e-6)
        entropy = entropy * active
        peak = activity.amax(dim=1, keepdim=True) / total.clamp_min(1e-6)
        peak = peak * active
        persistence = activity / activity.amax(dim=1, keepdim=True).clamp_min(1e-6)
        persistence = persistence.mean(dim=1, keepdim=True) * active

        pos_sum = pos.sum(dim=1, keepdim=True)
        neg_sum = neg.sum(dim=1, keepdim=True)
        polarity_focus = (pos_sum - neg_sum).abs() / total.clamp_min(1e-6)
        polarity_focus = polarity_focus.clamp(0.0, 1.0) * active
        polarity_mix = (1.0 - polarity_focus).clamp(0.0, 1.0) * active

        time = torch.linspace(-1.0, 1.0, self.num_bins, device=voxels.device, dtype=voxels.dtype).view(
            1, self.num_bins, 1, 1
        )
        time_center = (probability * time).sum(dim=1, keepdim=True) * active
        time_spread = torch.sqrt(
            (probability * (time - time_center).square()).sum(dim=1, keepdim=True).clamp_min(1e-8)
        ) * active
        signed_direction = ((pos - neg) * time).sum(dim=1, keepdim=True) / total.clamp_min(1e-6)
        signed_direction = signed_direction * active
        temporal_focus = (1.0 - entropy).clamp(0.0, 1.0)
        temporal_quality = temporal_focus * (0.25 + 0.75 * peak) * (0.25 + 0.75 * polarity_focus)
        temporal_quality = temporal_quality.clamp(0.0, 1.0) * active

        stats = torch.cat(
            [
                presence,
                persistence,
                polarity_mix,
                entropy,
                peak,
                time_spread,
                signed_direction,
                temporal_quality,
            ],
            dim=1,
        )
        reliability = torch.sigmoid(self.temporal_reliability(torch.cat([pooled, stats], dim=1)))
        reliability = F.interpolate(reliability, size=output_size, mode="bilinear", align_corners=False)
        presence = F.interpolate(presence, size=output_size, mode="bilinear", align_corners=False)
        temporal_quality = F.interpolate(temporal_quality, size=output_size, mode="bilinear", align_corners=False)
        return reliability, presence, temporal_quality


class StreamVGGT(TemporalDetailStreamVGGT):
    """Temporal-detail refiner whose event features are reliability-filtered."""

    def __init__(
        self,
        *args,
        event_hidden_dim: int = 16,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        residual_scale: float = 0.03,
        gate_downsample: int = 2,
        event_reliability_floor: float = 0.30,
        event_reliability_init_bias: float = 0.5,
        refine_points: bool = True,
        use_checkpoint: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            event_hidden_dim=event_hidden_dim,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.event_detail_refiner = ReliabilityFilteredTemporalVoxelDetailRefiner(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            gate_downsample=gate_downsample,
            reliability_floor=event_reliability_floor,
            reliability_init_bias=event_reliability_init_bias,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        reliability = getattr(self.event_detail_refiner, "last_reliability", None)
        presence = getattr(self.event_detail_refiner, "last_presence", None)
        temporal_quality = getattr(self.event_detail_refiner, "last_temporal_quality", None)
        event_filter = getattr(self.event_detail_refiner, "last_filter", None)
        if reliability is not None:
            for frame_idx, result in enumerate(output.ress):
                result["event_reliability"] = reliability[:, frame_idx]
                result["event_presence"] = presence[:, frame_idx]
                result["event_temporal_quality"] = temporal_quality[:, frame_idx]
                result["event_gate"] = event_filter[:, frame_idx]
        return output


__all__ = ["StreamVGGT", "StreamVGGTOutput", "ReliabilityFilteredTemporalVoxelDetailRefiner"]
