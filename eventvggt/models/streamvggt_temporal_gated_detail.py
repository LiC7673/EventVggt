"""Temporal-event gated dense detail refinement for StreamVGGT.

Events cannot directly write depth residual geometry in this variant.  The
residual proposal is predicted from RGB and coarse depth, while low-pass
temporal event evidence gates where that proposal is allowed to contribute.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from eventvggt.models.streamvggt_temporal_detail import (
    StreamVGGT as DenseTemporalStreamVGGT,
    StreamVGGTOutput,
    _ResidualBlock,
    _group_count,
)


class TemporalEventGatedDetailRefiner(nn.Module):
    """Use events as a reliability gate for RGB-supported depth corrections."""

    def __init__(
        self,
        *,
        num_bins: int = 10,
        hidden_dim: int = 16,
        count_cmax: float = 3.0,
        residual_scale: float = 0.01,
        gate_downsample: int = 4,
        refine_points: bool = True,
        use_checkpoint: bool = True,
        min_depth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_bins = max(1, int(num_bins))
        self.count_cmax = max(1.0, float(count_cmax))
        self.residual_scale = float(residual_scale)
        self.gate_downsample = max(1, int(gate_downsample))
        self.refine_points = bool(refine_points)
        self.use_checkpoint = bool(use_checkpoint)
        self.min_depth = float(min_depth)
        groups = _group_count(hidden_dim)

        # RGB/depth decides residual sign and shape. Events do not enter this path.
        self.geometry_proposal = nn.Sequential(
            nn.Conv2d(7, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            _ResidualBlock(hidden_dim, dilation=2),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

        # Temporal event information is deliberately processed on a coarse grid;
        # bilinear expansion cannot copy thin highlight sweep lines into depth.
        self.event_gate = nn.Sequential(
            nn.Conv2d(2 * self.num_bins, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.geometry_proposal[-1].weight)
        nn.init.zeros_(self.geometry_proposal[-1].bias)
        nn.init.zeros_(self.event_gate[-1].weight)
        nn.init.zeros_(self.event_gate[-1].bias)

    def forward(
        self,
        *,
        event_voxel: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        points: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch, seq_len, _, height, width = event_voxel.shape
        dtype = images.dtype
        voxel = self._prepare_voxels(event_voxel.to(dtype=dtype))

        image_flat = images.reshape(batch * seq_len, 3, height, width).to(dtype=dtype)
        depth_flat = depth.permute(0, 1, 4, 2, 3).reshape(batch * seq_len, 1, height, width)
        log_depth = torch.log(depth_flat.clamp_min(self.min_depth)).to(dtype=dtype)
        mean = log_depth.mean(dim=(-2, -1), keepdim=True)
        std = log_depth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        depth_feature = (log_depth - mean) / std
        depth_dx = F.pad(depth_feature[..., :, 1:] - depth_feature[..., :, :-1], (1, 0, 0, 0))
        depth_dy = F.pad(depth_feature[..., 1:, :] - depth_feature[..., :-1, :], (0, 0, 1, 0))
        depth_hf = depth_feature - F.avg_pool2d(depth_feature, kernel_size=5, stride=1, padding=2)

        proposal_input = torch.cat([image_flat, depth_feature, depth_dx, depth_dy, depth_hf], dim=1)
        if self.use_checkpoint and self.training and proposal_input.requires_grad:
            raw_proposal = checkpoint(self.geometry_proposal, proposal_input, use_reentrant=False)
        else:
            raw_proposal = self.geometry_proposal(proposal_input)

        event_gate, event_presence = self._event_confidence(voxel, output_size=(height, width))
        delta_log = torch.tanh(raw_proposal) * event_gate * self.residual_scale
        refined_flat = depth_flat.to(dtype=delta_log.dtype) * torch.exp(delta_log)
        refined = refined_flat.permute(0, 2, 3, 1).reshape(batch, seq_len, height, width, 1)
        refined = refined.to(dtype=depth.dtype).clamp_min(self.min_depth)
        residual = refined - depth

        refined_points = points
        if self.refine_points and points is not None:
            ratio = refined / depth.clamp_min(self.min_depth)
            refined_points = points * ratio.to(dtype=points.dtype)
        self.last_gate = event_gate.reshape(batch, seq_len, height, width).detach()
        self.last_presence = event_presence.reshape(batch, seq_len, height, width).detach()
        return refined, refined_points, residual

    def _event_confidence(self, voxel: torch.Tensor, *, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = F.avg_pool2d(
            voxel,
            kernel_size=self.gate_downsample,
            stride=self.gate_downsample,
            ceil_mode=True,
        )
        activity = pooled[:, : self.num_bins] + pooled[:, self.num_bins :]
        presence = torch.log1p(activity.sum(dim=1, keepdim=True))
        presence = presence / presence.flatten(2).amax(dim=-1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-6)
        presence = F.avg_pool2d(presence, kernel_size=3, stride=1, padding=1)
        learned_confidence = torch.sigmoid(self.event_gate(pooled))
        gate = presence * (0.25 + 0.75 * learned_confidence)
        gate = F.interpolate(gate, size=output_size, mode="bilinear", align_corners=False)
        presence = F.interpolate(presence, size=output_size, mode="bilinear", align_corners=False)
        return gate, presence

    def _prepare_voxels(self, voxel: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, height, width = voxel.shape
        source_bins = channels // 2
        if source_bins <= 0:
            raise ValueError("Temporal gated detail requires polarity-separated event voxel channels.")
        pos = voxel[:, :, :source_bins].clamp_min(0.0)
        neg = voxel[:, :, source_bins : 2 * source_bins].clamp_min(0.0)
        pos = self._resample_time(pos, self.num_bins)
        neg = self._resample_time(neg, self.num_bins)
        norm = torch.log1p(voxel.new_tensor(self.count_cmax))
        voxel = torch.cat(
            [
                torch.log1p(pos.clamp_max(self.count_cmax)) / norm,
                torch.log1p(neg.clamp_max(self.count_cmax)) / norm,
            ],
            dim=2,
        )
        return voxel.reshape(batch * seq_len, 2 * self.num_bins, height, width)

    @staticmethod
    def _resample_time(voxel: torch.Tensor, target_bins: int) -> torch.Tensor:
        if voxel.shape[2] == target_bins:
            return voxel
        batch, seq_len, bins, height, width = voxel.shape
        result = F.interpolate(
            voxel.reshape(batch * seq_len, 1, bins, height, width),
            size=(target_bins, height, width),
            mode="trilinear",
            align_corners=False,
        )
        result = result * (float(bins) / float(target_bins))
        return result.reshape(batch, seq_len, target_bins, height, width)


class StreamVGGT(DenseTemporalStreamVGGT):
    """Dense event-gated variant; the inherited forward path applies the gate."""

    def __init__(
        self,
        *args,
        event_hidden_dim: int = 16,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        residual_scale: float = 0.01,
        gate_downsample: int = 4,
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
        self.event_detail_refiner = TemporalEventGatedDetailRefiner(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            gate_downsample=gate_downsample,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        gate = getattr(self.event_detail_refiner, "last_gate", None)
        presence = getattr(self.event_detail_refiner, "last_presence", None)
        if gate is not None:
            for frame_idx, result in enumerate(output.ress):
                result["event_gate"] = gate[:, frame_idx]
                result["event_presence"] = presence[:, frame_idx]
        return output


__all__ = ["StreamVGGT", "StreamVGGTOutput", "TemporalEventGatedDetailRefiner"]
