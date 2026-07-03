"""Temporal event-detail VGGT with a hard causal event-support constraint.

This module is intentionally reliability-free. It is the common event branch
used by the clean paper ablation before the learned ReliabilityNet is enabled.
Zero events force a zero depth residual, so the branch cannot improve geometry
through its RGB/depth context alone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from eventvggt.models.streamvggt_temporal_detail import (
    StreamVGGT as TemporalDetailStreamVGGT,
    StreamVGGTOutput,
    TemporalVoxelDetailRefiner,
)


class CausalTemporalVoxelDetailRefiner(TemporalVoxelDetailRefiner):
    def __init__(
        self,
        *args,
        support_threshold: float = 0.01,
        support_dilate_kernel: int = 5,
        support_blur_kernel: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.support_threshold = max(float(support_threshold), 0.0)
        self.support_dilate_kernel = max(int(support_dilate_kernel), 1)
        self.support_blur_kernel = max(int(support_blur_kernel), 1)
        if self.support_dilate_kernel % 2 == 0:
            self.support_dilate_kernel += 1
        if self.support_blur_kernel % 2 == 0:
            self.support_blur_kernel += 1
        self.last_event_support: Optional[torch.Tensor] = None
        self.last_delta_log: Optional[torch.Tensor] = None

    def _event_support(self, event_voxel: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _, height, width = event_voxel.shape
        activity = event_voxel.detach().float().abs().sum(dim=2, keepdim=True)
        peak = activity.flatten(3).amax(dim=-1, keepdim=True).view(batch, seq_len, 1, 1, 1)
        normalized = activity / peak.clamp_min(1.0e-6)
        support = (normalized >= self.support_threshold).to(activity.dtype)
        support = support.reshape(batch * seq_len, 1, height, width)
        if self.support_dilate_kernel > 1:
            kernel = self.support_dilate_kernel
            support = F.max_pool2d(support, kernel, stride=1, padding=kernel // 2)
        if self.support_blur_kernel > 1:
            kernel = self.support_blur_kernel
            support = F.avg_pool2d(support, kernel, stride=1, padding=kernel // 2)
        return support.clamp(0.0, 1.0).reshape(batch, seq_len, height, width)

    def forward(
        self,
        *,
        event_voxel: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        points: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        refined_depth, _, _ = super().forward(
            event_voxel=event_voxel,
            images=images,
            depth=depth,
            points=points,
        )
        support = self._event_support(event_voxel)
        delta_log = torch.log(
            refined_depth.float().clamp_min(self.min_depth)
            / depth.float().clamp_min(self.min_depth)
        ).squeeze(-1)
        delta_log = delta_log * support.to(delta_log.dtype)
        refined_depth = depth.float() * torch.exp(delta_log.unsqueeze(-1))
        refined_depth = refined_depth.to(depth.dtype).clamp_min(self.min_depth)
        depth_residual = refined_depth - depth
        refined_points = points
        if self.refine_points and points is not None:
            ratio = refined_depth / depth.clamp_min(self.min_depth)
            refined_points = points * ratio.to(points.dtype)

        self.last_event_support = support
        self.last_delta_log = delta_log
        if self.last_gate is not None:
            self.last_gate = self.last_gate * support.to(self.last_gate.dtype)
        return refined_depth, refined_points, depth_residual


class StreamVGGT(TemporalDetailStreamVGGT):
    def __init__(
        self,
        *args,
        event_num_bins: int = 10,
        event_hidden_dim: int = 16,
        event_count_cmax: float = 3.0,
        residual_scale: float = 0.035,
        residual_highpass_kernel: int = 9,
        residual_patch_zero_mean: bool = True,
        residual_patch_size: int = 14,
        residual_abs_limit: float = 0.025,
        refine_points: bool = True,
        use_checkpoint: bool = True,
        support_threshold: float = 0.01,
        support_dilate_kernel: int = 5,
        support_blur_kernel: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_hidden_dim=event_hidden_dim,
            event_count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            residual_highpass_kernel=residual_highpass_kernel,
            residual_patch_zero_mean=residual_patch_zero_mean,
            residual_patch_size=residual_patch_size,
            residual_abs_limit=residual_abs_limit,
            reliability_gate_enabled=False,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.event_detail_refiner = CausalTemporalVoxelDetailRefiner(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            residual_highpass_kernel=residual_highpass_kernel,
            residual_patch_zero_mean=residual_patch_zero_mean,
            residual_patch_size=residual_patch_size,
            residual_abs_limit=residual_abs_limit,
            reliability_gate_enabled=False,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
            support_threshold=support_threshold,
            support_dilate_kernel=support_dilate_kernel,
            support_blur_kernel=support_blur_kernel,
        )

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        support = self.event_detail_refiner.last_event_support
        delta_log = self.event_detail_refiner.last_delta_log
        if support is not None and delta_log is not None:
            for frame_idx, result in enumerate(output.ress):
                result["event_support"] = support[:, frame_idx]
                result["event_delta_log"] = delta_log[:, frame_idx]
        return output


__all__ = [
    "StreamVGGT",
    "StreamVGGTOutput",
    "CausalTemporalVoxelDetailRefiner",
]
