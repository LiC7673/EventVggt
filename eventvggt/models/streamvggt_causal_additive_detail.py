"""Causal additive-event refinement that cannot bypass zero event input.

The additive decomposer predicts a geometry event token from full events and
RGB. The dense depth correction is then multiplied by a smooth support map
computed only from that geometry token. Consequently, an all-zero event voxel
produces an exactly zero depth residual even though the proposal network also
sees RGB and coarse depth context.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from eventvggt.models.streamvggt_additive_decomposition_detail import (
    StreamVGGT as AdditiveDecompositionStreamVGGT,
)
from eventvggt.models.streamvggt_stable_temporal_detail import (
    StableTemporalVoxelDetailRefiner,
)


class CausalEventSupportRefiner(StableTemporalVoxelDetailRefiner):
    def __init__(
        self,
        *args,
        event_support_tau: float = 0.50,
        event_support_dilate_kernel: int = 5,
        event_support_blur_kernel: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.event_support_tau = max(float(event_support_tau), 1e-4)
        self.event_support_dilate_kernel = self._odd_kernel(
            max(1, int(event_support_dilate_kernel))
        )
        self.event_support_blur_kernel = self._odd_kernel(
            max(1, int(event_support_blur_kernel))
        )

    def _event_support(self, event_voxel: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _, height, width = event_voxel.shape
        activity = event_voxel.float().clamp_min(0).sum(dim=2, keepdim=True)
        support = activity / (activity + self.event_support_tau)
        support = support.reshape(batch * seq_len, 1, height, width)
        if self.event_support_dilate_kernel > 1:
            kernel = self.event_support_dilate_kernel
            support = F.max_pool2d(support, kernel_size=kernel, stride=1, padding=kernel // 2)
        if self.event_support_blur_kernel > 1:
            kernel = self.event_support_blur_kernel
            support = F.avg_pool2d(support, kernel_size=kernel, stride=1, padding=kernel // 2)
        return support.clamp(0.0, 1.0)

    def forward(self, *, event_voxel, images, depth, points):
        proposed_depth, _, _ = super().forward(
            event_voxel=event_voxel,
            images=images,
            depth=depth,
            points=points,
        )
        batch, seq_len, height, width, _ = depth.shape
        coarse = depth.permute(0, 1, 4, 2, 3).reshape(batch * seq_len, 1, height, width)
        proposal = proposed_depth.permute(0, 1, 4, 2, 3).reshape(
            batch * seq_len, 1, height, width
        )
        delta_log = torch.log(
            proposal.clamp_min(self.min_depth) / coarse.clamp_min(self.min_depth)
        )
        support = self._event_support(event_voxel).to(device=delta_log.device, dtype=delta_log.dtype)
        delta_log = delta_log * support
        limit = self.residual_abs_limit if self.residual_abs_limit > 0 else self.residual_scale
        if limit > 0:
            delta_log = delta_log.clamp(-limit, limit)
        final_flat = coarse * torch.exp(delta_log)
        final_depth = final_flat.permute(0, 2, 3, 1).reshape(
            batch, seq_len, height, width, 1
        ).to(dtype=depth.dtype)
        depth_residual = final_depth - depth
        final_points = points
        if self.refine_points and points is not None:
            ratio = final_depth / depth.clamp_min(self.min_depth)
            final_points = points * ratio.to(dtype=points.dtype)

        support_sequence = support.reshape(batch, seq_len, height, width)
        self.last_event_support = support_sequence
        self.last_gate = self.last_gate * support_sequence.to(dtype=self.last_gate.dtype)
        return final_depth, final_points, depth_residual


class StreamVGGT(AdditiveDecompositionStreamVGGT):
    def __init__(
        self,
        *args,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        event_hidden_dim: int = 16,
        event_support_tau: float = 0.50,
        event_support_dilate_kernel: int = 5,
        event_support_blur_kernel: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            event_hidden_dim=event_hidden_dim,
            **kwargs,
        )
        self.event_detail_refiner = CausalEventSupportRefiner(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=float(kwargs.get("residual_scale", 0.03)),
            residual_highpass_kernel=int(kwargs.get("residual_highpass_kernel", 0)),
            residual_patch_zero_mean=bool(kwargs.get("residual_patch_zero_mean", False)),
            residual_patch_size=int(kwargs.get("residual_patch_size", 14)),
            residual_abs_limit=float(kwargs.get("residual_abs_limit", 0.0)),
            reliability_gate_enabled=bool(kwargs.get("reliability_gate_enabled", True)),
            reliability_gate_floor=float(kwargs.get("reliability_gate_floor", 0.10)),
            reliability_init_bias=float(kwargs.get("reliability_init_bias", 0.0)),
            refine_points=bool(kwargs.get("refine_points", True)),
            use_checkpoint=bool(kwargs.get("use_checkpoint", True)),
            event_support_tau=event_support_tau,
            event_support_dilate_kernel=event_support_dilate_kernel,
            event_support_blur_kernel=event_support_blur_kernel,
        )

    def forward(self, views, query_points: Optional[torch.Tensor] = None, **kwargs):
        output = super().forward(views, query_points=query_points, **kwargs)
        support = getattr(self.event_detail_refiner, "last_event_support", None)
        if support is not None:
            for view_idx, result in enumerate(output.ress):
                result["event_support"] = support[:, view_idx]
        return output


__all__ = ["StreamVGGT", "CausalEventSupportRefiner"]
