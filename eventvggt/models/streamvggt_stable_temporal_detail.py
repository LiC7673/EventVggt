"""Temporal-detail StreamVGGT with edge-aware residual stabilization.

This module keeps the existing temporal-detail architecture and parameter
names, but suppresses isolated event-count noise before depth derivatives turn
it into granular normal artifacts. Existing model files remain untouched.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from eventvggt.models.streamvggt_temporal_detail import (
    StreamVGGT as BaseTemporalDetailStreamVGGT,
    TemporalVoxelDetailRefiner,
)


class StableTemporalVoxelDetailRefiner(TemporalVoxelDetailRefiner):
    def __init__(
        self,
        *args,
        reliability_smooth_kernel: int = 3,
        reliability_smooth_strength: float = 0.50,
        residual_denoise_kernel: int = 3,
        residual_denoise_strength: float = 0.70,
        depth_edge_alpha: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reliability_smooth_kernel = max(1, int(reliability_smooth_kernel))
        self.reliability_smooth_strength = min(max(float(reliability_smooth_strength), 0.0), 1.0)
        self.residual_denoise_kernel = max(1, int(residual_denoise_kernel))
        self.residual_denoise_strength = min(max(float(residual_denoise_strength), 0.0), 1.0)
        self.depth_edge_alpha = max(float(depth_edge_alpha), 0.0)

    @staticmethod
    def _odd_kernel(kernel: int) -> int:
        return kernel if kernel % 2 == 1 else kernel + 1

    def forward(self, *, event_voxel, images, depth, points):
        refined_depth, _, _ = super().forward(
            event_voxel=event_voxel,
            images=images,
            depth=depth,
            points=points,
        )
        batch, seq_len, height, width, _ = depth.shape
        depth_flat = depth.permute(0, 1, 4, 2, 3).reshape(batch * seq_len, 1, height, width)
        refined_flat = refined_depth.permute(0, 1, 4, 2, 3).reshape(
            batch * seq_len, 1, height, width
        )
        delta_log = torch.log(
            refined_flat.clamp_min(self.min_depth) / depth_flat.clamp_min(self.min_depth)
        )

        reliability = self.last_reliability.reshape(batch * seq_len, 1, height, width)
        if self.reliability_smooth_kernel > 1 and self.reliability_smooth_strength > 0:
            kernel = self._odd_kernel(self.reliability_smooth_kernel)
            smooth_reliability = F.avg_pool2d(
                reliability, kernel_size=kernel, stride=1, padding=kernel // 2
            )
            reliability = torch.lerp(
                reliability, smooth_reliability, self.reliability_smooth_strength
            ).clamp(0.0, 1.0)
            old_gate = self.last_gate.reshape(batch * seq_len, 1, height, width).clamp_min(0.05)
            new_gate = self.reliability_gate_floor + (
                1.0 - self.reliability_gate_floor
            ) * reliability
            # Preserve the learned residual proposal while replacing its noisy
            # pointwise gate with the spatially consistent gate.
            delta_log = delta_log / old_gate * new_gate
            self.last_reliability = reliability.reshape(batch, seq_len, height, width)
            self.last_gate = new_gate.reshape(batch, seq_len, height, width)

        if self.residual_denoise_kernel > 1 and self.residual_denoise_strength > 0:
            kernel = self._odd_kernel(self.residual_denoise_kernel)
            smooth_delta = F.avg_pool2d(
                delta_log, kernel_size=kernel, stride=1, padding=kernel // 2
            )
            log_depth = torch.log(depth_flat.clamp_min(self.min_depth))
            local_depth = F.avg_pool2d(
                log_depth, kernel_size=kernel, stride=1, padding=kernel // 2
            )
            # Smooth strongly on locally flat surfaces and preserve depth
            # discontinuities where averaging would create edge halos.
            flat_weight = torch.exp(
                -self.depth_edge_alpha * (log_depth - local_depth).abs()
            ).clamp(0.0, 1.0)
            blend = self.residual_denoise_strength * flat_weight
            delta_log = delta_log * (1.0 - blend) + smooth_delta * blend

        limit = self.residual_abs_limit if self.residual_abs_limit > 0 else self.residual_scale
        if limit > 0:
            delta_log = delta_log.clamp(-limit, limit)
        final_flat = depth_flat * torch.exp(delta_log)
        final_depth = final_flat.permute(0, 2, 3, 1).reshape(
            batch, seq_len, height, width, 1
        ).to(dtype=depth.dtype)
        depth_residual = final_depth - depth
        final_points = points
        if self.refine_points and points is not None:
            ratio = final_depth / depth.clamp_min(self.min_depth)
            final_points = points * ratio.to(dtype=points.dtype)
        return final_depth, final_points, depth_residual


class StreamVGGT(BaseTemporalDetailStreamVGGT):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        event_hidden_dim: int = 16,
        head_frames_chunk_size: int = 8,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        residual_scale: float = 0.03,
        residual_highpass_kernel: int = 0,
        residual_patch_zero_mean: bool = False,
        residual_patch_size: int = 14,
        residual_abs_limit: float = 0.0,
        reliability_gate_enabled: bool = False,
        reliability_gate_floor: float = 0.10,
        reliability_init_bias: float = 0.0,
        refine_points: bool = True,
        use_checkpoint: bool = True,
        reliability_smooth_kernel: int = 3,
        reliability_smooth_strength: float = 0.50,
        residual_denoise_kernel: int = 3,
        residual_denoise_strength: float = 0.70,
        depth_edge_alpha: float = 30.0,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            event_hidden_dim=event_hidden_dim,
            head_frames_chunk_size=head_frames_chunk_size,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            residual_highpass_kernel=residual_highpass_kernel,
            residual_patch_zero_mean=residual_patch_zero_mean,
            residual_patch_size=residual_patch_size,
            residual_abs_limit=residual_abs_limit,
            reliability_gate_enabled=reliability_gate_enabled,
            reliability_gate_floor=reliability_gate_floor,
            reliability_init_bias=reliability_init_bias,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
        )
        # Same learnable submodule structure and names as the base refiner, so
        # full_img_reliability checkpoints remain load-compatible.
        self.event_detail_refiner = StableTemporalVoxelDetailRefiner(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            residual_highpass_kernel=residual_highpass_kernel,
            residual_patch_zero_mean=residual_patch_zero_mean,
            residual_patch_size=residual_patch_size,
            residual_abs_limit=residual_abs_limit,
            reliability_gate_enabled=reliability_gate_enabled,
            reliability_gate_floor=reliability_gate_floor,
            reliability_init_bias=reliability_init_bias,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
            reliability_smooth_kernel=reliability_smooth_kernel,
            reliability_smooth_strength=reliability_smooth_strength,
            residual_denoise_kernel=residual_denoise_kernel,
            residual_denoise_strength=residual_denoise_strength,
            depth_edge_alpha=depth_edge_alpha,
        )


__all__ = ["StreamVGGT", "StableTemporalVoxelDetailRefiner"]

