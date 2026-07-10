"""Reliability-weighted temporal events fused into the unified VGGT decoder.

Unlike the legacy repair model, this model has no coarse-depth residual path.
Stage-1 reliability weights the polarity/time voxel before event encoding;
event patch tokens are then fused with RGB patch tokens and the shared VGGT
heads directly predict camera, depth, and points.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eventvggt.models.streamvggt_paired_token_reliability_detail import FrozenOutputReliabilityGate
from eventvggt.models.streamvggt_temporal_detail import (
    StreamVGGT as RGBStreamVGGT,
    StreamVGGTOutput,
)


class PolarityTemporalPatchEncoder(nn.Module):
    """Keep polarity and temporal adjacency explicit until token pooling."""

    def __init__(self, num_bins: int, token_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        half = max(int(hidden_dim) // 2, 8)
        self.pos_stem = nn.Sequential(
            nn.Conv3d(2, half, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GroupNorm(4, half),
            nn.GELU(),
        )
        self.neg_stem = nn.Sequential(
            nn.Conv3d(2, half, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GroupNorm(4, half),
            nn.GELU(),
        )
        channels = 2 * half
        self.temporal_fuse = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
        )
        self.temporal_score = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
        )
        self.projection = nn.Conv2d(hidden_dim, token_dim, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, voxel: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        batch, seq_len, channels, height, width = voxel.shape
        bins = channels // 2
        if bins != self.num_bins:
            raise ValueError(f"Expected {2 * self.num_bins} event channels, got {channels}")
        pos = torch.log1p(voxel[:, :, :bins].float().clamp_min(0.0))
        neg = torch.log1p(voxel[:, :, bins : 2 * bins].float().clamp_min(0.0))
        time = torch.linspace(-1.0, 1.0, bins, device=voxel.device).view(1, 1, bins, 1, 1)
        pos = torch.stack((pos, pos * time), dim=2).reshape(batch * seq_len, 2, bins, height, width)
        neg = torch.stack((neg, neg * time), dim=2).reshape(batch * seq_len, 2, bins, height, width)
        feature = self.temporal_fuse(torch.cat((self.pos_stem(pos), self.neg_stem(neg)), dim=1))
        attention = torch.softmax(self.temporal_score(feature), dim=2)
        feature_2d = self.spatial((feature * attention).sum(dim=2))
        feature_2d = F.interpolate(feature_2d, size=grid_size, mode="bilinear", align_corners=False)
        tokens = self.projection(feature_2d).flatten(2).transpose(1, 2)
        return tokens.reshape(batch, seq_len, grid_size[0] * grid_size[1], -1)


class StreamVGGT(RGBStreamVGGT):
    def __init__(
        self,
        *args,
        reliability_checkpoint: str,
        reliability_base_channels: int = 32,
        reliability_frame_chunk_size: int = 1,
        reliability_rgb_input_range: str = "minus_one_one",
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        direct_event_hidden_dim: int = 32,
        direct_fusion_scale: float = 0.10,
        **kwargs,
    ) -> None:
        patch_size = int(kwargs.get("patch_size", 14))
        for key in (
            "residual_postfilter_kernel", "residual_postfilter_strength", "causal_output_gate",
            "causal_support_threshold", "causal_support_dilate_kernel", "causal_support_blur_kernel",
        ):
            kwargs.pop(key, None)
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            **kwargs,
        )
        # Reuse only the frozen Stage-1 predictor; its residual forward is never called.
        self.input_reliability = FrozenOutputReliabilityGate(
            nn.Identity(),
            checkpoint=reliability_checkpoint,
            num_bins=event_num_bins,
            base_channels=reliability_base_channels,
            count_cmax=event_count_cmax,
            gate_floor=0.0,
            dilate_kernel=1,
            frame_chunk_size=reliability_frame_chunk_size,
            rgb_input_range=reliability_rgb_input_range,
        )
        token_dim = 2 * int(kwargs.get("embed_dim", 1024))
        self.event_token_encoder = PolarityTemporalPatchEncoder(
            event_num_bins, token_dim, hidden_dim=direct_event_hidden_dim
        )
        self.direct_fusion_scale = float(direct_fusion_scale)
        self.direct_patch_size = patch_size

    def train(self, mode: bool = True):
        super().train(mode)
        self.input_reliability.reliability_net.eval()
        return self

    def forward(self, views, query_points: Optional[torch.Tensor] = None, **_kwargs):
        images = torch.stack([view["img"] for view in views], dim=0).permute(1, 0, 2, 3, 4)
        if images.ndim == 4:
            images = images.unsqueeze(0)
        event_voxel = torch.stack([view["event_voxel"] for view in views], dim=1).to(images.device)
        reliability = self.input_reliability._predict_reliability(event_voxel, images).detach()
        weighted_voxel = event_voxel * reliability.unsqueeze(2).to(event_voxel.dtype)

        tokens_list, patch_start_idx = self.aggregator(images)
        patch_count = tokens_list[-1].shape[2] - patch_start_idx
        grid_h = images.shape[-2] // self.direct_patch_size
        grid_w = images.shape[-1] // self.direct_patch_size
        if grid_h * grid_w != patch_count:
            raise RuntimeError(f"Event/RGB patch mismatch: grid={grid_h}x{grid_w}, tokens={patch_count}")
        event_tokens = self.event_token_encoder(weighted_voxel, (grid_h, grid_w))
        event_presence = (event_voxel.detach().abs().sum(dim=(2, 3, 4)) > 0).to(event_tokens.dtype)
        event_tokens = event_tokens * event_presence.unsqueeze(-1).unsqueeze(-1)
        fused_tokens = []
        for tokens in tokens_list:
            patch = tokens[:, :, patch_start_idx:]
            patch = patch + self.direct_fusion_scale * torch.tanh(event_tokens.to(patch.dtype))
            fused_tokens.append(torch.cat((tokens[:, :, :patch_start_idx], patch), dim=2))

        predictions = {}
        with torch.amp.autocast(device_type="cuda", enabled=False):
            predictions["pose_enc"] = self.camera_head(fused_tokens)[-1]
            predictions["depth"], predictions["depth_conf"] = self.depth_head(
                fused_tokens, images=images, patch_start_idx=patch_start_idx,
                frames_chunk_size=self.head_frames_chunk_size,
            )
            predictions["world_points"], predictions["world_points_conf"] = self.point_head(
                fused_tokens, images=images, patch_start_idx=patch_start_idx,
                frames_chunk_size=self.head_frames_chunk_size,
            )

        support = (event_voxel.detach().abs().sum(dim=2) > 0).to(reliability.dtype)
        ress = []
        for frame_idx in range(images.shape[1]):
            ress.append({
                "pts3d_in_other_view": predictions["world_points"][:, frame_idx],
                "conf": predictions["world_points_conf"][:, frame_idx],
                "depth": predictions["depth"][:, frame_idx],
                "depth_conf": predictions["depth_conf"][:, frame_idx],
                "camera_pose": predictions["pose_enc"][:, frame_idx],
                "event_reliability": reliability[:, frame_idx],
                "event_gate": reliability[:, frame_idx],
                "event_support": support[:, frame_idx],
                **({"valid_mask": views[frame_idx]["valid_mask"]} if "valid_mask" in views[frame_idx] else {}),
            })
        return StreamVGGTOutput(ress=ress, views=views)


__all__ = ["StreamVGGT", "PolarityTemporalPatchEncoder"]
