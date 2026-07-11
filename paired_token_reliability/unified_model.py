"""Final deployable RGB+event geometry model for the unified A/B/C pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.contribution_stage1 import ContributionNet, normalize_event_voxel
from stage2_geometry_adapter.model import StreamVGGT


class TemporalContributionNet(nn.Module):
    """Spatial geometry prior plus a polarity/bin-specific contribution head."""

    def __init__(self, num_bins, base_channels, coarse_feature_dim, count_cmax, initial_contribution):
        super().__init__()
        self.num_bins = int(num_bins)
        self.coarse_feature_dim = int(coarse_feature_dim)
        self.count_cmax = float(count_cmax)
        self.spatial = ContributionNet(
            num_bins=num_bins,
            base_channels=base_channels,
            coarse_feature_dim=coarse_feature_dim,
            count_cmax=count_cmax,
            initial_contribution=initial_contribution,
        )
        channels = 2 * self.num_bins
        self.temporal_delta = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        nn.init.zeros_(self.temporal_delta.weight)
        nn.init.zeros_(self.temporal_delta.bias)

    def forward(self, event_voxel, image, coarse_depth, coarse_normals, coarse_features=None):
        spatial = self.spatial(
            event_voxel, image, coarse_depth, coarse_normals, coarse_features
        )
        batch, views, channels, height, width = event_voxel.shape
        event = normalize_event_voxel(event_voxel, self.count_cmax).reshape(
            batch * views, channels, height, width
        )
        delta = self.temporal_delta(event).reshape(batch, views, channels, height, width)
        eps = 1.0e-5
        spatial_logit = torch.logit(spatial.clamp(eps, 1.0 - eps)).unsqueeze(2)
        return torch.sigmoid(spatial_logit + delta)


def contribution_override(event_voxel, mode="learned", min_keep=0.5, max_keep=1.0):
    """Build explicit full/random/none ablations; learned mode returns None."""
    if mode == "learned":
        return None
    if mode in {"full", "no_contribution"}:
        return torch.ones_like(event_voxel)
    if mode in {"none", "zero"}:
        return torch.zeros_like(event_voxel)
    if mode == "random":
        low, high = sorted((float(min_keep), float(max_keep)))
        keep = torch.empty(
            *event_voxel.shape[:2], 1, 1, 1,
            device=event_voxel.device,
            dtype=event_voxel.dtype,
        ).uniform_(max(0.0, low), min(1.0, high))
        active = event_voxel.ne(0)
        sampled = torch.rand_like(event_voxel) < keep
        return torch.where(active, sampled, torch.ones_like(sampled)).to(event_voxel.dtype)
    raise ValueError(f"Unknown contribution mode: {mode}")


class UnifiedGeometryContributionModel(StreamVGGT):
    # v2 predicts events on the patch grid and bilinearly resizes the residual,
    # bypassing the DPT ConvTranspose polyphase path used by v1 updates.
    checkpoint_schema = "unified_geometry_contribution_v2"

    def __init__(self, *args, **kwargs):
        # No legacy Stage-1 checkpoint: ContributionNet and final adapters are
        # initialized and trained inside the same deployable model.
        kwargs.pop("stage1_checkpoint", None)
        kwargs.pop("reliability_checkpoint", None)
        super().__init__(*args, stage1_checkpoint=None, **kwargs)
        self.contribution_net = TemporalContributionNet(
            num_bins=self.event_encoder.num_bins,
            base_channels=int(kwargs.get("contribution_channels", 32)),
            coarse_feature_dim=self.contribution_net.coarse_feature_dim,
            count_cmax=self.event_encoder.count_cmax,
            initial_contribution=float(kwargs.get("contribution_initial_value", 1.0)),
        )


__all__ = [
    "TemporalContributionNet",
    "UnifiedGeometryContributionModel",
    "contribution_override",
]
