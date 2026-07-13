"""Pixel multi-scale model consuming polarity-separated linear-time voxels."""
from __future__ import annotations
import torch
import torch.nn as nn
from paired_token_reliability.signed_multiscale_model import (
    SignedContributionNet, SignedMultiScaleEncoder, SignedMultiscalePixelModel,
)


class _ForcedFullContribution(nn.Module):
    """C=1 ablation while retaining a zero-gradient DDP dependency."""
    def __init__(self, learned):
        super().__init__(); self.learned=learned
        self.coarse_feature_dim=getattr(learned,"coarse_feature_dim",0)
    def forward(self,*args,**kwargs):
        predicted=self.learned(*args,**kwargs)
        return torch.ones_like(predicted)+0.0*predicted


class LinearVoxelMultiscalePixelModel(SignedMultiscalePixelModel):
    """Use V(x,y,b,p) directly; never collapse positive/negative polarity."""
    checkpoint_schema = "linear_time_voxel_multiscale_pixel_v1"

    def __init__(self, *args, voxel_bins=5, pixel_hidden=32,
                 force_full_contribution=False, **kwargs):
        super().__init__(*args, signed_event_bins=voxel_bins, pixel_hidden=pixel_hidden, **kwargs)
        self.voxel_bins = int(voxel_bins)
        channels = 2 * self.voxel_bins
        self.contribution_net = SignedContributionNet(channels, pixel_hidden,
                                                       kwargs.get("contribution_initial_value", .95))
        self.event_encoder = SignedMultiScaleEncoder(channels, pixel_hidden)
        self.force_full_contribution=bool(force_full_contribution)
        if self.force_full_contribution:
            self.contribution_net=_ForcedFullContribution(self.contribution_net)

    def _decayed_signed(self, views, split_event):
        if split_event.shape[2] != 2 * self.voxel_bins:
            raise ValueError(f"expected {2*self.voxel_bins} voxel channels, got {split_event.shape[2]}")
        # Linear-time voxel values are nonnegative polarity-separated masses.
        # log1p limits hot pixels without destroying temporal interpolation.
        voxel = torch.log1p(split_event.float().clamp_min(0))
        ranges = []
        for view in views:
            value = view.get("event_time_range")
            if not torch.is_tensor(value):
                raise KeyError("event_time_range is required for temporal decay")
            ranges.append(value.to(device=voxel.device, dtype=voxel.dtype))
        time_range = torch.stack(ranges, dim=1)
        t0, current = time_range[..., 0], time_range[..., 1]
        fraction = (torch.arange(self.voxel_bins, device=voxel.device, dtype=voxel.dtype) + .5) / self.voxel_bins
        centers = t0.unsqueeze(-1) + (current-t0).unsqueeze(-1) * fraction
        weights = torch.exp(-(current.unsqueeze(-1)-centers) / self.event_decay_tau).clamp(0, 1)
        polarity_weights = torch.cat((weights, weights), dim=-1)
        return voxel * polarity_weights.unsqueeze(-1).unsqueeze(-1), weights


__all__ = ["LinearVoxelMultiscalePixelModel"]
