"""Independent signed-event, pixel-level multi-scale geometry model.

Five temporal bins are represented as five signed channels: +1 positive,
-1 negative, 0 empty.  No event module receives split polarity channels and
no event update passes through a ViT/DPT patch grid.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.unified_model import UnifiedGeometryContributionModel
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


def split_to_signed(event_voxel: torch.Tensor, bins: int) -> torch.Tensor:
    if event_voxel.shape[2] != 2 * bins:
        raise ValueError(f"expected {2*bins} split-polarity channels, got {event_voxel.shape[2]}")
    positive = event_voxel[:, :, :bins].gt(0).to(event_voxel.dtype)
    negative = event_voxel[:, :, bins:].gt(0).to(event_voxel.dtype)
    return (positive - negative).clamp(-1, 1)


def temporal_decay_signed(signed: torch.Tensor, time_range: torch.Tensor,
                          tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponentially downweight old bins using inferred uniform bin centers."""
    if tau <= 0:
        raise ValueError("event_decay_tau must be positive seconds")
    bins = signed.shape[2]
    t0, current = time_range[..., 0], time_range[..., 1]
    fraction = (torch.arange(bins, device=signed.device, dtype=signed.dtype) + .5) / bins
    centers = t0.unsqueeze(-1) + (current - t0).unsqueeze(-1) * fraction
    weights = torch.exp(-(current.unsqueeze(-1) - centers) / float(tau)).clamp(0, 1)
    return signed * weights.unsqueeze(-1).unsqueeze(-1), weights


def signed_support(event: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    support = event.ne(0).any(dim=2, keepdim=True).to(event.dtype)
    if kernel > 1:
        b, v = support.shape[:2]
        support = F.max_pool2d(support.reshape(b*v, 1, *support.shape[-2:]), kernel, 1, kernel//2)
        support = support.reshape(b, v, 1, *support.shape[-2:])
    return support


class SignedMultiScaleEncoder(nn.Module):
    """Stride-one multi-dilation encoder; output remains at pixel resolution."""
    def __init__(self, bins=5, hidden=32):
        super().__init__()
        self.bins, self.hidden = int(bins), int(hidden)
        self.stem = nn.Sequential(nn.Conv2d(bins, hidden, 3, padding=1, bias=False), nn.GELU())
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d, bias=False), nn.GELU())
            for d in (1, 2, 4, 8)
        ])
        self.fuse = nn.Sequential(nn.Conv2d(4*hidden, hidden, 1, bias=False), nn.GELU())

    def forward(self, signed):
        b, v = signed.shape[:2]
        x = signed.reshape(b*v, self.bins, *signed.shape[-2:])
        x = self.stem(x)
        x = self.fuse(torch.cat([branch(x) for branch in self.branches], 1))
        return x.reshape(b, v, self.hidden, *signed.shape[-2:])


class SignedContributionNet(nn.Module):
    """Pixel contribution from signed temporal events plus attribution priors."""
    def __init__(self, bins=5, hidden=32, initial=0.95):
        super().__init__()
        self.coarse_feature_dim = 0
        self.net = nn.Sequential(
            nn.Conv2d(bins + 7, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, torch.logit(torch.tensor(float(initial))).item())

    def forward(self, signed, image, depth, normals, _features=None):
        b, v = signed.shape[:2]
        x = torch.cat((signed, image, depth.unsqueeze(2), normals.movedim(-1, 2)), 2)
        return torch.sigmoid(self.net(x.reshape(b*v, x.shape[2], *x.shape[-2:]))[:, 0]).reshape(b, v, *x.shape[-2:])


class SignedMultiscalePixelModel(UnifiedGeometryContributionModel):
    checkpoint_schema = "signed_multiscale_pixel_geometry_v1"

    def __init__(self, *args, signed_event_bins=5, pixel_hidden=32,
                 support_dilation_kernel=5, depth_update_scale=0.03,
                 event_decay_tau=0.003, **kwargs):
        kwargs["event_num_bins"] = int(signed_event_bins)
        super().__init__(*args, **kwargs)
        self.signed_event_bins = int(signed_event_bins)
        self.support_dilation_kernel = int(support_dilation_kernel)
        self.depth_update_scale = float(depth_update_scale)
        self.event_decay_tau = float(event_decay_tau)
        self.contribution_net = SignedContributionNet(self.signed_event_bins, pixel_hidden,
                                                       kwargs.get("contribution_initial_value", .95))
        self.event_encoder = SignedMultiScaleEncoder(self.signed_event_bins, pixel_hidden)
        self.event_normal_decoder = nn.Sequential(
            nn.Conv2d(pixel_hidden, pixel_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(pixel_hidden, 3, 1),
        )
        self.depth_local_head = nn.Sequential(
            nn.Conv2d(pixel_hidden, pixel_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(pixel_hidden, 1, 1),
        )
        nn.init.zeros_(self.depth_local_head[-1].weight)
        nn.init.zeros_(self.depth_local_head[-1].bias)
        # Historical patch-grid adapters are never used or trained.
        self.depth_head.geometry_adapters.requires_grad_(False)
        self.point_head.geometry_adapters.requires_grad_(False)

    def predict_contribution(self, views):
        images, split_event, intrinsics = self._stack_inputs(views)
        split_event, intrinsics = split_event.to(images.device), intrinsics.to(images.device)
        signed, _ = self._decayed_signed(views, split_event)
        tokens, patch_start = self.aggregator(images)
        coarse_depth, _ = self.depth_head(tokens, images=images, patch_start_idx=patch_start,
                                           frames_chunk_size=self.head_frames_chunk_size)
        coarse_map = coarse_depth[..., 0]
        coarse_normal = depth_to_normals(coarse_map.float(), intrinsics.float())
        return self.contribution_net(signed, images, coarse_map, coarse_normal)

    def _decayed_signed(self, views, split_event):
        signed = split_to_signed(split_event, self.signed_event_bins)
        ranges = []
        for view in views:
            value = view.get("event_time_range")
            if not torch.is_tensor(value):
                raise KeyError("event_time_range is required for physical temporal decay")
            ranges.append(value.to(device=signed.device, dtype=signed.dtype))
        time_range = torch.stack(ranges, dim=1)
        return temporal_decay_signed(signed, time_range, self.event_decay_tau)

    def forward(self, views, query_points: Optional[torch.Tensor] = None,
                contribution_override=None, **_kwargs):
        images, split_event, intrinsics = self._stack_inputs(views)
        split_event, intrinsics = split_event.to(images.device), intrinsics.to(images.device)
        signed, decay_weights = self._decayed_signed(views, split_event)
        tokens, patch_start = self.aggregator(images)
        pose = self.camera_head(tokens)[-1]
        coarse_depth, coarse_conf = self.depth_head(tokens, images=images, patch_start_idx=patch_start,
                                                     frames_chunk_size=self.head_frames_chunk_size)
        coarse_map = coarse_depth[..., 0]
        coarse_normal = depth_to_normals(coarse_map.float(), intrinsics.float())
        contribution = (self.contribution_net(signed, images, coarse_map, coarse_normal)
                        if contribution_override is None else
                        contribution_override.to(signed).mean(2) if contribution_override.ndim == 5
                        else contribution_override.to(signed))
        support = signed_support(signed, self.support_dilation_kernel)
        gate = support * contribution.unsqueeze(2).clamp(0, 1)
        feature = self.event_encoder(signed)
        b, v = signed.shape[:2]
        flat = feature.reshape(b*v, feature.shape[2], *feature.shape[-2:])
        event_normal = F.normalize(self.event_normal_decoder(flat).float(), dim=1, eps=1e-6)
        event_normal = event_normal.movedim(1, -1).reshape(b, v, *signed.shape[-2:], 3)
        delta_ratio = self.depth_update_scale * torch.tanh(self.depth_local_head(flat)[:, 0])
        delta_ratio = delta_ratio.reshape(b, v, *signed.shape[-2:]) * gate[:, :, 0]
        depth_map = coarse_map * (1.0 + delta_ratio)
        depth = depth_map.unsqueeze(-1)
        final_normal = depth_to_normals(depth_map.float(), intrinsics.float())
        points, point_conf = self.point_head(tokens, images=images, patch_start_idx=patch_start,
                                              frames_chunk_size=self.head_frames_chunk_size)
        results = []
        for i in range(v):
            results.append(dict(pts3d_in_other_view=points[:, i], conf=point_conf[:, i],
                depth=depth[:, i], normal=final_normal[:, i], depth_conf=coarse_conf[:, i],
                depth_coarse=coarse_depth[:, i], depth_coarse_conf=coarse_conf[:, i], camera_pose=pose[:, i],
                event_contribution=contribution[:, i], event_contribution_spatial=contribution[:, i],
                signed_event=signed[:, i], temporal_decay_weights=decay_weights[:, i],
                event_normal=event_normal[:, i],
                # Normal prediction/supervision is dense on the dataset valid
                # object mask. Event support only raises its confidence weight;
                # it never cuts the predicted normal into an event-shaped map.
                event_normal_reliability=gate[:, i, 0],
                event_normal_support=torch.ones_like(support[:, i, 0], dtype=torch.bool),
                depth_pixel_update=(depth_map-coarse_map)[:, i], adapter_update_loss=delta_ratio.abs().mean(),
                adapter_alpha_depth=depth_map.new_zeros(4), adapter_alpha_point=depth_map.new_zeros(4),
                adapter_depth_update_magnitudes=depth_map.new_zeros(4),
                adapter_point_update_magnitudes=depth_map.new_zeros(4), selected_event_mass=signed[:, i].abs().sum(1)))
        return GeometryAdapterOutput(ress=results, views=views)


__all__ = ["SignedMultiscalePixelModel", "SignedMultiScaleEncoder", "split_to_signed",
           "temporal_decay_signed"]
