"""Normal-oriented event refinement described by the root ``toDo.md``.

This is intentionally a new model variant.  Historical unified/Stage-2 files
remain unchanged so their checkpoints stay reproducible.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.contribution_stage1 import normalize_event_voxel
from paired_token_reliability.unified_model import UnifiedGeometryContributionModel
from stage2_geometry_adapter.model import (
    GeometryAdapterOutput,
    PolarityTemporalEventPyramid,
    depth_to_normals,
    dpt_feature_shapes,
)


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def soft_event_support(event_voxel: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """A local, soft support mask; it never opens an entire patch-grid cell."""
    support = event_voxel.abs().sum(dim=2, keepdim=True).clamp(0.0, 1.0)
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            raise ValueError("support_dilation_kernel must be odd")
        batch, views = support.shape[:2]
        support = F.max_pool2d(
            support.reshape(batch * views, 1, *support.shape[-2:]),
            kernel_size, stride=1, padding=kernel_size // 2,
        ).reshape_as(support)
    return support


class ZeroPreservingEventPyramid(PolarityTemporalEventPyramid):
    """Encode full events, then explicitly mask every output feature scale."""

    def __init__(self, *args, support_dilation_kernel: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_dilation_kernel = int(support_dilation_kernel)

    def forward(self, event_voxel, contribution, target_shapes):
        # The inherited encoder is reused, but receives C=1: contribution is
        # deliberately not applied to V.  Its output is masked again below.
        ones = torch.ones_like(contribution)
        features, _ = super().forward(event_voxel, ones, target_shapes)
        support = soft_event_support(event_voxel, self.support_dilation_kernel)
        batch, views = event_voxel.shape[:2]
        gates = []
        masked_features = []
        for feature, shape in zip(features, target_shapes):
            scale_support = F.interpolate(
                support.reshape(batch * views, 1, *support.shape[-2:]),
                size=shape, mode="bilinear", align_corners=False,
            )
            scale_contribution = F.interpolate(
                contribution.reshape(batch * views, 1, *contribution.shape[-2:]),
                size=shape, mode="bilinear", align_corners=False,
            ).clamp(0.0, 1.0)
            gate = scale_support * scale_contribution
            masked_features.append(feature * scale_support.reshape(batch, views, 1, *shape))
            gates.append(gate.reshape(batch, views, 1, *shape))
        return masked_features, gates


class EventOnlyFeatureAdapter(nn.Module):
    """Generate a bounded feature update from events only, with one soft gate."""

    def __init__(self, rgb_channels, event_channels, hidden_channels, enabled=True):
        super().__init__()
        hidden = max(min(int(hidden_channels), int(rgb_channels)), 32)
        self.enabled = bool(enabled)
        self.adapter = nn.Sequential(
            nn.Conv2d(event_channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=_groups(hidden), bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, rgb_channels, 1),
        )
        nn.init.normal_(self.adapter[-1].weight, std=1.0e-2)
        nn.init.zeros_(self.adapter[-1].bias)
        self.alpha_logit = nn.Parameter(torch.zeros(()))

    def forward(self, rgb_feature, event_feature, contribution):
        if not self.enabled:
            zero = torch.zeros_like(rgb_feature)
            return rgb_feature, zero, zero.mean()
        raw_update = torch.tanh(self.alpha_logit) * self.adapter(event_feature)
        applied_update = contribution.to(raw_update.dtype) * raw_update
        return rgb_feature + applied_update, applied_update, raw_update.abs().mean()


class EventOnlyNormalDecoder(nn.Module):
    """Predict an absolute unit normal from events only; no RGB/coarse input."""

    def __init__(self, event_channels, hidden_channels=64):
        super().__init__()
        hidden = max(int(hidden_channels), 16)
        self.decoder = nn.Sequential(
            nn.Conv2d(event_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        with torch.no_grad():
            self.decoder[-1].bias.copy_(torch.tensor([0.0, 0.0, 1.0]))

    def forward(self, event_feature, gate, output_size):
        raw = self.decoder(event_feature)
        if raw.shape[-2:] != tuple(output_size):
            raw = F.interpolate(raw, output_size, mode="bilinear", align_corners=False)
            gate = F.interpolate(gate, output_size, mode="bilinear", align_corners=False)
        # Gate is intentionally not mixed into the vector: multiplying a
        # normal changes magnitude, not direction. It defines valid/support
        # and supervision regions only.
        normal = F.normalize(raw.float(), dim=1, eps=1.0e-6)
        return normal.movedim(1, -1), gate[:, 0]


class NormalOrientedGeometryContributionModel(UnifiedGeometryContributionModel):
    checkpoint_schema = "normal_oriented_geometry_contribution_v1"

    def __init__(self, *args, event_adapter_levels=(0, 1),
                 support_dilation_kernel=5, enable_event_depth_residual=False, **kwargs):
        if enable_event_depth_residual:
            raise NotImplementedError("toDo.md keeps the optional depth residual disabled by default")
        super().__init__(*args, **kwargs)
        levels = {int(level) for level in event_adapter_levels}
        if not levels.issubset({0, 1, 2, 3}):
            raise ValueError(f"event_adapter_levels must be within 0..3, got {sorted(levels)}")
        event_channels = self.event_encoder.scale_projections[0][0].out_channels
        hidden = kwargs.get("adapter_hidden_channels", 128)
        for head in (self.depth_head, self.point_head):
            rgb_channels = [adapter.adapter[-1].out_channels for adapter in head.geometry_adapters]
            head.geometry_adapters = nn.ModuleList([
                EventOnlyFeatureAdapter(c, event_channels, hidden, index in levels)
                for index, c in enumerate(rgb_channels)
            ])
        self.event_encoder = ZeroPreservingEventPyramid(
            num_bins=self.event_encoder.num_bins,
            hidden_channels=kwargs.get("event_hidden_dim", 48),
            pyramid_channels=event_channels,
            count_cmax=self.event_encoder.count_cmax,
            support_dilation_kernel=support_dilation_kernel,
        )
        self.event_normal_decoder = EventOnlyNormalDecoder(
            event_channels, kwargs.get("event_hidden_dim", 48)
        )
        self.event_adapter_levels = tuple(sorted(levels))

    def forward(self, views, query_points: Optional[torch.Tensor] = None,
                contribution_override: Optional[torch.Tensor] = None,
                decode_event_normal: bool = True, **_kwargs):
        images, event_voxel, intrinsics = self._stack_inputs(views)
        event_voxel, intrinsics = event_voxel.to(images.device), intrinsics.to(images.device)
        if query_points is not None and query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)
        tokens, patch_start = self.aggregator(images)
        pose = self.camera_head(tokens)[-1]
        coarse_depth, coarse_conf = self.depth_head(
            tokens, images=images, patch_start_idx=patch_start,
            frames_chunk_size=self.head_frames_chunk_size,
        )
        coarse_map = coarse_depth[..., 0] if coarse_depth.shape[-1] == 1 else coarse_depth.squeeze(2)
        coarse_normals = depth_to_normals(coarse_map.float(), intrinsics.float())
        coarse_features = self._coarse_patch_features(tokens, patch_start, *images.shape[-2:])
        if contribution_override is None:
            depth_prior, normal_prior, feature_prior = coarse_map, coarse_normals, coarse_features
            if not self.contribution_use_geometry_prior:
                depth_prior, normal_prior, feature_prior = (
                    torch.ones_like(coarse_map), torch.zeros_like(coarse_normals),
                    torch.zeros_like(coarse_features),
                )
            contribution = self.contribution_net(
                event_voxel, images, depth_prior, normal_prior,
                feature_prior if self.contribution_net.coarse_feature_dim > 0 else None,
            )
        else:
            contribution = contribution_override.to(event_voxel).clamp(0, 1)
        if contribution.ndim == event_voxel.ndim:
            mass = event_voxel.abs()
            spatial_contribution = (contribution * mass).sum(2) / mass.sum(2).clamp_min(1e-6)
        else:
            spatial_contribution = contribution
        shapes = dpt_feature_shapes(*images.shape[-2:], self.patch_size)
        event_pyramid, gates = self.event_encoder(event_voxel, spatial_contribution, shapes)
        batch, views_count = images.shape[:2]
        event_normal = normal_gate = None
        if decode_event_normal:
            feature = event_pyramid[0].reshape(batch * views_count, event_pyramid[0].shape[2], *shapes[0])
            gate = gates[0].reshape(batch * views_count, 1, *shapes[0])
            event_normal, normal_gate = self.event_normal_decoder(feature, gate, images.shape[-2:])
            event_normal = event_normal.reshape(batch, views_count, *event_normal.shape[1:])
            normal_gate = normal_gate.reshape(batch, views_count, *normal_gate.shape[1:])
        depth, depth_conf = self.depth_head(tokens, images=images, patch_start_idx=patch_start,
            frames_chunk_size=self.head_frames_chunk_size, event_pyramid=event_pyramid, contribution_pyramid=gates)
        points, point_conf = self.point_head(tokens, images=images, patch_start_idx=patch_start,
            frames_chunk_size=self.head_frames_chunk_size, event_pyramid=event_pyramid, contribution_pyramid=gates)
        update_loss = .5 * (self.depth_head.last_update_loss + self.point_head.last_update_loss)
        results = []
        for index in range(views_count):
            item = dict(pts3d_in_other_view=points[:, index], conf=point_conf[:, index],
                depth=depth[:, index], normal=event_normal[:, index] if event_normal is not None else coarse_normals[:, index],
                depth_conf=depth_conf[:, index], depth_coarse=coarse_depth[:, index],
                depth_coarse_conf=coarse_conf[:, index], camera_pose=pose[:, index],
                event_contribution=contribution[:, index], event_contribution_spatial=spatial_contribution[:, index],
                selected_event_mass=event_voxel[:, index].abs().sum(1), adapter_update_loss=update_loss,
                adapter_alpha_depth=torch.stack([torch.tanh(a.alpha_logit) for a in self.depth_head.geometry_adapters]),
                adapter_alpha_point=torch.stack([torch.tanh(a.alpha_logit) for a in self.point_head.geometry_adapters]),
                adapter_depth_update_magnitudes=torch.stack(self.depth_head.last_update_magnitudes),
                adapter_point_update_magnitudes=torch.stack(self.point_head.last_update_magnitudes))
            if event_normal is not None:
                item.update(event_normal=event_normal[:, index],
                            event_normal_reliability=normal_gate[:, index],
                            event_normal_support=normal_gate[:, index] > 0)
            results.append(item)
        return GeometryAdapterOutput(ress=results, views=views)


__all__ = ["EventOnlyNormalDecoder", "EventOnlyFeatureAdapter",
           "NormalOrientedGeometryContributionModel", "ZeroPreservingEventPyramid",
           "soft_event_support"]
