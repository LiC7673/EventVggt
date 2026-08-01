"""Compact neural modules for the DN pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from release_artifacts.dn_core_showcase.core_showcase import (
    EventConditionedHDRAdapter,
    EventNormalDerivativeHead,
    PixelGeometryRefiner,
    signed_temporal_voxel,
)


class EventPyramidEncoder(nn.Module):
    """Dense pixel encoder; it avoids patch-only decoding of sparse details."""

    def __init__(self, input_channels=10, hidden=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )

    def forward(self, voxel):
        return self.network(voxel)


class GeometryRelevanceField(nn.Module):
    """Inference-time event relevance from full events, RGB and coarse geometry."""

    def __init__(self, event_channels, hidden=32):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(event_channels + 5, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid(),
        )

    def forward(self, event_feature, rgb, coarse_depth, coarse_normal):
        luminance = rgb.mean(1, keepdim=True)
        inputs = torch.cat((event_feature, luminance, coarse_depth,
                            coarse_normal), dim=1)
        return self.predictor(inputs)


class PatchEventProjection(nn.Module):
    def __init__(self, event_channels, token_channels):
        super().__init__()
        self.projection = nn.Conv2d(event_channels, token_channels, 1, bias=False)

    def forward(self, feature, relevance, grid_size):
        selected = feature * relevance
        pooled = F.adaptive_avg_pool2d(selected, grid_size)
        return self.projection(pooled).flatten(2).transpose(1, 2)


class DerivativeToNormalCue(nn.Module):
    """Convert local dN evidence into a bounded correction around coarse N."""

    def __init__(self, hidden=32, correction_limit=0.10):
        super().__init__()
        self.limit = float(correction_limit)
        self.integrator = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, derivative, base_normal, relevance):
        flat = derivative.flatten(1, 2)
        delta = self.limit * torch.tanh(self.integrator(flat) / self.limit)
        return F.normalize(base_normal + relevance * delta, dim=1, eps=1.0e-6)


class EventGuidedDNModel(nn.Module):
    """Minimal composition around an arbitrary frozen RGB geometry backbone."""

    def __init__(self, rgb_geometry_backbone, depth_head, normal_from_depth,
                 event_bins=5, event_channels=32, token_channels=256):
        super().__init__()
        self.rgb_backbone = rgb_geometry_backbone
        self.depth_head = depth_head
        self.normal_from_depth = normal_from_depth
        self.event_bins = event_bins
        self.event_encoder = EventPyramidEncoder(2 * event_bins, event_channels)
        self.relevance = GeometryRelevanceField(event_channels)
        self.event_dn_head = EventNormalDerivativeHead(event_channels)
        self.derivative_to_normal = DerivativeToNormalCue()
        self.event_projection = PatchEventProjection(event_channels, token_channels)
        self.hdr_adapter = EventConditionedHDRAdapter(token_channels, token_channels // 4)
        self.refiner = PixelGeometryRefiner(event_channels)

    def forward(self, image, event_voxel, intrinsics, bin_age=None):
        with torch.no_grad():
            ldr_tokens = self.rgb_backbone(image)
            coarse_depth = self.depth_head(ldr_tokens)
            coarse_normal = self.normal_from_depth(coarse_depth, intrinsics)
        representation = signed_temporal_voxel(
            event_voxel, self.event_bins, bin_age=bin_age
        )
        event_feature = self.event_encoder(representation)
        relevance = self.relevance(
            event_feature, image, coarse_depth.detach(), coarse_normal.detach()
        )
        event_derivative = self.event_dn_head(event_feature)
        grid_size = ldr_tokens.shape[-3:-1]
        event_token = self.event_projection(event_feature, relevance, grid_size)
        hdr_tokens, token_residual = self.hdr_adapter(ldr_tokens, event_token)
        hdr_depth = self.depth_head(hdr_tokens)
        hdr_normal = self.normal_from_depth(hdr_depth, intrinsics)

        local_normal_target = self.derivative_to_normal(
            event_derivative, hdr_normal, relevance
        )
        final_depth, log_update = self.refiner(
            event_feature * relevance, hdr_depth, hdr_normal,
            local_normal_target, relevance,
        )
        return {
            "coarse_depth": coarse_depth,
            "hdr_depth": hdr_depth,
            "final_depth": final_depth,
            "event_derivative": event_derivative,
            "event_relevance": relevance,
            "log_depth_update": log_update,
            "token_residual": token_residual,
        }
