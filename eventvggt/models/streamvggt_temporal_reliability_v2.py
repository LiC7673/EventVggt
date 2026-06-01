"""Temporal event reliability gated detail refinement.

The depth correction proposal is predicted from RGB/coarse geometry, while
events determine how much of that proposal may be applied.  This V2 branch
uses temporal statistics in addition to accumulated event activity so that
persistent or polarity-mixed highlight trajectories can be rejected during
GT-supervised training.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from eventvggt.models.streamvggt_temporal_detail import StreamVGGTOutput, _ResidualBlock, _group_count
from eventvggt.models.streamvggt_temporal_gated_detail import (
    StreamVGGT as TemporalGatedStreamVGGT,
    TemporalEventGatedDetailRefiner,
)


class TemporalReliabilityDetailRefinerV2(TemporalEventGatedDetailRefiner):
    """Apply RGB depth proposals only where temporal events are reliable."""

    def __init__(
        self,
        *,
        num_bins: int = 10,
        hidden_dim: int = 16,
        count_cmax: float = 3.0,
        residual_scale: float = 0.015,
        gate_downsample: int = 4,
        reliability_floor: float = 0.0,
        reliability_init_bias: float = 0.0,
        proposal_depth_lowpass: bool = False,
        event_proposal_weight: float = 0.0,
        refine_points: bool = True,
        use_checkpoint: bool = True,
        min_depth: float = 1e-6,
    ) -> None:
        super().__init__(
            num_bins=num_bins,
            hidden_dim=hidden_dim,
            count_cmax=count_cmax,
            residual_scale=residual_scale,
            gate_downsample=gate_downsample,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
            min_depth=min_depth,
        )
        self.reliability_floor = min(max(float(reliability_floor), 0.0), 1.0)
        self.proposal_depth_lowpass = bool(proposal_depth_lowpass)
        self.event_proposal_weight = float(event_proposal_weight)
        groups = _group_count(hidden_dim)
        self.num_temporal_stats = 8
        self.temporal_reliability = nn.Sequential(
            nn.Conv2d(2 * self.num_bins + self.num_temporal_stats, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            _ResidualBlock(hidden_dim, dilation=2),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        self.event_geometry_proposal = nn.Sequential(
            nn.Conv2d(2 * self.num_bins + 4, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            _ResidualBlock(hidden_dim, dilation=2),
            _ResidualBlock(hidden_dim, dilation=3),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.temporal_reliability[-1].weight)
        nn.init.constant_(self.temporal_reliability[-1].bias, float(reliability_init_bias))
        nn.init.zeros_(self.event_geometry_proposal[-1].weight)
        nn.init.zeros_(self.event_geometry_proposal[-1].bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        first_weight_key = prefix + "temporal_reliability.0.weight"
        checkpoint_weight = state_dict.get(first_weight_key)
        target_weight = self.temporal_reliability[0].weight
        if checkpoint_weight is not None and checkpoint_weight.shape != target_weight.shape:
            adapted = torch.zeros_like(target_weight)
            out_channels = min(adapted.shape[0], checkpoint_weight.shape[0])
            in_channels = min(adapted.shape[1], checkpoint_weight.shape[1])
            kernel_h = min(adapted.shape[2], checkpoint_weight.shape[2])
            kernel_w = min(adapted.shape[3], checkpoint_weight.shape[3])
            adapted[:out_channels, :in_channels, :kernel_h, :kernel_w] = checkpoint_weight[
                :out_channels, :in_channels, :kernel_h, :kernel_w
            ].to(dtype=adapted.dtype)
            state_dict[first_weight_key] = adapted

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        proposal_depth = depth_feature
        if self.proposal_depth_lowpass:
            proposal_depth = F.avg_pool2d(depth_feature, kernel_size=5, stride=1, padding=2)
        depth_dx = F.pad(proposal_depth[..., :, 1:] - proposal_depth[..., :, :-1], (1, 0, 0, 0))
        depth_dy = F.pad(proposal_depth[..., 1:, :] - proposal_depth[..., :-1, :], (0, 0, 1, 0))
        depth_hf = proposal_depth - F.avg_pool2d(proposal_depth, kernel_size=7, stride=1, padding=3)
        proposal_input = torch.cat([image_flat, proposal_depth, depth_dx, depth_dy, depth_hf], dim=1)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            raw_proposal = checkpoint(self.geometry_proposal, proposal_input, use_reentrant=False)
        else:
            raw_proposal = self.geometry_proposal(proposal_input)

        base_gate, event_presence = self._event_confidence(voxel, output_size=(height, width))
        reliability, persistence, entropy, temporal_quality = self._temporal_reliability(
            voxel,
            output_size=(height, width),
        )
        reverse_reliability, _, _, _ = self._temporal_reliability(
            self._reverse_time_voxel(voxel),
            output_size=(height, width),
        )
        swap_reliability, _, _, _ = self._temporal_reliability(
            self._swap_polarity_voxel(voxel),
            output_size=(height, width),
        )
        event_gate = base_gate * (
            self.reliability_floor + (1.0 - self.reliability_floor) * reliability
        )

        rgb_delta_log = torch.tanh(raw_proposal) * event_gate * self.residual_scale
        event_delta_log = torch.zeros_like(rgb_delta_log)
        if self.event_proposal_weight != 0.0:
            event_proposal_input = torch.cat([voxel, image_flat, proposal_depth], dim=1)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                raw_event_proposal = checkpoint(self.event_geometry_proposal, event_proposal_input, use_reentrant=False)
            else:
                raw_event_proposal = self.event_geometry_proposal(event_proposal_input)
            event_delta_log = (
                torch.tanh(raw_event_proposal)
                * event_gate
                * self.residual_scale
                * self.event_proposal_weight
            )
        delta_log = rgb_delta_log + event_delta_log
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
        self.last_reliability = reliability.reshape(batch, seq_len, height, width)
        self.last_reliability_reverse_time = reverse_reliability.reshape(batch, seq_len, height, width)
        self.last_reliability_swap_polarity = swap_reliability.reshape(batch, seq_len, height, width)
        self.last_persistence = persistence.reshape(batch, seq_len, height, width).detach()
        self.last_entropy = entropy.reshape(batch, seq_len, height, width).detach()
        self.last_temporal_quality = temporal_quality.reshape(batch, seq_len, height, width).detach()
        self.last_delta_log = delta_log.reshape(batch, seq_len, height, width)
        self.last_rgb_delta_log = rgb_delta_log.reshape(batch, seq_len, height, width)
        self.last_event_delta_log = event_delta_log.reshape(batch, seq_len, height, width)
        return refined, refined_points, residual

    def _temporal_reliability(
        self,
        voxel: torch.Tensor,
        *,
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = F.avg_pool2d(
            voxel,
            kernel_size=self.gate_downsample,
            stride=self.gate_downsample,
            ceil_mode=True,
        )
        pos = pooled[:, : self.num_bins]
        neg = pooled[:, self.num_bins :]
        activity = pos + neg
        total = activity.sum(dim=1, keepdim=True)
        active = (total > 1e-6).to(dtype=voxel.dtype)

        presence = torch.log1p(total)
        max_presence = presence.flatten(2).amax(dim=-1, keepdim=True).view(-1, 1, 1, 1)
        presence = presence / max_presence.clamp_min(1e-6)
        presence = F.avg_pool2d(presence, kernel_size=3, stride=1, padding=1)

        probability = activity / total.clamp_min(1e-6)
        entropy = -(probability * torch.log(probability.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        entropy = entropy / max(math.log(float(max(self.num_bins, 2))), 1e-6)
        entropy = entropy * active
        peak = activity.amax(dim=1, keepdim=True) / total.clamp_min(1e-6)
        peak = peak * active
        persistence = activity / activity.amax(dim=1, keepdim=True).clamp_min(1e-6)
        persistence = persistence.mean(dim=1, keepdim=True) * active

        pos_sum = pos.sum(dim=1, keepdim=True)
        neg_sum = neg.sum(dim=1, keepdim=True)
        polarity_focus = (pos_sum - neg_sum).abs() / total.clamp_min(1e-6)
        polarity_focus = polarity_focus.clamp(0.0, 1.0) * active
        polarity_mix = 1.0 - polarity_focus
        polarity_mix = polarity_mix.clamp(0.0, 1.0) * active

        time = torch.linspace(-1.0, 1.0, self.num_bins, device=voxel.device, dtype=voxel.dtype).view(
            1, self.num_bins, 1, 1
        )
        time_center = (probability * time).sum(dim=1, keepdim=True) * active
        time_spread = torch.sqrt(
            (probability * (time - time_center).square()).sum(dim=1, keepdim=True).clamp_min(1e-8)
        ) * active
        signed_direction = ((pos - neg) * time).sum(dim=1, keepdim=True) / total.clamp_min(1e-6)
        signed_direction = signed_direction * active
        temporal_focus = (1.0 - entropy).clamp(0.0, 1.0)
        temporal_quality = temporal_focus * (0.25 + 0.75 * peak) * (0.25 + 0.75 * polarity_focus)
        temporal_quality = temporal_quality.clamp(0.0, 1.0) * active
        stats = torch.cat(
            [
                presence,
                persistence,
                polarity_mix,
                entropy,
                peak,
                time_spread,
                signed_direction,
                temporal_quality,
            ],
            dim=1,
        )
        reliability_low = torch.sigmoid(self.temporal_reliability(torch.cat([pooled, stats], dim=1)))
        reliability = F.interpolate(reliability_low, size=output_size, mode="bilinear", align_corners=False)
        persistence = F.interpolate(persistence, size=output_size, mode="bilinear", align_corners=False)
        entropy = F.interpolate(entropy, size=output_size, mode="bilinear", align_corners=False)
        temporal_quality = F.interpolate(temporal_quality, size=output_size, mode="bilinear", align_corners=False)
        return reliability, persistence, entropy, temporal_quality

    def _reverse_time_voxel(self, voxel: torch.Tensor) -> torch.Tensor:
        pos = voxel[:, : self.num_bins].flip(1)
        neg = voxel[:, self.num_bins :].flip(1)
        return torch.cat([pos, neg], dim=1)

    def _swap_polarity_voxel(self, voxel: torch.Tensor) -> torch.Tensor:
        pos = voxel[:, : self.num_bins]
        neg = voxel[:, self.num_bins :]
        return torch.cat([neg, pos], dim=1)


class StreamVGGT(TemporalGatedStreamVGGT):
    """Temporal-reliability V2 wrapper with low-memory multi-LDR forwarding."""

    def __init__(
        self,
        *args,
        event_hidden_dim: int = 16,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        residual_scale: float = 0.015,
        gate_downsample: int = 4,
        event_reliability_floor: float = 0.0,
        event_reliability_init_bias: float = 0.0,
        proposal_depth_lowpass: bool = False,
        event_proposal_weight: float = 0.0,
        forward_batch_chunk: int = 1,
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
            gate_downsample=gate_downsample,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.forward_batch_chunk = max(0, int(forward_batch_chunk))
        self.event_detail_refiner = TemporalReliabilityDetailRefinerV2(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            gate_downsample=gate_downsample,
            reliability_floor=event_reliability_floor,
            reliability_init_bias=event_reliability_init_bias,
            proposal_depth_lowpass=proposal_depth_lowpass,
            event_proposal_weight=event_proposal_weight,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
        )

    def _forward_chunk(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        reliability = getattr(self.event_detail_refiner, "last_reliability", None)
        reliability_reverse_time = getattr(self.event_detail_refiner, "last_reliability_reverse_time", None)
        reliability_swap_polarity = getattr(self.event_detail_refiner, "last_reliability_swap_polarity", None)
        delta_log = getattr(self.event_detail_refiner, "last_delta_log", None)
        rgb_delta_log = getattr(self.event_detail_refiner, "last_rgb_delta_log", None)
        event_delta_log = getattr(self.event_detail_refiner, "last_event_delta_log", None)
        persistence = getattr(self.event_detail_refiner, "last_persistence", None)
        entropy = getattr(self.event_detail_refiner, "last_entropy", None)
        temporal_quality = getattr(self.event_detail_refiner, "last_temporal_quality", None)
        if reliability is not None:
            for frame_idx, result in enumerate(output.ress):
                result["event_reliability"] = reliability[:, frame_idx]
                result["event_reliability_reverse_time"] = reliability_reverse_time[:, frame_idx]
                result["event_reliability_swap_polarity"] = reliability_swap_polarity[:, frame_idx]
                result["depth_delta_log"] = delta_log[:, frame_idx]
                result["rgb_delta_log"] = rgb_delta_log[:, frame_idx]
                result["event_delta_log"] = event_delta_log[:, frame_idx]
                result["event_persistence"] = persistence[:, frame_idx]
                result["event_entropy"] = entropy[:, frame_idx]
                result["event_temporal_quality"] = temporal_quality[:, frame_idx]
        return output

    def forward(self, views, query_points: Optional[torch.Tensor] = None, **kwargs):
        batch = views[0]["img"].shape[0]
        chunk = self.forward_batch_chunk
        if chunk <= 0 or batch <= chunk:
            return self._forward_chunk(views, query_points=query_points, **kwargs)

        outputs = []
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            chunk_views = self._slice_views(views, start, end, batch)
            chunk_query = query_points
            if torch.is_tensor(query_points) and query_points.ndim > 0 and query_points.shape[0] == batch:
                chunk_query = query_points[start:end]
            outputs.append(self._forward_chunk(chunk_views, query_points=chunk_query, **kwargs))
        return self._concat_outputs(outputs, views)

    @staticmethod
    def _slice_views(views, start: int, end: int, batch: int):
        chunk_views = []
        for view in views:
            chunk_view = {}
            for key, value in view.items():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
                    chunk_view[key] = value[start:end]
                elif isinstance(value, (list, tuple)) and len(value) == batch:
                    chunk_view[key] = value[start:end]
                else:
                    chunk_view[key] = value
            chunk_views.append(chunk_view)
        return chunk_views

    @staticmethod
    def _concat_outputs(outputs, views):
        ress = []
        for frame_idx in range(len(outputs[0].ress)):
            frame = {}
            for key in outputs[0].ress[frame_idx]:
                values = [output.ress[frame_idx][key] for output in outputs]
                frame[key] = torch.cat(values, dim=0) if torch.is_tensor(values[0]) else values
            ress.append(frame)
        return StreamVGGTOutput(
            ress=ress,
            views=views,
            depth_coarse=torch.cat([output.depth_coarse for output in outputs], dim=0),
            depth_residual=torch.cat([output.depth_residual for output in outputs], dim=0),
        )


__all__ = ["StreamVGGT", "StreamVGGTOutput", "TemporalReliabilityDetailRefinerV2"]
