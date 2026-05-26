"""Exposure-invariant temporal event gating for fine depth refinement.

This variant extends the stable temporal gated refiner with a low-resolution
RGB/event matching branch. During multi-exposure training, the branch learns
an appearance descriptor that stays stable across LDR levels while agreeing
with the shared temporal event evidence. At inference it still consumes one
LDR sequence and its event voxels only.
"""

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


class ExposureInvariantEventGatedDetailRefiner(TemporalEventGatedDetailRefiner):
    """Gate RGB depth corrections by event agreement stable across exposure."""

    def __init__(
        self,
        *,
        num_bins: int = 10,
        hidden_dim: int = 16,
        count_cmax: float = 3.0,
        residual_scale: float = 0.01,
        gate_downsample: int = 4,
        match_dim: int = 8,
        agreement_floor: float = 0.25,
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
        self.match_dim = max(2, int(match_dim))
        self.agreement_floor = min(max(float(agreement_floor), 0.0), 1.0)
        groups = _group_count(hidden_dim)

        # Matching is intentionally coarse: events can assess reliability but
        # cannot stamp high-frequency specular trajectories into depth.
        self.appearance_match = nn.Sequential(
            nn.Conv2d(7, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            nn.Conv2d(hidden_dim, self.match_dim, kernel_size=1),
        )
        self.event_match = nn.Sequential(
            nn.Conv2d(2 * self.num_bins, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            nn.Conv2d(hidden_dim, self.match_dim, kernel_size=1),
        )
        self.agreement_head = nn.Sequential(
            nn.Conv2d(3 * self.match_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.agreement_head[-1].weight)
        nn.init.constant_(self.agreement_head[-1].bias, 6.0)

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
        depth_dx = F.pad(depth_feature[..., :, 1:] - depth_feature[..., :, :-1], (1, 0, 0, 0))
        depth_dy = F.pad(depth_feature[..., 1:, :] - depth_feature[..., :-1, :], (0, 0, 1, 0))
        depth_hf = depth_feature - F.avg_pool2d(depth_feature, kernel_size=5, stride=1, padding=2)
        proposal_input = torch.cat([image_flat, depth_feature, depth_dx, depth_dy, depth_hf], dim=1)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            raw_proposal = checkpoint(self.geometry_proposal, proposal_input, use_reentrant=False)
        else:
            raw_proposal = self.geometry_proposal(proposal_input)

        base_gate, event_presence = self._event_confidence(voxel, output_size=(height, width))
        pooled_proposal = F.avg_pool2d(
            proposal_input,
            kernel_size=self.gate_downsample,
            stride=self.gate_downsample,
            ceil_mode=True,
        )
        pooled_voxel = F.avg_pool2d(
            voxel,
            kernel_size=self.gate_downsample,
            stride=self.gate_downsample,
            ceil_mode=True,
        )
        appearance_feature = F.normalize(self.appearance_match(pooled_proposal), dim=1, eps=1e-6)
        event_feature = F.normalize(self.event_match(pooled_voxel), dim=1, eps=1e-6)
        agreement_input = torch.cat(
            [appearance_feature, event_feature, (appearance_feature - event_feature).abs()],
            dim=1,
        )
        agreement_low = torch.sigmoid(self.agreement_head(agreement_input))
        agreement = F.interpolate(agreement_low, size=(height, width), mode="bilinear", align_corners=False)
        event_gate = base_gate * (self.agreement_floor + (1.0 - self.agreement_floor) * agreement)

        delta_log = torch.tanh(raw_proposal) * event_gate * self.residual_scale
        refined_flat = depth_flat.to(dtype=delta_log.dtype) * torch.exp(delta_log)
        refined = refined_flat.permute(0, 2, 3, 1).reshape(batch, seq_len, height, width, 1)
        refined = refined.to(dtype=depth.dtype).clamp_min(self.min_depth)
        residual = refined - depth

        refined_points = points
        if self.refine_points and points is not None:
            ratio = refined / depth.clamp_min(self.min_depth)
            refined_points = points * ratio.to(dtype=points.dtype)

        match_height, match_width = appearance_feature.shape[-2:]
        self.last_gate = event_gate.reshape(batch, seq_len, height, width).detach()
        self.last_presence = event_presence.reshape(batch, seq_len, height, width).detach()
        self.last_agreement = agreement.reshape(batch, seq_len, height, width)
        self.last_appearance_feature = appearance_feature.reshape(
            batch, seq_len, self.match_dim, match_height, match_width
        )
        self.last_event_feature = event_feature.reshape(batch, seq_len, self.match_dim, match_height, match_width)
        return refined, refined_points, residual


class StreamVGGT(TemporalGatedStreamVGGT):
    """Temporal gated model with a train-time cross-exposure matching branch."""

    def __init__(
        self,
        *args,
        event_hidden_dim: int = 16,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        residual_scale: float = 0.01,
        gate_downsample: int = 4,
        exposure_match_dim: int = 8,
        exposure_agreement_floor: float = 0.25,
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
        self.event_detail_refiner = ExposureInvariantEventGatedDetailRefiner(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            gate_downsample=gate_downsample,
            match_dim=exposure_match_dim,
            agreement_floor=exposure_agreement_floor,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
        )

    def _forward_chunk(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        agreement = getattr(self.event_detail_refiner, "last_agreement", None)
        appearance_feature = getattr(self.event_detail_refiner, "last_appearance_feature", None)
        event_feature = getattr(self.event_detail_refiner, "last_event_feature", None)
        if agreement is not None:
            for frame_idx, result in enumerate(output.ress):
                result["event_agreement"] = agreement[:, frame_idx]
                result["exposure_feature"] = appearance_feature[:, frame_idx]
                result["exposure_event_feature"] = event_feature[:, frame_idx]
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


__all__ = ["StreamVGGT", "StreamVGGTOutput", "ExposureInvariantEventGatedDetailRefiner"]
