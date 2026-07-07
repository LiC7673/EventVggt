"""Temporal event detail with a frozen paired-token distilled reliability gate.

The full event voxel always reaches the detail refiner. Reliability only gates
the final log-depth correction, so uncertain pixels cannot erase event features
before temporal reasoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eventvggt.models.streamvggt_pretrained_reliability_detail import (
    _load_checkpoint,
    _unwrap_reliability_state,
)
from eventvggt.models.streamvggt_temporal_detail import StreamVGGT as TemporalDetailStreamVGGT
from reliability_pretrain.model import ReliabilityUNet


class FrozenOutputReliabilityGate(nn.Module):
    """Preserve full event features and gate only their predicted correction."""

    def __init__(
        self,
        base_refiner: nn.Module,
        *,
        checkpoint: str,
        num_bins: int = 10,
        base_channels: int = 32,
        count_cmax: float = 3.0,
        gate_floor: float = 0.15,
        dilate_kernel: int = 3,
        frame_chunk_size: int = 1,
        rgb_input_range: str = "minus_one_one",
    ) -> None:
        super().__init__()
        self.base_refiner = base_refiner
        self.num_bins = max(int(num_bins), 1)
        self.count_cmax = max(float(count_cmax), 1.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.dilate_kernel = max(int(dilate_kernel), 1)
        if self.dilate_kernel % 2 == 0:
            self.dilate_kernel += 1
        self.frame_chunk_size = max(int(frame_chunk_size), 1)
        self.rgb_input_range = str(rgb_input_range)

        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Paired-token ReliabilityUNet checkpoint not found: {checkpoint_path}")
        self.reliability_net = ReliabilityUNet(
            event_channels=2 * self.num_bins,
            image_channels=3,
            base_channels=int(base_channels),
        )
        state = _unwrap_reliability_state(_load_checkpoint(checkpoint_path))
        incompatible = self.reliability_net.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"ReliabilityUNet checkpoint mismatch: {incompatible}")
        self.reliability_net.requires_grad_(False).eval()
        self.reliability_checkpoint = str(checkpoint_path)

        self.last_reliability: Optional[torch.Tensor] = None
        self.last_gate: Optional[torch.Tensor] = None
        self.last_delta_log: Optional[torch.Tensor] = None
        self.last_event_support: Optional[torch.Tensor] = None
        self.last_filtered_event_abs_mean: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.reliability_net.eval()
        return self

    def _normalize_event(self, voxel: torch.Tensor) -> torch.Tensor:
        denominator = torch.log1p(voxel.new_tensor(self.count_cmax))
        return torch.log1p(voxel.clamp_min(0.0).clamp_max(self.count_cmax)) / denominator

    def _prepare_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.rgb_input_range == "minus_one_one" and float(rgb.detach().amin()) >= -0.05:
            return rgb * 2.0 - 1.0
        if self.rgb_input_range == "zero_one" and float(rgb.detach().amin()) < -0.05:
            return (rgb + 1.0) * 0.5
        return rgb

    @torch.no_grad()
    def _predict_reliability(self, event_voxel: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, height, width = event_voxel.shape
        if channels != 2 * self.num_bins:
            raise ValueError(
                f"ReliabilityUNet expects {2 * self.num_bins} channels, got {channels}. "
                "Keep data.event_resize_bins equal to the stage-1 value."
            )
        event_flat = event_voxel.reshape(batch * seq_len, channels, height, width)
        image_flat = images.reshape(batch * seq_len, 3, height, width)
        predictions = []
        for start in range(0, event_flat.shape[0], self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, event_flat.shape[0])
            with torch.autocast(device_type=event_voxel.device.type, enabled=False):
                predictions.append(
                    self.reliability_net(
                        self._normalize_event(event_flat[start:end].float()),
                        self._prepare_rgb(image_flat[start:end].float()),
                    )
                )
        reliability = torch.cat(predictions, dim=0)
        if self.dilate_kernel > 1:
            reliability = F.max_pool2d(
                reliability,
                kernel_size=self.dilate_kernel,
                stride=1,
                padding=self.dilate_kernel // 2,
            )
        return reliability.reshape(batch, seq_len, height, width)

    def forward(
        self,
        *,
        event_voxel: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        points: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        reliability = self._predict_reliability(event_voxel, images)

        # Causal invariant: base_refiner receives the untouched full voxel.
        raw_depth, raw_points, _ = self.base_refiner(
            event_voxel=event_voxel,
            images=images,
            depth=depth,
            points=points,
        )
        min_depth = float(getattr(self.base_refiner, "min_depth", 1.0e-6))
        raw_delta_log = torch.log(
            raw_depth.float().clamp_min(min_depth) / depth.float().clamp_min(min_depth)
        ).squeeze(-1)
        gate = self.gate_floor + (1.0 - self.gate_floor) * reliability
        delta_log = raw_delta_log * gate.to(dtype=raw_delta_log.dtype)
        refined_depth = depth.float() * torch.exp(delta_log.unsqueeze(-1))
        refined_depth = refined_depth.to(dtype=depth.dtype).clamp_min(min_depth)
        depth_residual = refined_depth - depth
        refined_points = points
        if points is not None:
            ratio = refined_depth / depth.clamp_min(min_depth)
            refined_points = points * ratio.to(dtype=points.dtype)

        event_support = (event_voxel.detach().float().abs().sum(dim=2) > 0.0).to(reliability.dtype)
        self.last_reliability = reliability
        self.last_gate = gate
        self.last_delta_log = delta_log
        self.last_event_support = event_support
        # This diagnostic deliberately reports the original voxel, not a filtered surrogate.
        self.last_filtered_event_abs_mean = event_voxel.detach().abs().mean()
        return refined_depth, refined_points, depth_residual


class StreamVGGT(TemporalDetailStreamVGGT):
    def __init__(
        self,
        *args,
        reliability_checkpoint: str,
        reliability_base_channels: int = 32,
        reliability_gate_floor: float = 0.15,
        reliability_dilate_kernel: int = 3,
        reliability_frame_chunk_size: int = 1,
        reliability_rgb_input_range: str = "minus_one_one",
        residual_postfilter_kernel: int = 1,
        residual_postfilter_strength: float = 0.0,
        causal_output_gate: bool = False,
        causal_support_threshold: float = 0.0,
        causal_support_dilate_kernel: int = 1,
        causal_support_blur_kernel: int = 1,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        **kwargs,
    ) -> None:
        del (
            residual_postfilter_kernel,
            residual_postfilter_strength,
            causal_output_gate,
            causal_support_threshold,
            causal_support_dilate_kernel,
            causal_support_blur_kernel,
        )
        # Internal reliability must be disabled; only the external distilled map gates output.
        kwargs["reliability_gate_enabled"] = False
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            **kwargs,
        )
        self.event_detail_refiner = FrozenOutputReliabilityGate(
            self.event_detail_refiner,
            checkpoint=reliability_checkpoint,
            num_bins=event_num_bins,
            base_channels=reliability_base_channels,
            count_cmax=event_count_cmax,
            gate_floor=reliability_gate_floor,
            dilate_kernel=reliability_dilate_kernel,
            frame_chunk_size=reliability_frame_chunk_size,
            rgb_input_range=reliability_rgb_input_range,
        )

    def forward(self, views, query_points=None, **kwargs):
        output = super().forward(views, query_points=query_points, **kwargs)
        delta_log = self.event_detail_refiner.last_delta_log
        event_support = self.event_detail_refiner.last_event_support
        if delta_log is not None:
            for frame_index, result in enumerate(output.ress):
                result["event_delta_log"] = delta_log[:, frame_index]
                if event_support is not None:
                    result["event_support"] = event_support[:, frame_index]
        return output


__all__ = ["StreamVGGT", "FrozenOutputReliabilityGate"]
