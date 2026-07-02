"""Temporal-detail VGGT filtered by a frozen pretrained ReliabilityNet.

The external reliability network sees the original polarity-separated temporal
voxel and RGB image. Its soft prediction filters the event voxel before the
dense temporal-detail refiner, while a non-zero floor avoids deleting all event
evidence when the reliability estimate is conservative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eventvggt.models.streamvggt_temporal_detail import (
    StreamVGGT as TemporalDetailStreamVGGT,
    StreamVGGTOutput,
)
from reliability_pretrain.model import ReliabilityUNet


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unwrap_reliability_state(checkpoint):
    state = checkpoint
    if isinstance(state, dict):
        for key in ("model", "state_dict", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Reliability checkpoint does not contain a state_dict.")
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "reliability_net.", "external_reliability_net."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


class FrozenReliabilityEventFilter(nn.Module):
    """Wrap a temporal-detail refiner with a frozen full-resolution event gate."""

    def __init__(
        self,
        base_refiner: nn.Module,
        *,
        checkpoint: str,
        num_bins: int = 10,
        base_channels: int = 32,
        count_cmax: float = 3.0,
        gate_floor: float = 0.10,
        frame_chunk_size: int = 1,
        rgb_input_range: str = "minus_one_one",
        residual_postfilter_kernel: int = 3,
        residual_postfilter_strength: float = 0.75,
    ) -> None:
        super().__init__()
        self.base_refiner = base_refiner
        self.num_bins = max(int(num_bins), 1)
        self.count_cmax = max(float(count_cmax), 1.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), 1.0)
        self.frame_chunk_size = max(int(frame_chunk_size), 1)
        self.rgb_input_range = str(rgb_input_range)
        self.residual_postfilter_kernel = max(int(residual_postfilter_kernel), 1)
        if self.residual_postfilter_kernel % 2 == 0:
            self.residual_postfilter_kernel += 1
        self.residual_postfilter_strength = min(
            max(float(residual_postfilter_strength), 0.0), 1.0
        )

        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Frozen ReliabilityNet checkpoint not found: {checkpoint_path}. "
                "Run real_reliability_stage/run_two_stage_real_reliability.sh first."
            )
        self.reliability_net = ReliabilityUNet(
            event_channels=2 * self.num_bins,
            image_channels=3,
            base_channels=int(base_channels),
        )
        state = _unwrap_reliability_state(_load_checkpoint(checkpoint_path))
        incompatible = self.reliability_net.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected ReliabilityNet checkpoint mismatch: {incompatible}")
        self.reliability_net.requires_grad_(False)
        self.reliability_net.eval()
        self.reliability_checkpoint = str(checkpoint_path)

        self.last_reliability: Optional[torch.Tensor] = None
        self.last_gate: Optional[torch.Tensor] = None
        self.last_filtered_event_abs_mean: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        super().train(mode)
        # The stage-1 reliability estimator must remain deterministic and
        # frozen while VGGT/detail parameters are optimized.
        self.reliability_net.eval()
        return self

    def _normalize_event(self, voxel: torch.Tensor) -> torch.Tensor:
        denominator = torch.log1p(voxel.new_tensor(self.count_cmax))
        return torch.log1p(voxel.clamp_min(0.0).clamp_max(self.count_cmax)) / denominator

    def _prepare_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.rgb_input_range == "minus_one_one":
            # finetune_event denormalizes views to [0,1] before model.forward,
            # whereas the standalone ReliabilityNet was trained on the
            # dataset's normalized [-1,1] images.
            if float(rgb.detach().amin()) >= -0.05:
                rgb = rgb * 2.0 - 1.0
        elif self.rgb_input_range == "zero_one" and float(rgb.detach().amin()) < -0.05:
            rgb = (rgb + 1.0) * 0.5
        return rgb

    @torch.no_grad()
    def _predict_reliability(self, event_voxel: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, height, width = event_voxel.shape
        expected_channels = 2 * self.num_bins
        if channels != expected_channels:
            raise ValueError(
                f"ReliabilityNet expects {expected_channels} event channels, got {channels}. "
                "Keep data.event_resize_bins equal to the stage-1 num_bins."
            )
        event_flat = event_voxel.reshape(batch * seq_len, channels, height, width)
        image_flat = images.reshape(batch * seq_len, 3, height, width)
        predictions = []
        for start in range(0, event_flat.shape[0], self.frame_chunk_size):
            end = min(start + self.frame_chunk_size, event_flat.shape[0])
            # Run the frozen U-Net in fp32. No activation graph is retained, so
            # frame-wise execution has a modest memory footprint.
            with torch.autocast(device_type=event_voxel.device.type, enabled=False):
                event_input = self._normalize_event(event_flat[start:end].float())
                rgb_input = self._prepare_rgb(image_flat[start:end].float())
                predictions.append(self.reliability_net(event_input, rgb_input))
        return torch.cat(predictions, dim=0).reshape(batch, seq_len, height, width)

    def forward(
        self,
        *,
        event_voxel: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        points: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        reliability = self._predict_reliability(event_voxel, images)
        gate = self.gate_floor + (1.0 - self.gate_floor) * reliability
        filtered_event = event_voxel * gate.unsqueeze(2).to(dtype=event_voxel.dtype)

        refined_depth, refined_points, depth_residual = self.base_refiner(
            event_voxel=filtered_event,
            images=images,
            depth=depth,
            points=points,
        )
        if self.residual_postfilter_kernel > 1 and self.residual_postfilter_strength > 0.0:
            min_depth = float(getattr(self.base_refiner, "min_depth", 1.0e-6))
            delta_log = torch.log(
                refined_depth.float().clamp_min(min_depth) / depth.float().clamp_min(min_depth)
            )
            batch, seq_len, height, width, _ = delta_log.shape
            delta_flat = delta_log.permute(0, 1, 4, 2, 3).reshape(
                batch * seq_len, 1, height, width
            )
            pad = self.residual_postfilter_kernel // 2
            smooth_delta = F.avg_pool2d(
                F.pad(delta_flat, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=self.residual_postfilter_kernel,
                stride=1,
            )
            strength = self.residual_postfilter_strength
            delta_flat = (1.0 - strength) * delta_flat + strength * smooth_delta
            delta_log = delta_flat.reshape(batch, seq_len, 1, height, width).permute(0, 1, 3, 4, 2)
            refined_depth = depth.float() * torch.exp(delta_log)
            refined_depth = refined_depth.to(dtype=depth.dtype).clamp_min(min_depth)
            depth_residual = refined_depth - depth
            if points is not None:
                ratio = refined_depth / depth.clamp_min(min_depth)
                refined_points = points * ratio.to(dtype=points.dtype)
        self.last_reliability = reliability
        self.last_gate = gate
        self.last_filtered_event_abs_mean = filtered_event.detach().abs().mean()
        return refined_depth, refined_points, depth_residual


class StreamVGGT(TemporalDetailStreamVGGT):
    """VGGT temporal-detail model using a frozen stage-1 reliability filter."""

    def __init__(
        self,
        *args,
        reliability_checkpoint: str,
        reliability_base_channels: int = 32,
        reliability_gate_floor: float = 0.10,
        reliability_frame_chunk_size: int = 1,
        reliability_rgb_input_range: str = "minus_one_one",
        residual_postfilter_kernel: int = 3,
        residual_postfilter_strength: float = 0.75,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        **kwargs,
    ) -> None:
        # Disable the temporal-detail branch's internal gate. The pretrained
        # external gate is the only reliability mechanism in this ablation.
        kwargs["reliability_gate_enabled"] = False
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            **kwargs,
        )
        self.event_detail_refiner = FrozenReliabilityEventFilter(
            self.event_detail_refiner,
            checkpoint=reliability_checkpoint,
            num_bins=event_num_bins,
            base_channels=reliability_base_channels,
            count_cmax=event_count_cmax,
            gate_floor=reliability_gate_floor,
            frame_chunk_size=reliability_frame_chunk_size,
            rgb_input_range=reliability_rgb_input_range,
            residual_postfilter_kernel=residual_postfilter_kernel,
            residual_postfilter_strength=residual_postfilter_strength,
        )


__all__ = ["StreamVGGT", "StreamVGGTOutput", "FrozenReliabilityEventFilter"]
