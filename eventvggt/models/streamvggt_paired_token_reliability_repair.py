"""Repair variant for frozen paired-token reliability refinement.

The first paired-token reliability model was too permissive: most pixels were
classified as reliable while the event residual stayed very small.  This module
keeps the same frozen ReliabilityUNet, but calibrates its output before it gates
the log-depth correction:

* sharpen reliability with a threshold and temperature;
* require local event support, optionally dilated;
* optionally keep only the top fraction of reliable pixels per frame;
* allow a modest residual gain before clamping.

The full event voxel still reaches the temporal refiner.  Only the final
event-conditioned correction is gated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from eventvggt.models.streamvggt_paired_token_reliability_detail import (
    FrozenOutputReliabilityGate,
    StreamVGGT as PairedTokenReliabilityStreamVGGT,
)


def _odd_kernel(value: int) -> int:
    value = max(int(value), 1)
    return value + 1 if value % 2 == 0 else value


class RepairedFrozenOutputReliabilityGate(FrozenOutputReliabilityGate):
    def __init__(
        self,
        *args,
        reliability_threshold: float = 0.58,
        reliability_temperature: float = 0.12,
        reliability_top_fraction: float = 0.35,
        event_support_threshold: float = 0.0,
        event_support_dilate_kernel: int = 5,
        event_support_floor: float = 0.05,
        residual_gain: float = 1.6,
        output_abs_limit: float = 0.06,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reliability_threshold = float(reliability_threshold)
        self.reliability_temperature = max(float(reliability_temperature), 1.0e-3)
        self.reliability_top_fraction = float(reliability_top_fraction)
        self.event_support_threshold = float(event_support_threshold)
        self.event_support_dilate_kernel = _odd_kernel(event_support_dilate_kernel)
        self.event_support_floor = min(max(float(event_support_floor), 0.0), 1.0)
        self.residual_gain = float(residual_gain)
        self.output_abs_limit = float(output_abs_limit)
        self.last_raw_reliability: Optional[torch.Tensor] = None
        self.last_support_gate: Optional[torch.Tensor] = None

    def _event_support_gate(self, event_voxel: torch.Tensor) -> torch.Tensor:
        support = event_voxel.detach().float().abs().sum(dim=2)
        support = (support > self.event_support_threshold).to(event_voxel.dtype)
        if self.event_support_dilate_kernel > 1:
            flat = support.flatten(0, 1).unsqueeze(1)
            support = F.max_pool2d(
                flat,
                kernel_size=self.event_support_dilate_kernel,
                stride=1,
                padding=self.event_support_dilate_kernel // 2,
            ).squeeze(1).view_as(support)
        return self.event_support_floor + (1.0 - self.event_support_floor) * support

    def _top_fraction_mask(self, reliability: torch.Tensor) -> torch.Tensor:
        fraction = self.reliability_top_fraction
        if fraction <= 0.0 or fraction >= 1.0:
            return torch.ones_like(reliability)
        flat = reliability.flatten(2)
        k = max(int(flat.shape[-1] * fraction), 1)
        threshold = torch.topk(flat, k, dim=-1).values[..., -1:].unsqueeze(-1)
        return (reliability >= threshold).to(reliability.dtype)

    @torch.no_grad()
    def _predict_reliability(self, event_voxel: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        raw = super()._predict_reliability(event_voxel, images)
        sharpened = torch.sigmoid(
            (raw - self.reliability_threshold) / self.reliability_temperature
        )
        support_gate = self._event_support_gate(event_voxel).to(device=sharpened.device, dtype=sharpened.dtype)
        reliability = sharpened * support_gate
        reliability = reliability * self._top_fraction_mask(reliability)
        self.last_raw_reliability = raw
        self.last_support_gate = support_gate
        return reliability.clamp(0.0, 1.0)

    def forward(
        self,
        *,
        event_voxel: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        points: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        reliability = self._predict_reliability(event_voxel, images)
        raw_depth, raw_points, _ = self.base_refiner(
            event_voxel=event_voxel,
            images=images,
            depth=depth,
            points=points,
        )
        del raw_points
        min_depth = float(getattr(self.base_refiner, "min_depth", 1.0e-6))
        raw_delta_log = torch.log(
            raw_depth.float().clamp_min(min_depth) / depth.float().clamp_min(min_depth)
        ).squeeze(-1)
        gate = self.gate_floor + (1.0 - self.gate_floor) * reliability
        delta_log = raw_delta_log * gate.to(dtype=raw_delta_log.dtype) * self.residual_gain
        if self.output_abs_limit > 0.0:
            delta_log = delta_log.clamp(-self.output_abs_limit, self.output_abs_limit)
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
        self.last_filtered_event_abs_mean = (event_voxel.detach().abs() * reliability.unsqueeze(2)).mean()
        return refined_depth, refined_points, depth_residual


class StreamVGGT(PairedTokenReliabilityStreamVGGT):
    def __init__(
        self,
        *args,
        repair_reliability_threshold: float = 0.58,
        repair_reliability_temperature: float = 0.12,
        repair_reliability_top_fraction: float = 0.35,
        repair_event_support_threshold: float = 0.0,
        repair_event_support_dilate_kernel: int = 5,
        repair_event_support_floor: float = 0.05,
        repair_residual_gain: float = 1.6,
        repair_output_abs_limit: float = 0.06,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        old_gate = self.event_detail_refiner
        self.event_detail_refiner = RepairedFrozenOutputReliabilityGate(
            old_gate.base_refiner,
            checkpoint=old_gate.reliability_checkpoint,
            num_bins=old_gate.num_bins,
            base_channels=old_gate.reliability_net.enc1.net[0].out_channels,
            count_cmax=old_gate.count_cmax,
            gate_floor=old_gate.gate_floor,
            dilate_kernel=1,
            frame_chunk_size=old_gate.frame_chunk_size,
            rgb_input_range=old_gate.rgb_input_range,
            reliability_threshold=repair_reliability_threshold,
            reliability_temperature=repair_reliability_temperature,
            reliability_top_fraction=repair_reliability_top_fraction,
            event_support_threshold=repair_event_support_threshold,
            event_support_dilate_kernel=repair_event_support_dilate_kernel,
            event_support_floor=repair_event_support_floor,
            residual_gain=repair_residual_gain,
            output_abs_limit=repair_output_abs_limit,
        )


__all__ = ["StreamVGGT", "RepairedFrozenOutputReliabilityGate"]
