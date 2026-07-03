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
    nested = {}
    nested_prefixes = (
        "event_detail_refiner.reliability_net.",
        "event_detail_refiner.external_reliability_net.",
    )
    for key, value in state.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        for prefix in nested_prefixes:
            if name.startswith(prefix):
                nested[name[len(prefix) :]] = value
                break
    if nested:
        return nested

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
        causal_output_gate: bool = False,
        causal_support_threshold: float = 0.01,
        causal_support_dilate_kernel: int = 5,
        causal_support_blur_kernel: int = 3,
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
        self.causal_output_gate = bool(causal_output_gate)
        self.causal_support_threshold = max(float(causal_support_threshold), 0.0)
        self.causal_support_dilate_kernel = max(int(causal_support_dilate_kernel), 1)
        self.causal_support_blur_kernel = max(int(causal_support_blur_kernel), 1)
        if self.causal_support_dilate_kernel % 2 == 0:
            self.causal_support_dilate_kernel += 1
        if self.causal_support_blur_kernel % 2 == 0:
            self.causal_support_blur_kernel += 1

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
        self.last_delta_log: Optional[torch.Tensor] = None
        self.last_event_support: Optional[torch.Tensor] = None

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

    def _event_support(self, event_voxel: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _, height, width = event_voxel.shape
        activity = event_voxel.detach().float().abs().sum(dim=2, keepdim=True)
        peak = activity.flatten(3).amax(dim=-1, keepdim=True).view(batch, seq_len, 1, 1, 1)
        normalized = activity / peak.clamp_min(1.0e-6)
        support = (normalized >= self.causal_support_threshold).to(dtype=activity.dtype)
        support = support.reshape(batch * seq_len, 1, height, width)
        if self.causal_support_dilate_kernel > 1:
            kernel = self.causal_support_dilate_kernel
            support = F.max_pool2d(support, kernel, stride=1, padding=kernel // 2)
        if self.causal_support_blur_kernel > 1:
            kernel = self.causal_support_blur_kernel
            support = F.avg_pool2d(support, kernel, stride=1, padding=kernel // 2)
        return support.clamp(0.0, 1.0).reshape(batch, seq_len, height, width)

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
        event_support = self._event_support(event_voxel)
        if self.causal_output_gate:
            # This is the causal constraint: without observed events, the
            # detail branch cannot alter RGB coarse geometry. Reliability has
            # already filtered the event input; support only enforces where a
            # residual is allowed to be written.
            min_depth = float(getattr(self.base_refiner, "min_depth", 1.0e-6))
            delta_log = torch.log(
                refined_depth.float().clamp_min(min_depth) / depth.float().clamp_min(min_depth)
            ).squeeze(-1)
            delta_log = delta_log * event_support.to(dtype=delta_log.dtype)
            refined_depth = depth.float() * torch.exp(delta_log.unsqueeze(-1))
            refined_depth = refined_depth.to(dtype=depth.dtype).clamp_min(min_depth)
            depth_residual = refined_depth - depth
            if points is not None:
                ratio = refined_depth / depth.clamp_min(min_depth)
                refined_points = points * ratio.to(dtype=points.dtype)
        self.last_reliability = reliability
        self.last_gate = (
            gate * event_support.to(dtype=gate.dtype) if self.causal_output_gate else gate
        )
        self.last_filtered_event_abs_mean = filtered_event.detach().abs().mean()
        self.last_event_support = event_support
        self.last_delta_log = torch.log(
            refined_depth.float().clamp_min(1.0e-6) / depth.float().clamp_min(1.0e-6)
        ).squeeze(-1)
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
        causal_output_gate: bool = False,
        causal_support_threshold: float = 0.01,
        causal_support_dilate_kernel: int = 5,
        causal_support_blur_kernel: int = 3,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        forward_batch_chunk: int = 1,
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
            causal_output_gate=causal_output_gate,
            causal_support_threshold=causal_support_threshold,
            causal_support_dilate_kernel=causal_support_dilate_kernel,
            causal_support_blur_kernel=causal_support_blur_kernel,
        )
        self.forward_batch_chunk = max(int(forward_batch_chunk), 0)

    def _forward_chunk(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        delta_log = self.event_detail_refiner.last_delta_log
        event_support = self.event_detail_refiner.last_event_support
        if delta_log is not None:
            for frame_idx, result in enumerate(output.ress):
                result["event_delta_log"] = delta_log[:, frame_idx]
                if event_support is not None:
                    result["event_support"] = event_support[:, frame_idx]
        return output

    def forward(self, views, query_points=None, **kwargs):
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
    def _slice_views(views, start, end, batch):
        sliced = []
        for view in views:
            item = {}
            for key, value in view.items():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
                    item[key] = value[start:end]
                elif isinstance(value, (list, tuple)) and len(value) == batch:
                    item[key] = value[start:end]
                else:
                    item[key] = value
            sliced.append(item)
        return sliced

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


__all__ = ["StreamVGGT", "StreamVGGTOutput", "FrozenReliabilityEventFilter"]
