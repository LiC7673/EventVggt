"""Temporal-detail StreamVGGT gated by a separately pretrained ReliabilityNet."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from eventvggt.models.streamvggt_temporal_detail import StreamVGGT as TemporalDetailStreamVGGT
from reliability_pretrain.model import ReliabilityUNet


class StreamVGGT(TemporalDetailStreamVGGT):
    def __init__(
        self,
        *args,
        reliability_checkpoint: str,
        reliability_num_bins: int = 5,
        reliability_base_channels: int = 32,
        reliability_count_cmax: float = 3.0,
        freeze_reliability: bool = True,
        **kwargs,
    ) -> None:
        # The external ReliabilityNet is the only gate. Disable the old internal gate.
        kwargs["reliability_gate_enabled"] = False
        super().__init__(*args, **kwargs)
        self.reliability_num_bins = max(1, int(reliability_num_bins))
        self.reliability_count_cmax = max(1.0, float(reliability_count_cmax))
        self.freeze_reliability = bool(freeze_reliability)
        self.reliability_net = ReliabilityUNet(
            event_channels=2 * self.reliability_num_bins,
            base_channels=int(reliability_base_channels),
        )
        self._load_reliability_checkpoint(reliability_checkpoint)

    def _load_reliability_checkpoint(self, checkpoint: str) -> None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Stage-1 ReliabilityNet checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu")
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        self.reliability_net.load_state_dict(state, strict=True)

    def _prepare_reliability_voxel(self, voxel: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, height, width = voxel.shape
        source_bins = channels // 2
        if source_bins <= 0:
            raise ValueError("ReliabilityNet requires polarity-separated event voxel channels.")
        pos = voxel[:, :, :source_bins].clamp_min(0.0)
        neg = voxel[:, :, source_bins : 2 * source_bins].clamp_min(0.0)
        if source_bins != self.reliability_num_bins:
            def resize_time(value: torch.Tensor) -> torch.Tensor:
                resized = F.interpolate(
                    value.reshape(batch * seq_len, 1, source_bins, height, width),
                    size=(self.reliability_num_bins, height, width),
                    mode="trilinear",
                    align_corners=False,
                )
                resized = resized * (float(source_bins) / float(self.reliability_num_bins))
                return resized.reshape(batch, seq_len, self.reliability_num_bins, height, width)

            pos = resize_time(pos)
            neg = resize_time(neg)
        norm = torch.log1p(voxel.new_tensor(self.reliability_count_cmax))
        pos = torch.log1p(pos.clamp_max(self.reliability_count_cmax)) / norm
        neg = torch.log1p(neg.clamp_max(self.reliability_count_cmax)) / norm
        return torch.cat([pos, neg], dim=2)

    def forward(self, views, query_points: Optional[torch.Tensor] = None, **kwargs):
        if not all("event_voxel" in view for view in views):
            return super().forward(views, query_points=query_points, **kwargs)

        images = torch.stack([view["img"] for view in views], dim=1)
        event_voxel = torch.stack([view["event_voxel"] for view in views], dim=1).to(images.device)
        prepared = self._prepare_reliability_voxel(event_voxel.to(dtype=images.dtype))
        batch, seq_len, channels, height, width = prepared.shape
        flat_event = prepared.reshape(batch * seq_len, channels, height, width)
        flat_rgb = images.reshape(batch * seq_len, 3, height, width)

        if self.freeze_reliability:
            with torch.no_grad():
                reliability = self.reliability_net(flat_event, flat_rgb)
            reliability = reliability.detach()
        else:
            reliability = self.reliability_net(flat_event, flat_rgb)
        reliability = reliability.reshape(batch, seq_len, height, width)

        # R_geo gates every polarity/time channel without collapsing temporal bins.
        gated_voxel = event_voxel * reliability.unsqueeze(2).to(dtype=event_voxel.dtype)
        gated_views = []
        for view_idx, view in enumerate(views):
            gated_view = dict(view)
            gated_view["event_voxel"] = gated_voxel[:, view_idx]
            gated_view["event_reliability_pred"] = reliability[:, view_idx]
            gated_views.append(gated_view)

        output = super().forward(gated_views, query_points=query_points, **kwargs)
        for view_idx, result in enumerate(output.ress):
            result["event_reliability"] = reliability[:, view_idx]
            result["event_gate"] = reliability[:, view_idx]
        output.views = gated_views
        return output


__all__ = ["StreamVGGT"]

