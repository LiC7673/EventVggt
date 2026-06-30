"""Full-img reliability driven by a frozen additive event decomposer.

Stage 1 predicts geometry/material/noise event streams from full events and
RGB. Stage 2 freezes that decomposition network and feeds its predicted
geometry stream into the unchanged temporal-detail refinement path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from eventvggt.models.streamvggt_additive_decomposition_detail import (
    AdditiveEventTokenDecomposer,
)
from eventvggt.models.streamvggt_stable_temporal_detail import (
    StreamVGGT as TemporalDetailStreamVGGT,
)


class StreamVGGT(TemporalDetailStreamVGGT):
    def __init__(
        self,
        *args,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        decomposition_hidden_dim: int = 24,
        decomposition_checkpoint: str,
        geometry_event_floor: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            **kwargs,
        )
        self.geometry_event_floor = min(max(float(geometry_event_floor), 0.0), 1.0)
        self.event_branch_decomposer = AdditiveEventTokenDecomposer(
            num_bins=event_num_bins,
            hidden_dim=decomposition_hidden_dim,
            count_cmax=event_count_cmax,
        )
        self._load_decomposer(decomposition_checkpoint)
        self.event_branch_decomposer.requires_grad_(False)
        self.event_branch_decomposer.eval()

    def _load_decomposer(self, checkpoint: str) -> None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Stage-1 additive decomposer checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu")
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        self.event_branch_decomposer.load_state_dict(state, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Stage 2 must never update BN/dropout-like state in the frozen teacher.
        self.event_branch_decomposer.eval()
        return self

    def forward(self, views, query_points: Optional[torch.Tensor] = None, **kwargs):
        if not all("event_voxel" in view for view in views):
            return super().forward(views, query_points=query_points, **kwargs)

        images = torch.stack([view["img"] for view in views], dim=1)
        full_voxel = torch.stack([view["event_voxel"] for view in views], dim=1).to(images.device)
        with torch.no_grad():
            branches, probabilities = self.event_branch_decomposer(
                full_voxel.to(dtype=images.dtype), images
            )
            predicted_geometry = branches[:, :, 0].to(dtype=full_voxel.dtype)
            filtered_geometry = (
                self.geometry_event_floor * full_voxel
                + (1.0 - self.geometry_event_floor) * predicted_geometry
            )

        filtered_views = []
        for view_idx, view in enumerate(views):
            filtered_view = dict(view)
            filtered_view["event_voxel"] = filtered_geometry[:, view_idx]
            filtered_views.append(filtered_view)

        output = super().forward(filtered_views, query_points=query_points, **kwargs)
        for view_idx, result in enumerate(output.ress):
            result["stage1_geometry_event"] = predicted_geometry[:, view_idx]
            result["stage1_filtered_event"] = filtered_geometry[:, view_idx]
            result["stage1_geometry_probability"] = probabilities[:, view_idx, 0]
        output.views = filtered_views
        return output


__all__ = ["StreamVGGT"]
