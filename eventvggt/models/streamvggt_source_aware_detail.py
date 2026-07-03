"""VGGT detail refinement guided by a frozen three-source event decomposer.

The source network is pretrained with additive geometry/material/noise labels.
At VGGT training and inference it receives only RGB plus the full event voxel.
Only the predicted geometry token is allowed to drive the causal detail branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from eventvggt.models.streamvggt_additive_decomposition_detail import (
    AdditiveEventTokenDecomposer,
)
from eventvggt.models.streamvggt_causal_temporal_detail import (
    StreamVGGT as CausalTemporalDetailVGGT,
)


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state(checkpoint):
    state = checkpoint
    if isinstance(state, dict):
        for key in ("model", "state_dict", "module"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Source-aware checkpoint does not contain a state_dict")
    cleaned = {}
    prefixes = ("module.", "event_source_decomposer.", "event_branch_decomposer.")
    for name, value in state.items():
        name = str(name)
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


class StreamVGGT(CausalTemporalDetailVGGT):
    SOURCE_NAMES = ("geometry", "material", "noise")

    def __init__(
        self,
        *args,
        source_checkpoint: str,
        source_hidden_dim: int = 24,
        source_gate_floor: float = 0.05,
        source_frame_chunk_size: int = 1,
        source_ablation_mode: str = "learned",
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            event_num_bins=event_num_bins,
            event_count_cmax=event_count_cmax,
            **kwargs,
        )
        checkpoint_path = Path(source_checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Source-aware event checkpoint missing: {checkpoint_path}")
        self.event_source_decomposer = AdditiveEventTokenDecomposer(
            num_bins=event_num_bins,
            hidden_dim=int(source_hidden_dim),
            count_cmax=event_count_cmax,
        )
        state = _extract_state(_load_checkpoint(checkpoint_path))
        message = self.event_source_decomposer.load_state_dict(state, strict=True)
        if message.missing_keys or message.unexpected_keys:
            raise RuntimeError(f"Source-aware checkpoint mismatch: {message}")
        self.event_source_decomposer.requires_grad_(False)
        self.event_source_decomposer.eval()
        self.source_checkpoint = str(checkpoint_path)
        self.source_gate_floor = min(max(float(source_gate_floor), 0.0), 1.0)
        self.source_frame_chunk_size = max(int(source_frame_chunk_size), 1)
        self.source_ablation_mode = str(source_ablation_mode).lower()
        if self.source_ablation_mode not in {"learned", "uniform", "material_as_geometry"}:
            raise ValueError(
                "source_ablation_mode must be learned, uniform, or material_as_geometry"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.event_source_decomposer.eval()
        return self

    @staticmethod
    def _source_rgb(images: torch.Tensor) -> torch.Tensor:
        # finetune_event converts model RGB to [0,1], while source pretraining
        # uses the dataset-normalized [-1,1] range.
        if float(images.detach().amin()) >= -0.05:
            return images * 2.0 - 1.0
        return images

    @torch.no_grad()
    def _predict_sources(self, full_voxel: torch.Tensor, images: torch.Tensor):
        batch, seq_len = full_voxel.shape[:2]
        flat_voxel = full_voxel.reshape(batch * seq_len, *full_voxel.shape[2:])
        flat_images = images.reshape(batch * seq_len, *images.shape[2:])
        branches = []
        for start in range(0, flat_voxel.shape[0], self.source_frame_chunk_size):
            end = min(start + self.source_frame_chunk_size, flat_voxel.shape[0])
            with torch.autocast(device_type=full_voxel.device.type, enabled=False):
                branch, _ = self.event_source_decomposer(
                    flat_voxel[start:end].float().unsqueeze(1),
                    self._source_rgb(flat_images[start:end].float()).unsqueeze(1),
                )
            branches.append(branch.squeeze(1))
        branch_voxels = torch.cat(branches, dim=0).reshape(
            batch, seq_len, 3, *branches[0].shape[2:]
        )
        return branch_voxels

    def forward(
        self,
        views,
        query_points: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not views or not all("event_voxel" in view for view in views):
            return super().forward(views, query_points=query_points, **kwargs)

        images = torch.stack([view["img"] for view in views], dim=1)
        full_voxel = torch.stack([view["event_voxel"] for view in views], dim=1).to(
            device=images.device, dtype=images.dtype
        )
        branch_voxels = self._predict_sources(full_voxel, images)
        if self.source_ablation_mode == "uniform":
            # Parameter-count and compute-matched placebo: run the source net
            # but replace its decision with an uninformative uniform split.
            branch_voxels = full_voxel.unsqueeze(2).expand(-1, -1, 3, -1, -1, -1) / 3.0
        elif self.source_ablation_mode == "material_as_geometry":
            branch_voxels = branch_voxels[:, :, [1, 0, 2]]
        geometry_voxel = branch_voxels[:, :, 0].to(dtype=full_voxel.dtype)
        if self.source_gate_floor > 0.0:
            rejected = (full_voxel - geometry_voxel).clamp_min(0.0)
            geometry_voxel = geometry_voxel + self.source_gate_floor * rejected

        geometry_views = []
        for view_index, view in enumerate(views):
            geometry_view = dict(view)
            geometry_view["event_voxel"] = geometry_voxel[:, view_index]
            geometry_views.append(geometry_view)

        output = super().forward(
            geometry_views,
            query_points=query_points,
            **kwargs,
        )
        source_energy = branch_voxels.clamp_min(0.0).sum(dim=3)
        source_probability = source_energy / source_energy.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        for view_index, result in enumerate(output.ress):
            result["pred_event_source_probability"] = source_probability[:, view_index]
            result["pred_event_geometry_probability"] = source_probability[:, view_index, 0]
        output.views = geometry_views
        return output


__all__ = ["StreamVGGT"]
