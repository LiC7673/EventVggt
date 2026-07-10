"""One Stage-2 architecture with controlled contribution-map sources."""

from __future__ import annotations

import torch

from stage2_geometry_adapter.model import StreamVGGT as GeometryAdapterStreamVGGT


class AblationStreamVGGT(GeometryAdapterStreamVGGT):
    MODES = {"rgb_only", "raw_event", "ours", "no_multildr", "saturation_mask"}

    def __init__(self, *args, ablation_method: str = "ours", saturation_threshold: float = 0.98, **kwargs):
        super().__init__(
            *args,
            allow_non_bridge_stage1=(ablation_method == "no_multildr"),
            **kwargs,
        )
        if ablation_method not in self.MODES:
            raise ValueError(f"Unknown ablation method: {ablation_method!r}")
        self.ablation_method = ablation_method
        self.saturation_threshold = float(saturation_threshold)
        if ablation_method in {"rgb_only", "raw_event", "saturation_mask"}:
            self.contribution_net.requires_grad_(False).eval()

    def _fixed_contribution(self, views):
        images, events, _ = self._stack_inputs(views)
        shape = events.shape[:2] + events.shape[-2:]
        if self.ablation_method == "raw_event":
            return torch.ones(shape, device=events.device, dtype=events.dtype)
        if self.ablation_method == "rgb_only":
            return torch.zeros(shape, device=events.device, dtype=events.dtype)
        if self.ablation_method == "saturation_mask":
            return (images.float().amax(dim=2) >= self.saturation_threshold).to(events.dtype)
        return None

    def forward(self, views, query_points=None, contribution_override=None, **kwargs):
        fixed = self._fixed_contribution(views)
        if contribution_override is not None and fixed is not None:
            raise ValueError("External and method-fixed contribution overrides cannot be combined")
        return super().forward(
            views,
            query_points=query_points,
            contribution_override=fixed if fixed is not None else contribution_override,
            **kwargs,
        )


__all__ = ["AblationStreamVGGT"]
