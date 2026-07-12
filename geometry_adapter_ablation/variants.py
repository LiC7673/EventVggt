from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.unified_model import UnifiedGeometryContributionModel


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _AdapterBase(nn.Module):
    def __init__(self, rgb_channels: int, event_channels: int, hidden_channels: int, *, event_only: bool):
        super().__init__()
        hidden = max(min(int(hidden_channels), int(rgb_channels)), 32)
        in_channels = event_channels if event_only else rgb_channels + event_channels
        self.event_only = bool(event_only)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=_group_count(hidden)),
            nn.GELU(),
            nn.Conv2d(hidden, rgb_channels, 1),
        )
        nn.init.normal_(self.adapter[-1].weight, std=1.0e-2)
        nn.init.zeros_(self.adapter[-1].bias)
        self.alpha_logit = nn.Parameter(torch.zeros(()))

    def raw(self, rgb_feature: torch.Tensor, event_feature: torch.Tensor) -> torch.Tensor:
        inputs = event_feature if self.event_only else torch.cat((rgb_feature, event_feature), dim=1)
        return torch.tanh(self.alpha_logit) * self.adapter(inputs)


class EventOnlyHardAdapter(_AdapterBase):
    """Experiment A: event features are the only source of an update."""

    def __init__(self, rgb_channels: int, event_channels: int, hidden_channels: int):
        super().__init__(rgb_channels, event_channels, hidden_channels, event_only=True)

    def forward(self, rgb_feature, event_feature, contribution):
        raw_update = self.raw(rgb_feature, event_feature)
        support = (contribution > 1.0e-6).to(raw_update.dtype)
        update = support * raw_update
        penalty = ((1.0 - contribution) * raw_update).abs().mean()
        return rgb_feature + update, update, penalty


class SoftContributionAdapter(_AdapterBase):
    """Experiment B/C: retain a continuous gate instead of thresholding it."""

    def __init__(self, rgb_channels: int, event_channels: int, hidden_channels: int):
        super().__init__(rgb_channels, event_channels, hidden_channels, event_only=False)

    def forward(self, rgb_feature, event_feature, contribution):
        raw_update = self.raw(rgb_feature, event_feature)
        gate = contribution.clamp(0.0, 1.0).to(raw_update.dtype)
        update = gate * raw_update
        penalty = ((1.0 - gate) * raw_update).abs().mean()
        return rgb_feature + update, update, penalty


class DisabledAdapter(nn.Module):
    """Return an exact zero update while keeping the DPT call contract."""

    def __init__(self):
        super().__init__()
        self.register_buffer("alpha_logit", torch.zeros(()), persistent=False)

    def forward(self, rgb_feature, event_feature, contribution):
        update = torch.zeros_like(rgb_feature)
        return rgb_feature, update, rgb_feature.new_zeros(())


class ExplicitSupportPyramid(nn.Module):
    """Experiment C: explicitly multiply the soft gate by raw-event support."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @property
    def num_bins(self):
        return self.encoder.num_bins

    @property
    def count_cmax(self):
        return self.encoder.count_cmax

    def forward(self, selected_event, contribution, target_shapes):
        event_pyramid, contribution_pyramid = self.encoder(
            selected_event, contribution, target_shapes
        )
        batch, views, _, height, width = selected_event.shape
        support = (selected_event.abs().sum(dim=2, keepdim=True) > 0.0).float()
        support = support.reshape(batch * views, 1, height, width)
        constrained = []
        for gate, shape in zip(contribution_pyramid, target_shapes):
            support_level = F.interpolate(
                support, size=shape, mode="bilinear", align_corners=False
            ).reshape(batch, views, 1, *shape)
            constrained.append(gate * support_level)
        return event_pyramid, constrained


def _adapter_dimensions(adapter: nn.Module) -> tuple[int, int, int]:
    first = adapter.adapter[0]
    last = adapter.adapter[-1]
    rgb_channels = int(last.out_channels)
    event_channels = int(first.in_channels) - rgb_channels
    hidden_channels = int(first.out_channels)
    return rgb_channels, event_channels, hidden_channels


def _replace_all_adapters(model: nn.Module, adapter_type: Type[nn.Module]) -> None:
    for head in (model.depth_head, model.point_head):
        replacements = []
        for old in head.geometry_adapters:
            replacements.append(adapter_type(*_adapter_dimensions(old)))
        head.geometry_adapters = nn.ModuleList(replacements)


def _disable_low_resolution_levels(model: nn.Module) -> None:
    # DPT indices 0/1 are the two high-resolution resize paths. Indices 2/3
    # remain present in the call graph but write an exact zero update.
    for head in (model.depth_head, model.point_head):
        adapters = list(head.geometry_adapters)
        adapters[2] = DisabledAdapter()
        adapters[3] = DisabledAdapter()
        head.geometry_adapters = nn.ModuleList(adapters)


def make_variant_model(variant: str):
    variant = str(variant).strip().lower()
    valid = {"remove_rgb_shortcut", "soft_gating", "support_constrained", "high_resolution"}
    if variant not in valid:
        raise ValueError(f"Unknown GeometryAdapter ablation {variant!r}; expected {sorted(valid)}")

    class VariantModel(UnifiedGeometryContributionModel):
        checkpoint_schema = f"unified_geometry_contribution_ablation_{variant}_v1"
        ablation_variant = variant

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if variant == "remove_rgb_shortcut":
                _replace_all_adapters(self, EventOnlyHardAdapter)
            elif variant == "soft_gating":
                _replace_all_adapters(self, SoftContributionAdapter)
            elif variant == "support_constrained":
                _replace_all_adapters(self, SoftContributionAdapter)
                self.event_encoder = ExplicitSupportPyramid(self.event_encoder)
            elif variant == "high_resolution":
                _disable_low_resolution_levels(self)

    VariantModel.__name__ = "GeometryAdapterAblationModel"
    return VariantModel


__all__ = ["make_variant_model"]

