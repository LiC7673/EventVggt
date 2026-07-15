"""No-contribution event-source ablation: oracle E_geo versus raw E_full."""
from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_dual_alignment_hdr_model import (
    DualAlignmentHDRLinearVoxelModel,
)


class ConstantOneContribution(nn.Module):
    """Parameter-free removal of ContributionNet: every event has C=1."""

    coarse_feature_dim = 0

    def forward(self, event, rgb, coarse_depth, coarse_normal):
        del event, rgb, coarse_normal
        return torch.ones_like(coarse_depth, dtype=torch.float32)


class IdentityFeatureAlignment(nn.Module):
    """E_geo is already the oracle source and needs no full-to-geo adapter."""

    def __init__(self):
        super().__init__()
        self.reliability = nn.Identity()

    def forward(self, feature):
        zero = torch.zeros_like(feature)
        return feature, zero, feature[:, :1] * 0.0


class NoContributionSourceAblationModel(DualAlignmentHDRLinearVoxelModel):
    checkpoint_schema = "linear_voxel_no_contribution_event_source_ablation_v1"

    def __init__(self, *args, event_source="full", **kwargs):
        super().__init__(*args, **kwargs)
        source = str(event_source).strip().lower()
        if source not in {"full", "geo"}:
            raise ValueError(f"event_source must be 'full' or 'geo', got {event_source!r}")
        self.event_source = source
        # The reliability module is structurally absent in this ablation.
        self.contribution_net = ConstantOneContribution()
        # Keep both arms architecturally identical: neither arm receives a
        # full-to-geo alignment adapter.  Event source is the only variable.
        self.full_geo_aligner = IdentityFeatureAlignment()

    def _select_event_source(self, views):
        if self.event_source == "full":
            return views
        selected = []
        for index, view in enumerate(views):
            if bool(view.get("event_source_preselected", False)):
                selected.append(view)
                continue
            geo = view.get("geometry_event_voxel")
            if not torch.is_tensor(geo):
                raise RuntimeError(
                    "The oracle E_geo ablation requires geometry_event_voxel at "
                    f"train and test time (missing view {index})."
                )
            current = dict(view)
            current["event_voxel"] = geo
            current["event_source_preselected"] = True
            selected.append(current)
        return selected

    def forward(self, views, *args, **kwargs):
        output = super().forward(self._select_event_source(views), *args, **kwargs)
        for item in output.ress:
            item["event_source_is_geo"] = item["depth"].new_tensor(
                float(self.event_source == "geo")
            )
        return output


__all__ = [
    "ConstantOneContribution",
    "IdentityFeatureAlignment",
    "NoContributionSourceAblationModel",
]
