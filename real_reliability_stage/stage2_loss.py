"""Loss adapter that uses frozen reliability to weight event supervision."""

from __future__ import annotations

from typing import Dict, List

import torch

from mul_loss_fine.event_supported_mv_loss import _stack_output_field
from mul_loss_fine.launcher import make_configured_loss


class FrozenReliabilityWeightedEventLossMixin:
    """Replace event voxels used by event-aware losses with R.detach() * V."""

    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]):
        reliability = _stack_output_field(model_output, "event_reliability")
        model_gate = _stack_output_field(model_output, "event_gate")
        if reliability is None:
            total, details, aux = super().forward(model_output, views)
            details.update(
                {
                    "stage2_reliability_mean": 0.0,
                    "stage2_gate_mean": 0.0,
                    "stage2_reliability_positive_ratio": 0.0,
                    "stage2_weighted_event_abs_mean": 0.0,
                }
            )
            return total, details, aux

        reliability = reliability.detach()
        weighted_views = []
        weighted_event_sum = reliability.new_tensor(0.0)
        weighted_event_count = reliability.new_tensor(0.0)
        for view_idx, view in enumerate(views):
            weighted_view = dict(view)
            event_voxel = view.get("event_voxel")
            if torch.is_tensor(event_voxel) and event_voxel.numel() > 0:
                rel = reliability[:, view_idx].to(device=event_voxel.device, dtype=event_voxel.dtype)
                weighted_event = event_voxel * rel.unsqueeze(1)
                weighted_view["event_voxel"] = weighted_event
                weighted_event_sum = weighted_event_sum + weighted_event.detach().abs().sum()
                weighted_event_count = weighted_event_count + weighted_event.numel()
            weighted_views.append(weighted_view)

        total, details, aux = super().forward(model_output, weighted_views)
        details.update(
            {
                "stage2_reliability_mean": float(reliability.mean()),
                "stage2_gate_mean": float(model_gate.detach().mean()) if model_gate is not None else 0.0,
                "stage2_reliability_positive_ratio": float((reliability >= 0.5).float().mean()),
                "stage2_weighted_event_abs_mean": float(
                    (weighted_event_sum / weighted_event_count.clamp_min(1.0)).detach()
                ),
            }
        )
        aux["event_reliability"] = reliability
        if model_gate is not None:
            aux["event_gate"] = model_gate.detach()
        return total, details, aux


def make_stage2_reliability_weighted_loss(cfg):
    configured_base = make_configured_loss(cfg)

    class ConfiguredStage2ReliabilityLoss(
        FrozenReliabilityWeightedEventLossMixin,
        configured_base,
    ):
        pass

    return ConfiguredStage2ReliabilityLoss


__all__ = ["make_stage2_reliability_weighted_loss"]
