"""Reliability-ratio supervision layered on the geometry-detail objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import finetune_event as fe
from mul_loss_fine.launcher import make_configured_loss


def make_staged_reliability_loss(cfg):
    base_loss = make_configured_loss(cfg)

    class StagedReliabilityLoss(base_loss):
        def forward(self, model_output, views):
            total, details, aux = super().forward(model_output, views)
            reliability = torch.stack(
                [result["event_reliability"] for result in model_output.ress], dim=1
            )
            target = fe.stack_view_field(views, "event_reliability_gt").to(
                device=reliability.device, dtype=reliability.dtype
            )
            presence = fe.stack_view_field(views, "event_full_presence").to(
                device=reliability.device, dtype=reliability.dtype
            )
            weight = 0.05 + 0.95 * presence
            denominator = weight.sum().clamp_min(1.0)
            bce = (
                F.binary_cross_entropy(
                    reliability.clamp(1.0e-5, 1.0 - 1.0e-5), target, reduction="none"
                )
                * weight
            ).sum() / denominator
            l1 = ((reliability - target).abs() * weight).sum() / denominator
            reliability_loss = float(getattr(cfg.loss, "staged_reliability_weight", 0.0)) * (
                bce + 0.5 * l1
            )
            total = total + reliability_loss
            previous_extra = float(details.get("extra_loss_total", 0.0))
            details.update(
                {
                    "staged_reliability_loss": float(reliability_loss.detach()),
                    "staged_reliability_bce": float(bce.detach()),
                    "staged_reliability_l1": float(l1.detach()),
                    "staged_reliability_pred_mean": float(reliability.detach().mean()),
                    "staged_reliability_target_mean": float(target.detach().mean()),
                    "staged_event_weight_mean": float((reliability.detach() * presence).mean()),
                    "extra_loss_total": previous_extra + float(reliability_loss.detach()),
                    "total_loss_with_extra": float(total.detach()),
                }
            )
            aux["event_reliability"] = reliability.detach()
            aux["event_reliability_gt"] = target.detach()
            return total, details, aux

    return StagedReliabilityLoss
