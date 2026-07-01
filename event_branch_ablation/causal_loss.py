"""Direct additive-branch reliability supervision for causal refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import finetune_event as fe
from event_branch_ablation.loss import make_additive_token_loss


def make_causal_additive_loss(cfg):
    base_loss = make_additive_token_loss(cfg)

    class CausalAdditiveReliabilityLoss(base_loss):
        def forward(self, model_output, views):
            total, details, aux = super().forward(model_output, views)
            required = (
                "event_voxel",
                "event_geometry_voxel",
                "event_material_voxel",
                "event_noise_voxel",
            )
            if not all(key in views[0] for key in required):
                details.update(
                    {
                        "causal_reliability_loss": 0.0,
                        "causal_partition_loss": 0.0,
                        "causal_reliability_mae": 0.0,
                        "causal_reliability_target_mean": 0.0,
                        "causal_reliability_pred_mean": 0.0,
                    }
                )
                return total, details, aux

            pred_geometry = aux["pred_event_geometry_token"]
            pred_material = aux["pred_event_material_token"]
            pred_noise = aux["pred_event_noise_token"]
            targets = [
                fe.stack_view_field(views, key).to(
                    device=pred_geometry.device, dtype=pred_geometry.dtype
                )
                for key in (
                    "event_geometry_voxel",
                    "event_material_voxel",
                    "event_noise_voxel",
                )
            ]
            full = fe.stack_view_field(views, "event_voxel").to(
                device=pred_geometry.device, dtype=pred_geometry.dtype
            ).clamp_min(0)
            full_energy = full.sum(dim=2, keepdim=True)
            target_geometry_energy = targets[0].clamp_min(0).sum(dim=2, keepdim=True)
            pred_geometry_energy = pred_geometry.clamp_min(0).sum(dim=2, keepdim=True)
            presence = (full_energy > 0).to(dtype=full.dtype)
            target_reliability = (target_geometry_energy / full_energy.clamp_min(1e-6)).clamp(0, 1)
            pred_reliability = (pred_geometry_energy / full_energy.clamp_min(1e-6)).clamp(0, 1)

            reliability_error = F.binary_cross_entropy(
                pred_reliability.clamp(1e-5, 1 - 1e-5),
                target_reliability,
                reduction="none",
            )
            reliability_loss = (reliability_error * presence).sum() / presence.sum().clamp_min(1)
            reliability_mae = (
                (pred_reliability - target_reliability).abs() * presence
            ).sum() / presence.sum().clamp_min(1)

            target_stack = torch.stack([target.clamp_min(0) for target in targets], dim=2)
            prediction_stack = torch.stack(
                [pred_geometry.clamp_min(0), pred_material.clamp_min(0), pred_noise.clamp_min(0)],
                dim=2,
            )
            target_sum = target_stack.sum(dim=2, keepdim=True).clamp_min(1e-6)
            prediction_sum = prediction_stack.sum(dim=2, keepdim=True).clamp_min(1e-6)
            target_partition = target_stack / target_sum
            pred_partition = prediction_stack / prediction_sum
            channel_presence = (target_sum > 1e-6).to(dtype=full.dtype)
            partition_error = (pred_partition - target_partition).abs()
            partition_loss = (partition_error * channel_presence).sum() / (
                channel_presence.sum().clamp_min(1) * 3.0
            )

            reliability_weight = float(getattr(cfg.loss, "causal_reliability_weight", 0.50))
            partition_weight = float(getattr(cfg.loss, "causal_partition_weight", 0.50))
            causal_loss = reliability_weight * reliability_loss + partition_weight * partition_loss
            total = total + causal_loss
            previous_extra = float(details.get("extra_loss_total", 0.0))
            details.update(
                {
                    "causal_reliability_loss": float(reliability_loss.detach()),
                    "causal_partition_loss": float(partition_loss.detach()),
                    "causal_reliability_mae": float(reliability_mae.detach()),
                    "causal_reliability_target_mean": float(
                        (target_reliability * presence).sum().detach()
                        / presence.sum().clamp_min(1).detach()
                    ),
                    "causal_reliability_pred_mean": float(
                        (pred_reliability * presence).sum().detach()
                        / presence.sum().clamp_min(1).detach()
                    ),
                    "extra_loss_total": previous_extra + float(causal_loss.detach()),
                    "total_loss_with_extra": float(total.detach()),
                }
            )
            aux["causal_reliability_pred"] = pred_reliability.squeeze(2).detach()
            aux["causal_reliability_target"] = target_reliability.squeeze(2).detach()
            return total, details, aux

    return CausalAdditiveReliabilityLoss


__all__ = ["make_causal_additive_loss"]
