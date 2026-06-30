"""Diagnostics for the frozen Stage-1 geometry stream."""

import torch

import finetune_event as fe
from mul_loss_fine.image_guided_event_reliability_loss import (
    make_configured_image_guided_event_reliability_loss,
)


def make_two_stage_reliability_loss(cfg):
    base = make_configured_image_guided_event_reliability_loss(cfg)

    class TwoStageReliabilityLoss(base):
        def forward(self, model_output, views):
            filtered_views = model_output.views if model_output.views is not None else views
            total, details, aux = super().forward(model_output, filtered_views)
            predicted = torch.stack(
                [result["stage1_geometry_event"] for result in model_output.ress], dim=1
            )
            filtered = torch.stack(
                [result["stage1_filtered_event"] for result in model_output.ress], dim=1
            )
            full = fe.stack_view_field(views, "event_voxel").to(
                device=predicted.device, dtype=predicted.dtype
            )
            full_energy = full.abs().mean().clamp_min(1e-8)
            details.update(
                {
                    "stage1_geometry_energy_ratio": float(
                        (predicted.abs().mean() / full_energy).detach()
                    ),
                    "stage1_filtered_energy_ratio": float(
                        (filtered.abs().mean() / full_energy).detach()
                    ),
                    "stage1_geometry_nonzero_ratio": float((predicted > 1e-6).float().mean().detach()),
                }
            )
            return total, details, aux

    return TwoStageReliabilityLoss

