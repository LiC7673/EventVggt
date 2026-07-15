"""Alternating HDR model with independently staged fusion/refinement gates."""
from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_dual_alignment_hdr_final_pixel_refiner_model import (
    FinalEventGeometryPixelRefinerModel,
)


class DelayedGate(nn.Module):
    def __init__(self, learned: nn.Module, delay=1000, transition=1000):
        super().__init__()
        self.learned = learned
        self.delay = int(delay)
        self.transition = max(1, int(transition))
        self.stage = "geo"
        self.advance_step = True
        self.register_buffer("full_step", torch.zeros((), dtype=torch.long))
        self.last_prediction = None
        self.last_ramp = 0.0

    def set_stage(self, stage):
        if stage not in {"geo", "full"}:
            raise ValueError(stage)
        self.stage = stage
        if stage == "geo":
            self.last_ramp = 0.0

    def forward(self, *args, **kwargs):
        prediction = self.learned(*args, **kwargs)
        self.last_prediction = prediction
        if self.stage == "geo":
            self.last_ramp = 0.0
            return torch.ones_like(prediction) + prediction * 0.0
        if self.training and self.advance_step:
            self.full_step.add_(1)
        ramp = max(0.0, min(1.0, (self.full_step.item() - self.delay) / self.transition))
        self.last_ramp = float(ramp)
        return 1.0 + float(ramp) * (prediction - 1.0)


class AlternatingDetailFirstModel(FinalEventGeometryPixelRefinerModel):
    checkpoint_schema = "alternating_geo_detail_then_dual_c_v1"

    def __init__(self, *args, c_delay_steps=1000, c_transition_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.contribution_net = DelayedGate(
            self.contribution_net, c_delay_steps, c_transition_steps
        )
        self.normal_fusion_gate = DelayedGate(
            self.normal_fusion_gate, c_delay_steps, c_transition_steps
        )

    def set_confidence_stage(self, stage):
        self.contribution_net.set_stage(stage)
        self.normal_fusion_gate.set_stage(stage)

    def predict_contribution(self, views):
        # Paired/reference diagnostics must not consume a Full-C training step.
        previous = self.contribution_net.advance_step
        self.contribution_net.advance_step = False
        try:
            return super().predict_contribution(views)
        finally:
            self.contribution_net.advance_step = previous

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        for item in output.ress:
            item["c_fusion_ramp"] = item["depth"].new_tensor(
                self.contribution_net.last_ramp
            )
            item["c_refine_ramp"] = item["depth"].new_tensor(
                self.normal_fusion_gate.last_ramp
            )
            item["c_fusion_step"] = self.contribution_net.full_step.detach().clone()
            item["c_refine_step"] = self.normal_fusion_gate.full_step.detach().clone()
        return output


__all__ = ["AlternatingDetailFirstModel", "DelayedGate"]
