"""Two-stage geometry model: learn normals on E_geo, then learn C on E_full."""
from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_multiscale_model import (
    LinearVoxelMultiscalePixelModel,
)


class DelayedContribution(nn.Module):
    """C=1 first, then smoothly hand control to the learned full-event C."""

    def __init__(self, learned, delay=1000, transition=1000):
        super().__init__()
        self.learned = learned
        self.coarse_feature_dim = getattr(learned, "coarse_feature_dim", 0)
        self.delay = int(delay)
        self.transition = max(1, int(transition))
        self.stage = "geo"
        self.advance_step = True
        self.register_buffer("full_step", torch.zeros((), dtype=torch.long))
        self.last_predicted = None
        self.last_ramp = 0.0

    def set_stage(self, stage):
        if stage not in {"geo", "full"}:
            raise ValueError(stage)
        self.stage = stage

    def forward(self, *args, **kwargs):
        predicted = self.learned(*args, **kwargs)
        self.last_predicted = predicted
        if self.stage == "geo":
            self.last_ramp = 0.0
            return torch.ones_like(predicted) + 0.0 * predicted
        if self.training and self.advance_step:
            self.full_step.add_(1)
        ramp = ((float(self.full_step.item()) - self.delay) / self.transition)
        ramp = max(0.0, min(1.0, ramp))
        self.last_ramp = ramp
        # Before delay this remains connected with exactly zero gradient, so
        # DDP sees the parameters without training them prematurely.
        return 1.0 + ramp * (predicted - 1.0)


class GeoNormalThenFullCModel(LinearVoxelMultiscalePixelModel):
    checkpoint_schema = "linear_voxel_geo_normal_then_delayed_full_c_v2"

    def __init__(self, *args, c_delay_steps=1000, c_transition_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.contribution_net = DelayedContribution(
            self.contribution_net,
            delay=c_delay_steps,
            transition=c_transition_steps,
        )

    def set_stage(self, stage):
        self.contribution_net.set_stage(stage)

    def predict_contribution(self, views):
        """Reference/validation queries must not advance the training clock."""
        previous = self.contribution_net.advance_step
        self.contribution_net.advance_step = False
        try:
            return super().predict_contribution(views)
        finally:
            self.contribution_net.advance_step = previous

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        predicted = self.contribution_net.last_predicted
        ramp = self.contribution_net.last_ramp
        for index, item in enumerate(output.ress):
            item["predicted_full_contribution"] = (
                predicted[:, index] if predicted is not None
                else item["event_contribution"].detach()
            )
            item["contribution_learning_ramp"] = item["depth"].new_tensor(ramp)
            item["contribution_full_step"] = self.contribution_net.full_step.detach().clone()
        return output


__all__ = ["GeoNormalThenFullCModel", "DelayedContribution"]
