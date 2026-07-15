"""Fixed detail-first model: direct Geo path and a real refinement confidence gate."""
from __future__ import annotations

import torch
import torch.nn as nn

from paired_token_reliability.linear_voxel_alternating_detail_first_model import (
    AlternatingDetailFirstModel, DelayedGate,
)


class GeoBypassFullAligner(nn.Module):
    """Use Geo features unchanged in Geo epochs; align only Full epochs."""
    def __init__(self, learned):
        super().__init__()
        self.learned = learned
        self.stage = "geo"

    def set_stage(self, stage): self.stage = stage

    def forward(self, value):
        if self.stage == "geo":
            zero = torch.zeros_like(value)
            # Preserve the three-value FullToGeoAlignment interface.
            return value, zero, value[:, :1] * 0.0
        return self.learned(value)


class AlternatingDetailFirstFixedModel(AlternatingDetailFirstModel):
    checkpoint_schema = "alternating_geo_direct_detail_dual_c_fixed_v2"

    def __init__(self, *args, pixel_hidden=32, pixel_refiner_delay=500, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        self.full_geo_aligner = GeoBypassFullAligner(self.full_geo_aligner)

        # The derivative parent installed _ZeroNormalGate. Replace it with an
        # actual confidence predictor for the pixel/normal refinement route.
        hidden = int(pixel_hidden)
        refine_predictor = nn.Sequential(
            nn.Conv2d(hidden + 6, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid(),
        )
        nn.init.zeros_(refine_predictor[-2].weight)
        nn.init.constant_(refine_predictor[-2].bias, 0.84729786)  # sigmoid=0.7
        delay = self.contribution_net.delay
        transition = self.contribution_net.transition
        self.normal_fusion_gate = DelayedGate(refine_predictor, delay, transition)
        self.pixel_refiner_delay = max(0, int(pixel_refiner_delay))

    def set_confidence_stage(self, stage):
        super().set_confidence_stage(stage)
        self.full_geo_aligner.set_stage(stage)

    def _refinement_contribution(self, output, fusion_contribution):
        # The common HDR path has already evaluated normal_fusion_gate from
        # aligned event features and coarse geometry. Use that independently
        # predicted/deployed map for pixel refinement, not C_fusion.
        return torch.stack(
            [item["normal_fusion_gate"] for item in output.ress], dim=1
        )

    def forward(self, *args, **kwargs):
        # The inherited pixel refiner activates at internal step 1000. Shift
        # only the temporary forward clock so its effective delay is 500,
        # while keeping the stored/resumed clock equal to real train steps.
        offset = 1000 - self.pixel_refiner_delay
        self._dual_alignment_step += offset
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._dual_alignment_step -= offset


__all__ = ["AlternatingDetailFirstFixedModel", "GeoBypassFullAligner"]
