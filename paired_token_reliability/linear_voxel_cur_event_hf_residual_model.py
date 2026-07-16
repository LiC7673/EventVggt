"""Cur-event model whose clean-Geo epochs activate pixel refinement immediately."""
from __future__ import annotations

from paired_token_reliability.linear_voxel_alternating_detail_first_fixed_model import (
    AlternatingDetailFirstFixedModel,
)


class CurEventHFResidualModel(AlternatingDetailFirstFixedModel):
    checkpoint_schema = "cur_event_explicit_hf_residual_final_derivative_v1"

    def forward(self, *args, **kwargs):
        # The parent pixel refiner normally ramps only after the global clock.
        # During clean E_geo epochs that made an allegedly trainable refiner a
        # zero-output branch.  Force full pixel coupling only for Geo; cur_event
        # confidence epochs retain the original delayed deployment.
        geo = getattr(self.full_geo_aligner, "stage", "full") == "geo"
        if geo:
            self._dual_alignment_step += 1500
        try:
            return super().forward(*args, **kwargs)
        finally:
            if geo:
                self._dual_alignment_step -= 1500


__all__ = ["CurEventHFResidualModel"]
