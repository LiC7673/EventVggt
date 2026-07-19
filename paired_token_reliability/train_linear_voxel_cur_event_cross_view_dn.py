"""Refiner-first training plus adjacent-view event dNormal consistency."""
from __future__ import annotations

import os
import sys

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_cur_event_hf_residual as hf
from paired_token_reliability import train_linear_voxel_cur_event_refiner_first as refiner_first
from paired_token_reliability.cross_view_event_normal_consistency import (
    cross_view_patch_loss, save_patch_diagnostics,
)
from paired_token_reliability.linear_voxel_cur_event_hf_residual_model import CurEventHFResidualModel


WEIGHT = float(os.environ.get("CROSS_VIEW_DN_WEIGHT", "0.20"))
PATCH = int(os.environ.get("CROSS_VIEW_PATCH_SIZE", "14"))
MIN_OVERLAP = int(os.environ.get("CROSS_VIEW_MIN_OVERLAP", "8"))


class CrossViewObjective:
    def __init__(self, base, phase):
        self.base, self.phase, self.calls = base, phase, 0

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        loss, diagnostics = cross_view_patch_loss(
            output, views, patch_size=PATCH, min_overlap=MIN_OVERLAP
        )
        # The contribution epoch freezes the derivative branch. Compute the
        # diagnostic there, but only optimize it during geometry/adapter epochs.
        if self.phase == "adapter":
            result.loss = result.loss + WEIGHT * loss
        result.details["cross_view_event_dn"] = loss
        result.details["cross_view_accepted_pairs"] = result.loss.new_tensor(
            sum(int(item["accepted"]) for item in diagnostics)
        )
        result.details["cross_view_overlap"] = result.loss.new_tensor(
            sum(item["overlap_ratio"] for item in diagnostics) / max(len(diagnostics), 1)
        )
        result.details["loss"] = result.loss
        output.cross_view_patch_diagnostics = diagnostics
        if self.calls % 20 == 0:
            accepted = sum(int(item["accepted"]) for item in diagnostics)
            overlap = sum(item["overlap_ratio"] for item in diagnostics) / max(len(diagnostics), 1)
            print(
                f"[cross-view-dN/{self.phase}] loss={float(loss.detach()):.6f} "
                f"accepted={accepted}/{len(diagnostics)} mean_overlap={overlap:.4f}",
                flush=True,
            )
        self.calls += 1
        return result


def criterion_for(args, phase):
    return CrossViewObjective(hf.criterion_for(args, phase), phase)


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    alternating.final_base.save_visual(
        output_root, phase, epoch, batch_index, views, reference_views,
        event, bridge, output, aux,
    )
    diagnostics = getattr(output, "cross_view_patch_diagnostics", None)
    if diagnostics:
        save_patch_diagnostics(output_root, phase, epoch, batch_index, diagnostics)


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = refiner_first.prepare_pair
    pipeline.build_alternating_phase_schedule = alternating.schedule
    pipeline.build_model = hf.build_model
    pipeline.configure_phase = refiner_first.configure_phase
    pipeline.optimizer_for = refiner_first.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.capture_runtime_state = alternating.capture_runtime_state
    pipeline.restore_runtime_state = alternating.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = CurEventHFResidualModel
    print(
        f"[CROSS-VIEW-dN] adjacent pairs, patch={PATCH}, min_overlap={MIN_OVERLAP}, "
        f"weight={WEIGHT}; pose/depth mapping detached",
        flush=True,
    )
    pipeline.main(refiner_first._force(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
