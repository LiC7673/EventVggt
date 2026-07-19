"""Four-scene evaluation with adjacent-view dNormal patch diagnostics."""
from __future__ import annotations

import sys

from paired_token_reliability import evaluate_alternating_detail_first_fixed_four_scenes as evaluator
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import build_model
from paired_token_reliability.cross_view_event_normal_consistency import (
    cross_view_patch_loss, save_patch_diagnostics,
)


_base_save_visual = evaluator.save_visual


def save_visual(root, scene, exposure, index, views, output, depth_gt, valid,
                intrinsics, event_source_mode):
    _base_save_visual(root, scene, exposure, index, views, output, depth_gt,
                      valid, intrinsics, event_source_mode)
    _, diagnostics = cross_view_patch_loss(
        output, views, patch_size=14, min_overlap=8,
        min_overlap_ratio=.03, depth_tolerance=.20,
    )
    save_patch_diagnostics(
        root, f"test_{scene}_{exposure}", 0, index, diagnostics
    )


def main():
    evaluator.build_model = build_model
    evaluator.save_visual = save_visual
    if "--event-source-mode" not in sys.argv:
        sys.argv.extend(("--event-source-mode", "cur_event"))
    evaluator.main()


if __name__ == "__main__":
    main()

