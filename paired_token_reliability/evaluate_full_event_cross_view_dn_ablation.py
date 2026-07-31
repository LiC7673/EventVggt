"""Evaluate a full/geo sole-input DN ablation with both C maps fixed to one."""
from __future__ import annotations

import os
import sys

from omegaconf import OmegaConf

from paired_token_reliability import (
    evaluate_alternating_detail_first_fixed_four_scenes as evaluator,
)
from paired_token_reliability import evaluate_cur_event_hf_residual_four_scenes as v2
from paired_token_reliability import evaluate_cur_event_cross_view_dn_four_scenes as dn_eval
import real_reliability_stage.evaluate_stage2_heldout as protocol


EVENT_BRANCH = os.environ.get("FULL_DN_EVENT_BRANCH", "full").strip()
if EVENT_BRANCH not in {"full", "geometry_motion"}:
    raise ValueError(
        f"FULL_DN_EVENT_BRANCH={EVENT_BRANCH!r}; expected full or geometry_motion"
    )


def build_model(checkpoint, device, depth_scale):
    model, cfg = v2.build_model(checkpoint, device, depth_scale)
    # Training uses adapter/Geo-stage gate semantics: both independently
    # defined C maps are exactly one.  Reproduce that deployment here instead
    # of evaluating an untrained predicted-confidence path.
    model.set_confidence_stage("geo")
    model.require_geo_teacher = False
    print(
        f"[FULL-DN eval] sole event branch={EVENT_BRANCH}; "
        "C_fusion=C_refine=1; no geometry teacher",
        flush=True,
    )
    return model, cfg


_build_loader = protocol.build_loader


def build_loader(cfg, args):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.decomposition_event_root = "events_additive"
    cfg.data.decomposition_full_branch = EVENT_BRANCH
    cfg.data.decomposition_supervision = False
    return _build_loader(cfg, args)


def main():
    evaluator.build_model = build_model
    evaluator.save_visual = dn_eval.save_visual
    protocol.build_loader = build_loader
    if "--event-source-mode" not in sys.argv:
        sys.argv.extend(("--event-source-mode", "decomposition_full"))
    evaluator.main()


if __name__ == "__main__":
    main()
