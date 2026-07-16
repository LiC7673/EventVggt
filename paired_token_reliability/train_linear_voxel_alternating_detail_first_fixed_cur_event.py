"""Alternating detail-first training with a strict cur_event main stream.

Unlike the generic fixed trainer, this entry cannot silently inherit or be
overridden to decomposition_full/current.  Phase A consumes geometry_motion;
phase B consumes <scene>/cur_event/events.h5 and learns Full-to-Geo alignment
plus the two confidence maps.
"""
from __future__ import annotations

import sys

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_alternating_detail_first_fixed as fixed
from paired_token_reliability.linear_voxel_alternating_detail_first_fixed_model import (
    AlternatingDetailFirstFixedModel,
)


def prepare_pair(batch, device, args, phase):
    target, reference, event, bridge = pipeline._ORIGINAL_PREPARE_PAIR(
        batch, device, args, phase
    )
    for student, teacher in zip(target, reference):
        student["hdr_img"] = teacher["img"]
        student["event_source_label"] = (
            "E_geo: events_additive/geometry_motion"
            if phase == "adapter"
            else "E_cur: cur_event/events.h5"
        )
    return target, reference, event, bridge


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.set_confidence_stage("geo")
        for module in (
            model.event_encoder,
            model.event_normal_decoder,
            model.event_token_projection,
            model.ldr_event_hdr_aligner,
            model.pixel_depth_refiner,
        ):
            module.requires_grad_(True)
        label = "geometry_motion -> dN/pixel refiner; C_fusion=C_refine=1"
    elif phase == "contribution":
        model.set_confidence_stage("full")
        model.full_geo_aligner.learned.requires_grad_(True)
        model.contribution_net.learned.requires_grad_(True)
        model.normal_fusion_gate.learned.requires_grad_(True)
        label = "strict cur_event -> Geo aligner + C_fusion + C_refine"
    else:
        raise ValueError(phase)
    model.train()
    model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(f"[cur-event/{phase}] {label}", flush=True)


def _force_cur_event(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    blocked_prefixes = (
        "data.event_source_mode=",
        "data.decomposition_supervision=",
        "data.decomposition_geo_branch=",
    )
    values = [x for x in values if not x.startswith(blocked_prefixes)]
    values.extend((
        "data.event_source_mode=cur_event",
        "data.decomposition_supervision=true",
        "data.decomposition_geo_branch=geometry_motion",
    ))
    return values


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = prepare_pair
    pipeline.build_alternating_phase_schedule = alternating.schedule
    pipeline.build_model = fixed.build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = fixed.optimizer_for
    pipeline.criterion_for = alternating.criterion_for
    pipeline.save_visual = alternating.final_base.save_visual
    pipeline.capture_runtime_state = alternating.capture_runtime_state
    pipeline.restore_runtime_state = alternating.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = AlternatingDetailFirstFixedModel
    print(
        "[STRICT CUR EVENT] main=cur_event/events.h5; "
        "Geo teacher=events_additive/geometry_motion/events.h5",
        flush=True,
    )
    pipeline.main(_force_cur_event(argv))


if __name__ == "__main__":
    main()
