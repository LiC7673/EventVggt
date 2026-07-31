"""Full-event-only cross-view dN ablations.

This entry deliberately does not request or consume ``geometry_event_voxel``.
All variants share the same full-event input, model, optimizer and three
adapter epochs.  The only difference is the explicitly removed objective:

* full:              all HDR/refiner/dN objectives;
* no_hdr_align:      remove LDR+event -> HDR token alignment;
* no_refiner_loss:   remove explicit HF residual and final-depth dN coupling;
* geo_only:          use geometry_motion as the sole event input.

The event-normal derivative objective and cross-view dN consistency remain
enabled in every variant so the ablations change one mechanism at a time.
"""
from __future__ import annotations

import os
import sys

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_cur_event_hf_residual as hf
from paired_token_reliability import train_linear_voxel_cur_event_refiner_first as refiner_first
from paired_token_reliability import train_linear_voxel_cur_event_cross_view_dn as cross_view
from paired_token_reliability.linear_voxel_cur_event_hf_residual_model import (
    CurEventHFResidualModel,
)


VALID_VARIANTS = {"full", "no_hdr_align", "no_refiner_loss", "geo_only"}
VARIANT = os.environ.get("FULL_DN_ABLATION", "full").strip().lower()
if VARIANT not in VALID_VARIANTS:
    raise ValueError(
        f"FULL_DN_ABLATION={VARIANT!r}; expected one of {sorted(VALID_VARIANTS)}"
    )


def prepare_pair(batch, device, args, phase):
    """Select the degraded LDR/full-event student and the better RGB teacher.

    Calling the common preparer with ``contribution`` prevents its legacy
    adapter-phase E_geo substitution.  No geometry-event tensor is accessed.
    """
    target, reference, event, bridge = pipeline._ORIGINAL_PREPARE_PAIR(
        batch, device, args, "contribution"
    )
    for student, teacher in zip(target, reference):
        student["hdr_img"] = teacher["img"]
        source = "E_geo only" if VARIANT == "geo_only" else "E_full only"
        student["event_source_label"] = (
            f"{source} + degraded Multi-LDR -> better-exposure RGB teacher"
        )
        # Fail loudly if a future dataset/preparer accidentally reintroduces
        # the controlled geometry branch into this full-only experiment.
        student.pop("geometry_event_voxel", None)
        teacher.pop("geometry_event_voxel", None)
    return target, reference, event, bridge


def build_model(cfg, args, device):
    model = hf.build_model(cfg, args, device)
    # E_geo is neither loaded nor required.  HDR supervision remains available
    # through ``hdr_img`` attached by prepare_pair.
    model.require_geo_teacher = False
    return model


def three_full_event_epochs(_epochs_a, _epochs_b, _epochs_c=0):
    return ["adapter", "adapter", "adapter"]


def configure_phase(model, phase, train_heads_a=False):
    if phase != "adapter":
        raise ValueError(f"full-only ablation supports adapter epochs only, got {phase}")
    refiner_first.configure_phase(model, phase, train_heads_a)
    print(
        f"[FULL-DN/{VARIANT}] "
        f"{'E_geo sole input' if VARIANT == 'geo_only' else 'E_full sole input'}; "
        "C_fusion=C_refine=1; "
        "no Full->Geo teacher/alignment",
        flush=True,
    )


class ObjectiveAblation:
    def __init__(self, base):
        self.base = base

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        zero = result.loss.new_zeros(())
        removed = zero

        if VARIANT == "no_hdr_align":
            removed = result.details.get("hdr_token_alignment", zero)
            result.details["hdr_token_alignment_removed"] = removed.detach()
            result.details["hdr_token_alignment"] = zero
        elif VARIANT == "no_refiner_loss":
            hf_residual = result.details.get("explicit_hf_residual", zero)
            final_dn = result.details.get("final_gt_normal_derivative", zero)
            # CleanCurEventGeometryObjective adds 2*HF + 1*final-depth-dN.
            removed = 2.0 * hf_residual + final_dn
            result.details["refiner_objective_removed"] = removed.detach()
            result.details["explicit_hf_residual"] = zero
            result.details["final_gt_normal_derivative"] = zero
            result.details["depth_event_normal"] = zero

        result.loss = result.loss - removed
        result.details["loss"] = result.loss
        return result


def criterion_for(args, phase):
    base = ObjectiveAblation(hf.criterion_for(args, phase))
    # Retain the DN script's pose-aware adjacent-view patch consistency.
    return cross_view.CrossViewObjective(base, phase)


def _force(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    blocked = (
        "data.event_source_mode=",
        "data.decomposition_supervision=",
        "data.decomposition_geo_branch=",
        "data.decomposition_full_branch=",
    )
    values = [item for item in values if not item.startswith(blocked)]
    sole_branch = "geometry_motion" if VARIANT == "geo_only" else "full"
    values += [
        "data.event_source_mode=decomposition_full",
        "data.decomposition_supervision=false",
        f"data.decomposition_full_branch={sole_branch}",
    ]
    return values


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = prepare_pair
    pipeline.build_alternating_phase_schedule = three_full_event_epochs
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = refiner_first.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = cross_view.save_visual
    pipeline.capture_runtime_state = alternating.capture_runtime_state
    pipeline.restore_runtime_state = alternating.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = CurEventHFResidualModel
    print(
        f"[FULL-DN ABLATION] variant={VARIANT}; schedule=adapter x3; "
        f"event=events_additive/"
        f"{'geometry_motion' if VARIANT == 'geo_only' else 'full'}; "
        "geometry_event_voxel teacher=DISABLED",
        flush=True,
    )
    pipeline.main(_force(argv))


if __name__ == "__main__":
    main()
