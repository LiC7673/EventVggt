"""Three controlled ablations of the latest cur-event refiner-first route."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import finetune_event as fe
import torch

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_cur_event_hf_residual as hf
from paired_token_reliability import train_linear_voxel_cur_event_refiner_first as full
from paired_token_reliability.linear_voxel_cur_event_hf_residual_model import CurEventHFResidualModel


VARIANT = os.environ.get("ABLATION_VARIANT", "").strip().lower()
VALID = {"noisy_event_only", "multi_ldr_only", "without_refiner_normal"}
if VARIANT not in VALID:
    raise RuntimeError(f"set ABLATION_VARIANT to one of {sorted(VALID)}, got {VARIANT!r}")


def repeated_adapter_schedule(warmup, cycles, joint=0):
    if int(joint) != 0: raise ValueError("ablation uses --epochs-c 0")
    # Match the full route's total epoch budget without introducing a hidden B
    # phase that would activate E_geo/C supervision.
    return ["adapter"] * (int(warmup) + 2 * int(cycles))


def _ev0_views(batch, device):
    labels = [str(x).lower() for x in batch.get("ldr_a", ())]
    if not labels or any(x != "ev_0" for x in labels):
        raise RuntimeError(f"anchor ev_0 must be views_a, got {labels}")
    return fe.maybe_denormalize_views(pipeline.move_views_to_device(batch["views_a"], device))


def prepare_pair(batch, device, args, phase):
    if VARIANT in {"noisy_event_only", "multi_ldr_only"}:
        # Ask the original preparer for the real cur_event, never its Phase-A
        # geometry-event substitution.
        target, reference, event, bridge = pipeline._ORIGINAL_PREPARE_PAIR(
            batch, device, args, "contribution"
        )
        if VARIANT == "noisy_event_only":
            # Remove Multi-LDR: both RGB inputs are the fixed clean ev_0 image;
            # only the noisy event stream can explain residual geometry.
            for student, ev0 in zip(target, _ev0_views(batch, device)):
                student["img"] = ev0["img"]; student["hdr_img"] = ev0["img"]
                student["event_source_label"] = "noisy cur_event only"
        else:
            # Preserve the true degraded LDR input and use ev_0 solely as the
            # detached HDR target.
            for student, teacher in zip(target, reference):
                student["hdr_img"] = teacher["img"]
                student["event_source_label"] = "Multi-LDR only: LDR+cur_event->ev0"
        return target, reference, event, bridge
    return full.prepare_pair(batch, device, args, phase)


def configure_phase(model, phase, _train_heads=False):
    if VARIANT == "without_refiner_normal":
        return full.configure_phase(model, phase, _train_heads)
    model.requires_grad_(False); model.set_confidence_stage("geo")
    if VARIANT == "noisy_event_only":
        for module in (model.event_encoder, model.event_normal_decoder,
                       model.pixel_depth_refiner): module.requires_grad_(True)
        model.disable_pixel_refiner = False
        label = "noisy cur_event geometry only; no Geo/Multi-LDR/C"
    else:
        for module in (model.event_encoder, model.event_token_projection,
                       model.ldr_event_hdr_aligner): module.requires_grad_(True)
        model.disable_pixel_refiner = True
        label = "Multi-LDR token/base only; refiner/dN/C disabled"
    model.train(); model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(f"[ABLATION {VARIANT}] {label}", flush=True)


def optimizer_for(model, phase, args):
    if VARIANT == "without_refiner_normal": return full.optimizer_for(model, phase, args)
    if VARIANT == "noisy_event_only":
        groups = [
            {"params": list(model.pixel_depth_refiner.parameters()), "lr": 3*args.lr},
            {"params": list(model.event_encoder.parameters()), "lr": 2*args.lr},
            {"params": list(model.event_normal_decoder.parameters()), "lr": 2*args.lr},
        ]
    else:
        groups = [
            {"params": list(model.event_encoder.parameters()), "lr": args.lr},
            {"params": list(model.event_token_projection.parameters()), "lr": args.lr},
            {"params": list(model.ldr_event_hdr_aligner.parameters()), "lr": args.lr},
        ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(.9, .95))


class AblationObjective:
    def __init__(self, base, args): self.base, self.args = base, args
    def __call__(self, *a, **kw):
        result = self.base(*a, **kw)
        if VARIANT == "noisy_event_only":
            # No HDR-token/Multi-LDR term in this branch.
            term = result.details.get("hdr_token_alignment", result.loss.new_zeros(()))
            result.loss = result.loss - term
            result.details["hdr_token_alignment"] = term.detach() * 0
        elif VARIANT == "multi_ldr_only":
            # Keep final/base depth, ordinary normal and HDR alignment; remove
            # every event-derivative/refiner-specific objective.
            en = result.details.get("event_normal", result.loss.new_zeros(()))
            dn = result.details.get("depth_event_normal", result.loss.new_zeros(()))
            hf_term = result.details.get("explicit_hf_residual", result.loss.new_zeros(()))
            result.loss = result.loss - self.args.event_normal_weight * en - dn - 2.0 * hf_term
            result.details.update(event_normal=en.detach()*0, depth_event_normal=dn.detach()*0,
                                  explicit_hf_residual=hf_term.detach()*0)
        else:
            # Remove ordinary final-normal supervision and the explicit
            # final-depth normal-derivative coupling. Event dN supervision and
            # signed HF depth residual remain intact.
            normal = result.details.get("normal", result.loss.new_zeros(()))
            dn = result.details.get("depth_event_normal", result.loss.new_zeros(()))
            result.loss = result.loss - self.args.normal_weight * normal - dn
            result.details.update(normal=normal.detach()*0, depth_event_normal=dn.detach()*0,
                                  final_gt_normal_derivative=dn.detach()*0)
        result.details["loss"] = result.loss
        return result


def criterion_for(args, phase): return AblationObjective(hf.criterion_for(args, phase), args)


def force(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    blocked=("data.event_source_mode=", "data.decomposition_supervision=",
             "data.decomposition_geo_branch=")
    values=[x for x in values if not x.startswith(blocked)]
    values += ["data.event_source_mode=cur_event", "data.decomposition_supervision=true",
               "data.decomposition_geo_branch=geometry_motion"]
    return values


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = prepare_pair
    pipeline.build_alternating_phase_schedule = (
        alternating.schedule if VARIANT == "without_refiner_normal" else repeated_adapter_schedule
    )
    pipeline.build_model = hf.build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = alternating.final_base.save_visual
    pipeline.capture_runtime_state = alternating.capture_runtime_state
    pipeline.restore_runtime_state = alternating.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = CurEventHFResidualModel
    print(f"[LATEST ABLATION] variant={VARIANT}", flush=True)
    pipeline.main(force(argv))


if __name__ == "__main__": main()
