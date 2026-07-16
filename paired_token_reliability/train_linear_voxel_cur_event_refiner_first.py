"""Refiner-first schedule: freeze HDR modules for ~1k steps, then joint adapt."""
from __future__ import annotations

import sys
import torch

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_alternating_detail_first_fixed_cur_event as cur
from paired_token_reliability import train_linear_voxel_cur_event_hf_residual as hf
from paired_token_reliability.linear_voxel_cur_event_hf_residual_model import CurEventHFResidualModel


REFINER_FIRST_STEPS = 1000


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    step = int(getattr(model, "_dual_alignment_step", 0))
    if phase == "adapter":
        model.set_confidence_stage("geo")
        # Event representation and its differential geometry prediction must
        # learn together with the pixel refiner from the first batch.
        for module in (model.event_encoder, model.event_normal_decoder,
                       model.pixel_depth_refiner):
            module.requires_grad_(True)
        if step >= REFINER_FIRST_STEPS:
            # Stage 2: cautiously open the HDR-token route. The optimizer has
            # separate low-LR groups for these modules.
            model.event_token_projection.requires_grad_(True)
            model.ldr_event_hdr_aligner.requires_grad_(True)
            label = "joint Geo refinement; HDR Adapter opened at low LR"
        else:
            label = (
                f"refiner-first {step}/{REFINER_FIRST_STEPS}; "
                "HDR projection/Adapter frozen; both C=1"
            )
    elif phase == "contribution":
        model.set_confidence_stage("full")
        # Full->Geo and both confidence maps only start after the initial Geo
        # epoch. Their deployment is already ramped by c_transition_steps.
        model.full_geo_aligner.learned.requires_grad_(True)
        model.contribution_net.learned.requires_grad_(True)
        model.normal_fusion_gate.learned.requires_grad_(True)
        label = "cur_event Full->Geo + gradual C_fusion/C_refine, low LR"
    else:
        raise ValueError(phase)
    model.train(); model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(f"[REFINER-FIRST/{phase}] {label}", flush=True)


def optimizer_for(model, phase, args):
    if phase == "adapter":
        # Include the initially frozen HDR groups now: AdamW simply skips them
        # while grad=None, then starts them without rebuilding the optimizer.
        groups = [
            {"params": list(model.pixel_depth_refiner.parameters()), "lr": 3.0 * args.lr},
            {"params": list(model.event_encoder.parameters()), "lr": 2.0 * args.lr},
            {"params": list(model.event_normal_decoder.parameters()), "lr": 2.0 * args.lr},
            {"params": list(model.event_token_projection.parameters()), "lr": .30 * args.lr},
            {"params": list(model.ldr_event_hdr_aligner.parameters()), "lr": .30 * args.lr},
        ]
    else:
        groups = [
            {"params": list(model.full_geo_aligner.learned.parameters()), "lr": .50 * args.lr},
            {"params": list(model.contribution_net.learned.parameters()), "lr": .30 * args.lr},
            {"params": list(model.normal_fusion_gate.learned.parameters()), "lr": .30 * args.lr},
        ]
    params = [p for group in groups for p in group["params"]]
    if not params:
        raise RuntimeError(f"phase {phase} has no optimizer parameters")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(.9, .95))


def _force(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    blocked = ("data.event_source_mode=", "data.decomposition_supervision=",
               "data.decomposition_geo_branch=", "model.c_delay_steps=",
               "model.c_transition_steps=")
    values = [x for x in values if not x.startswith(blocked)]
    values += [
        "data.event_source_mode=cur_event",
        "data.decomposition_supervision=true",
        "data.decomposition_geo_branch=geometry_motion",
        # C starts only after the ~1k Geo/refiner epoch and then takes another
        # 1k forward steps to reach its fully predicted value.
        "model.c_delay_steps=1000",
        "model.c_transition_steps=1000",
    ]
    return values


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = cur.prepare_pair
    pipeline.build_alternating_phase_schedule = alternating.schedule
    pipeline.build_model = hf.build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = hf.criterion_for
    pipeline.save_visual = alternating.final_base.save_visual
    pipeline.capture_runtime_state = alternating.capture_runtime_state
    pipeline.restore_runtime_state = alternating.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = CurEventHFResidualModel
    print(
        "[REFINER-FIRST] Geo epoch ~=1000 steps: train event encoder+dN+refiner; "
        "freeze HDR Adapter/two C; then low-LR HDR+C alternating adaptation",
        flush=True,
    )
    pipeline.main(_force(argv))


if __name__ == "__main__":
    main()
