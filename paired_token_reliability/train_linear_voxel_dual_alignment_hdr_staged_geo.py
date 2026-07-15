"""Sequential A: learn E_geo geometry; B: freeze it and learn E_full alignment."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10
from paired_token_reliability.linear_voxel_dual_alignment_hdr_staged_model import (
    StagedGeoTeacherDualAlignmentModel,
)


def sequential_schedule(geo_epochs, full_epochs, joint_epochs=0):
    if int(geo_epochs) <= 0 or int(full_epochs) <= 0 or int(joint_epochs):
        raise ValueError("staged route requires epochs-a>0, epochs-b>0, epochs-c=0")
    return ["adapter"] * int(geo_epochs) + ["contribution"] * int(full_epochs)


def build_model(cfg, args, device):
    original = v10.DualAlignmentHDRLinearVoxelModel
    v10.DualAlignmentHDRLinearVoxelModel = StagedGeoTeacherDualAlignmentModel
    try:
        model = v10.build_model(cfg, args, device)
        # The external initialization checkpoint has no teacher-prefixed
        # keys.  Initialize stage A from the loaded event representation,
        # rather than from the constructor state that existed before loading.
        model.geo_event_encoder.load_state_dict(model.event_encoder.state_dict())
        model.geo_normal_decoder.load_state_dict(model.event_normal_decoder.state_dict())
        return model
    finally:
        v10.DualAlignmentHDRLinearVoxelModel = original


def configure_phase(model, phase, _train_heads_a=False):
    if phase == "adapter":
        model.requires_grad_(False)
        model.geo_event_encoder.requires_grad_(True)
        model.geo_normal_decoder.requires_grad_(True)
        model.train()
        model.aggregator.eval(); model.depth_head.eval(); model.point_head.eval()
        model.event_encoder.eval(); model.event_normal_decoder.eval()
        print("[stage A] train E_geo encoder+normal decoder only", flush=True)
        return
    if phase != "contribution":
        raise ValueError(f"unsupported staged phase {phase}")
    copied = model.initialize_full_student_from_geo()
    v10.configure_phase(model, "adapter", _train_heads_a)
    model.geo_event_encoder.requires_grad_(False)
    model.geo_normal_decoder.requires_grad_(False)
    model.geo_event_encoder.eval(); model.geo_normal_decoder.eval()
    print(
        f"[stage B] frozen E_geo teacher; train E_full alignment+fusion "
        f"initialized_from_geo={int(copied)}", flush=True,
    )


def optimizer_for(model, phase, args):
    if phase == "adapter":
        return torch.optim.AdamW([
            {"params": model.geo_event_encoder.parameters(), "lr": 2.0 * args.lr},
            {"params": model.geo_normal_decoder.parameters(), "lr": 5.0 * args.lr},
        ], lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))
    return v10.optimizer_for(model, "adapter", args)


class GeoTeacherObjective:
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic

    def __call__(self, output, views, *args, **kwargs):
        result = self.diagnostic(output, views, *args, **kwargs)
        pred = torch.stack([x["event_normal_geo"] for x in output.ress], 1).float()
        gt = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6)
        pred = F.normalize(pred, dim=-1, eps=1e-6)
        support = torch.stack([x["geo_event_support"] for x in output.ress], 1).bool()
        valid = result.aux["normal_valid_live"].bool() & support
        weight = valid.float()
        cosine = ((1.0 - (pred * gt).sum(-1).clamp(-1, 1)) * weight).sum() / weight.sum().clamp_min(1)
        hf = pred.new_zeros(())
        for kernel in (3, 7):
            error = (v10._hf(pred, kernel) - v10._hf(gt, kernel)).abs().mean(-1)
            hf = hf + (error * weight).sum() / weight.sum().clamp_min(1)
        hf = .5 * hf
        result.loss = 2.0 * cosine + hf
        result.details["geo_teacher_normal"] = cosine
        result.details["geo_teacher_hf"] = hf
        result.details["loss"] = result.loss
        return result


def criterion_for(args, phase):
    diagnostic = v10.criterion_for(args, "adapter")
    return GeoTeacherObjective(diagnostic) if phase == "adapter" else diagnostic


def capture_runtime_state(model):
    state = v10.capture_runtime_state(model)
    state["full_initialized_from_geo"] = int(model.full_initialized_from_geo.item())
    return state


def restore_runtime_state(model, state):
    v10.restore_runtime_state(model, state)
    model.full_initialized_from_geo.fill_(int(state.get("full_initialized_from_geo", 0)))


def main(argv=None):
    pipeline.prepare_pair = v10.prepare_dual_alignment_pair
    pipeline.build_alternating_phase_schedule = sequential_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = v10.save_visual
    pipeline.capture_runtime_state = capture_runtime_state
    pipeline.restore_runtime_state = restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = StagedGeoTeacherDualAlignmentModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
