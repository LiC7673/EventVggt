"""Latest cur-event refiner-first training with a delayed supervised pose path."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

import finetune_event as fe
from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_cur_event_hf_residual as hf
from paired_token_reliability import train_linear_voxel_cur_event_refiner_first as refiner_first
from paired_token_reliability import train_linear_voxel_cur_event_cross_view_dn as cross_view
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_cur_event_pose_refine_model import (
    CurEventPoseRefineModel,
)


POSE_WEIGHT = float(os.environ.get("POSE_REFINE_WEIGHT", "0.10"))
POSE_ROT_WEIGHT = float(os.environ.get("POSE_ROTATION_WEIGHT", "0.50"))
POSE_REG_WEIGHT = float(os.environ.get("POSE_RESIDUAL_REG_WEIGHT", "0.001"))


def build_model(cfg, args, device):
    m = cfg.model
    model = CurEventPoseRefineModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        event_count_cmax=float(getattr(m, "event_count_cmax", 3.0)),
        pixel_refiner_hidden=int(getattr(m, "pixel_refiner_hidden", 64)),
        pixel_refine_log_limit=float(getattr(m, "pixel_refine_log_limit", .30)),
        pixel_refiner_delay=int(getattr(m, "pixel_refiner_delay", 0)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .0015)),
        alignment_confidence_tau=.10, hdr_token_bottleneck=256,
        hdr_warmup_steps=0, normal_refine_iterations=1, normal_refine_step_limit=.05,
        c_delay_steps=int(getattr(m, "c_delay_steps", 1000)),
        c_transition_steps=int(getattr(m, "c_transition_steps", 1000)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
        pose_refiner_delay=int(getattr(m, "pose_refiner_delay", 1000)),
        pose_refiner_transition=int(getattr(m, "pose_refiner_transition", 1000)),
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    compatible = {key: value for key, value in state.items()
                  if key in own and own[key].shape == value.shape}
    loaded = model.load_state_dict(compatible, strict=False)
    required = [key for key in loaded.missing_keys
                if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"missing frozen VGGT weights: {required[:10]}")
    print("[POSE-REFINE] frozen camera head + delayed event-conditioned SE(3) residual", flush=True)
    return model.to(device)


def _relative_pose_loss(output, views):
    pose = torch.stack([item["camera_pose"] for item in output.ress], 1).float()
    height, width = output.ress[0]["depth"].shape[-3:-1]
    pred, _ = fe.pose_encoding_to_c2w(pose, image_size_hw=(height, width))
    gt = fe.ensure_homogeneous(fe.stack_view_field(views, "camera_pose").to(pred).float())
    valid_fields = [view.get("pose_valid") for view in views]
    if all(torch.is_tensor(value) for value in valid_fields):
        valid = torch.stack(valid_fields, 1).to(pred.device).bool()
        while valid.ndim > 2:
            valid = valid.any(-1)
    else:
        valid = torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool)

    pred_rel = torch.linalg.inv(pred[:, :1]) @ pred
    gt_rel = torch.linalg.inv(gt[:, :1]) @ gt
    mask = valid & valid[:, :1]
    mask[:, 0] = False
    if not mask.any():
        zero = pose.sum() * 0.0
        return zero, zero, zero
    translation = F.smooth_l1_loss(
        pred_rel[..., :3, 3], gt_rel[..., :3, 3], beta=.01, reduction="none"
    ).mean(-1)
    relative_rotation = (
        pred_rel[..., :3, :3].transpose(-1, -2) @ gt_rel[..., :3, :3]
    )
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.) * .5)
    rotation = torch.acos(cosine.clamp(-1. + 1.e-6, 1. - 1.e-6))
    return translation[mask].mean(), rotation[mask].mean(), mask.sum()


class PoseObjective:
    def __init__(self, base, phase):
        self.base, self.phase = base, phase

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        translation, rotation, pairs = _relative_pose_loss(output, views)
        residual_t = torch.stack(
            [item["pose_residual_translation"] for item in output.ress], 1
        )
        residual_r = torch.stack(
            [item["pose_residual_rotation"] for item in output.ress], 1
        )
        regularizer = residual_t.square().mean() + residual_r.square().mean()
        if self.phase == "adapter":
            result.loss = result.loss + POSE_WEIGHT * (
                translation + POSE_ROT_WEIGHT * rotation
            ) + POSE_REG_WEIGHT * regularizer
        result.details.update(
            pose_translation=translation,
            pose_rotation_rad=rotation,
            pose_valid_pairs=pairs,
            pose_residual_regularizer=regularizer,
            loss=result.loss,
        )
        return result


def criterion_for(args, phase):
    # Preserve the latest adjacent-view event-dNormal consistency objective,
    # then append pose refinement supervision rather than replacing it.
    return PoseObjective(cross_view.criterion_for(args, phase), phase)


def configure_phase(model, phase, train_heads_a=False):
    refiner_first.configure_phase(model, phase, train_heads_a)
    step = int(getattr(model, "_dual_alignment_step", 0))
    if phase == "adapter" and step >= model.pose_refiner_delay:
        model.pose_residual_head.requires_grad_(True)
        print(
            f"[POSE-REFINE/adapter] trainable; step={step}, "
            f"coupling ramp={model.pose_refiner_transition}",
            flush=True,
        )


def optimizer_for(model, phase, args):
    optimizer = refiner_first.optimizer_for(model, phase, args)
    if phase == "adapter":
        optimizer.add_param_group({
            "params": list(model.pose_residual_head.parameters()),
            "lr": .30 * args.lr,
        })
    return optimizer


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = refiner_first.prepare_pair
    pipeline.build_alternating_phase_schedule = alternating.schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = cross_view.save_visual
    pipeline.capture_runtime_state = alternating.capture_runtime_state
    pipeline.restore_runtime_state = alternating.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = CurEventPoseRefineModel
    values = refiner_first._force(sys.argv[1:] if argv is None else argv)
    print(
        f"[POSE-REFINE] weight={POSE_WEIGHT}, rotation_weight={POSE_ROT_WEIGHT}; "
        "relative-pose supervision, first-view gauge anchor",
        flush=True,
    )
    pipeline.main(values)


if __name__ == "__main__":
    main()
