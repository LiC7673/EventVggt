"""Train the spatial event pose refiner on top of the cur-event pipeline."""
from __future__ import annotations

import sys

import finetune_event as fe
from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_cur_event_refiner_first as refiner_first
from paired_token_reliability import train_linear_voxel_cur_event_cross_view_dn as cross_view
from paired_token_reliability.train_linear_voxel_cur_event_pose_refine import (
    PoseObjective,
)
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_cur_event_pose_refine_v2_model import (
    CurEventPoseRefineV2Model,
)


def build_model(cfg, args, device):
    m = cfg.model
    model = CurEventPoseRefineV2Model(
        img_size=int(m.img_size), patch_size=int(m.patch_size),
        embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        event_count_cmax=float(getattr(m, "event_count_cmax", 3.0)),
        pixel_refiner_hidden=int(getattr(m, "pixel_refiner_hidden", 64)),
        pixel_refine_log_limit=float(getattr(m, "pixel_refine_log_limit", .30)),
        pixel_refiner_delay=int(getattr(m, "pixel_refiner_delay", 0)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .0015)),
        alignment_confidence_tau=.10, hdr_token_bottleneck=256,
        hdr_warmup_steps=0, normal_refine_iterations=1,
        normal_refine_step_limit=.05, c_delay_steps=1000,
        c_transition_steps=1000, event_hidden_dim=32,
        event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
        pose_feature_dim=int(getattr(m, "pose_feature_dim", 96)),
        pose_refiner_hidden=int(getattr(m, "pose_refiner_hidden", 192)),
        pose_refiner_delay=int(getattr(m, "pose_refiner_delay", 0)),
        pose_refiner_transition=int(getattr(m, "pose_refiner_transition", 500)),
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in own and own[key].shape == value.shape
    }
    loaded = model.load_state_dict(compatible, strict=False)
    required = [
        key for key in loaded.missing_keys
        if key.startswith(("aggregator.", "camera_head."))
    ]
    if required:
        raise RuntimeError(f"missing frozen VGGT weights: {required[:10]}")
    print(
        "[POSE-V2] final pose = frozen RGB camera pose @ spatial-event SE(3); "
        "event layout and anchor-relative differences retained",
        flush=True,
    )
    return model.to(device)


def criterion_for(args, phase):
    return PoseObjective(cross_view.criterion_for(args, phase), phase)


def configure_phase(model, phase, train_heads_a=False):
    refiner_first.configure_phase(model, phase, train_heads_a)
    if phase == "adapter":
        model.pose_spatial_refiner.requires_grad_(True)
        print("[POSE-V2/adapter] spatial pose refiner trainable", flush=True)


def optimizer_for(model, phase, args):
    optimizer = refiner_first.optimizer_for(model, phase, args)
    if phase == "adapter":
        optimizer.add_param_group({
            "params": list(model.pose_spatial_refiner.parameters()),
            "lr": .5 * args.lr,
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
    pipeline.UnifiedGeometryContributionModel = CurEventPoseRefineV2Model
    pipeline.main(refiner_first._force(
        sys.argv[1:] if argv is None else argv
    ))


if __name__ == "__main__":
    main()
