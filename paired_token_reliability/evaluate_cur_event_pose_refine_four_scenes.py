"""Evaluate the synthetic-trained cur-event pose-refinement model."""
from __future__ import annotations

import sys

import finetune_event as fe
from ablation.eag3r_metrics_eval import cfg_from_checkpoint
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.cross_view_event_normal_consistency import (
    cross_view_patch_loss,
    save_patch_diagnostics,
)
from paired_token_reliability.linear_voxel_cur_event_pose_refine_model import (
    CurEventPoseRefineModel,
)
from paired_token_reliability import (
    evaluate_alternating_detail_first_fixed_four_scenes as evaluator,
)


def build_model(checkpoint, device, depth_scale):
    raw = torch_load(checkpoint)
    expected = CurEventPoseRefineModel.checkpoint_schema
    if raw.get("schema") != expected:
        raise ValueError(
            f"checkpoint schema={raw.get('schema')!r}, expected={expected!r}; "
            "use a checkpoint from train_linear_voxel_cur_event_pose_refine"
        )
    cfg = cfg_from_checkpoint(raw, None)
    m = cfg.model
    model = CurEventPoseRefineModel(
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
        normal_refine_step_limit=.05,
        c_delay_steps=int(getattr(m, "c_delay_steps", 1000)),
        c_transition_steps=int(getattr(m, "c_transition_steps", 1000)),
        event_hidden_dim=32, event_pyramid_channels=32,
        adapter_hidden_channels=64, contribution_channels=32,
        contribution_initial_value=.70,
        pose_refiner_hidden=int(getattr(m, "pose_refiner_hidden", 128)),
        pose_refiner_delay=int(getattr(m, "pose_refiner_delay", 1000)),
        pose_refiner_transition=int(getattr(m, "pose_refiner_transition", 1000)),
    )
    state = strip_module_prefix(fe.unwrap_state_dict(raw))
    model.load_state_dict(state, strict=True)
    runtime = raw.get("runtime_state") or raw.get(
        "trainer_state", {}
    ).get("runtime_state", {})
    model._dual_alignment_step = max(
        int(runtime.get("dual_alignment_step", 0)), 2500
    )
    model.set_confidence_stage("full")
    model.fixed_eval_depth_scale = float(depth_scale)
    print(
        f"[POSE synthetic eval] schema={expected}; cur_event=ON; "
        f"pose_refiner=ON; fixed_depth_scale={depth_scale}",
        flush=True,
    )
    return model.to(device).eval(), cfg


_base_save_visual = evaluator.save_visual


def save_visual(root, scene, exposure, index, views, output, depth_gt, valid,
                intrinsics, event_source_mode):
    _base_save_visual(
        root, scene, exposure, index, views, output, depth_gt, valid,
        intrinsics, event_source_mode,
    )
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
