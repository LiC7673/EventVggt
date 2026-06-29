"""Shared configuration for frozen and joint reliability finetuning stages."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

import finetune_event as fe
from mul_loss_fine.launcher import configure_mul_loss_cfg
from reliability_staged_finetune.data import build_staged_reliability_loader
from reliability_staged_finetune.loss import make_staged_reliability_loss
from reliability_staged_finetune.model import StreamVGGT


ROOT_DIR = Path(__file__).resolve().parents[1]


DETAIL_WEIGHTS = {
    "mv_normal_weight": 0.0,
    "mv_presence_weight": 0.0,
    "mv_hf_weight": 0.0,
    "mv_orient_weight": 0.0,
    "detail_gt_normal_weight": 0.25,
    "detail_gt_hf_weight": 1.0,
    "detail_gt_grad_weight": 1.0,
    "detail_gt_event_boost": 1.5,
    "detail_gt_threshold": 0.02,
    "detail_gt_weight_power": 1.0,
    "detail_gt_normal_source": "depth",
    "detail_gt_chunk_size": 1,
    "mv_event_support_mode": "temporal_polarity",
    "mv_event_threshold": 0.20,
    "mv_event_dilate_kernel": 1,
    "mv_event_blur_kernel": 1,
    "mv_event_power": 2.0,
    "mv_event_top_fraction": 0.20,
    "residual_smooth_weight": 0.02,
    "residual_second_order_weight": 0.02,
    "residual_abs_weight": 0.01,
    "residual_smooth_alpha": 10.0,
    "final_grid_weight": 0.02,
    "final_phase_weight": 0.01,
    "final_grid_patch_size": 14,
    "final_grid_band": 1,
    "final_grid_detail_threshold": 0.02,
}


def _build_model(cfg):
    return StreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=cfg.model.event_hidden_dim,
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        event_num_bins=int(getattr(cfg.model, "event_num_bins", cfg.data.event_resize_bins)),
        event_count_cmax=float(getattr(cfg.model, "event_count_cmax", 3.0)),
        residual_scale=float(getattr(cfg.model, "refiner_residual_scale", 0.035)),
        residual_highpass_kernel=int(getattr(cfg.model, "event_delta_highpass_kernel", 9)),
        residual_patch_zero_mean=bool(getattr(cfg.model, "event_delta_patch_zero_mean", True)),
        residual_patch_size=int(getattr(cfg.model, "event_delta_patch_size", 14)),
        residual_abs_limit=float(getattr(cfg.model, "event_delta_abs_limit", 0.025)),
        refine_points=bool(getattr(cfg.model, "refiner_refine_points", True)),
        use_checkpoint=bool(getattr(cfg.model, "refiner_use_checkpoint", True)),
        reliability_checkpoint=str(cfg.model.stage1_reliability_checkpoint),
        reliability_num_bins=int(getattr(cfg.model, "reliability_num_bins", 5)),
        reliability_base_channels=int(getattr(cfg.model, "reliability_base_channels", 32)),
        reliability_count_cmax=float(getattr(cfg.model, "reliability_count_cmax", 3.0)),
        freeze_reliability=bool(cfg.model.freeze_reliability),
    )


def _configure_trainable(model, cfg):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith("event_detail_refiner.") and not name.startswith(
            "event_detail_refiner.reliability_head."
        ):
            parameter.requires_grad = True
    for module_name in ("depth_head", "point_head"):
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad = True
    if not bool(cfg.model.freeze_reliability):
        for parameter in model.reliability_net.parameters():
            parameter.requires_grad = True


def _optimizer_groups(model, cfg):
    reliability = []
    geometry = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (reliability if name.startswith("reliability_net.") else geometry).append(parameter)
    groups = [{"params": geometry, "lr_scale": 1.0}]
    if reliability:
        groups.append(
            {
                "params": reliability,
                "lr_scale": float(getattr(cfg.train, "reliability_lr_scale", 0.25)),
            }
        )
    return groups


def prepare_cfg(cfg, *, stage: int):
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data"):
        OmegaConf.set_struct(getattr(cfg, branch_name), False)
    cfg.model.variant = "staged_reliability_temporal_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.035
    cfg.model.event_delta_highpass_kernel = 9
    cfg.model.event_delta_patch_zero_mean = True
    cfg.model.event_delta_patch_size = 14
    cfg.model.event_delta_abs_limit = 0.025
    cfg.model.reliability_num_bins = int(getattr(cfg.model, "reliability_num_bins", 5))
    cfg.model.reliability_base_channels = int(getattr(cfg.model, "reliability_base_channels", 32))
    cfg.model.freeze_reliability = stage == 2
    cfg.model.stage1_reliability_checkpoint = str(
        getattr(
            cfg.model,
            "stage1_reliability_checkpoint",
            ROOT_DIR / "checkpoints" / "reliability_net_stage1_scene12" / "checkpoint-best.pth",
        )
    )
    cfg.data.additive_event_root = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    cfg.data.return_normal_gt = True
    cfg.loss.pose_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = 1.0
    cfg.loss.staged_reliability_weight = 0.0 if stage == 2 else float(
        getattr(cfg.loss, "joint_reliability_weight", 0.30)
    )
    cfg.train.reliability_lr_scale = float(getattr(cfg.train, "reliability_lr_scale", 0.25))
    cfg.lr = min(float(cfg.lr), 4.0e-5 if stage == 2 else 1.0e-5)
    return configure_mul_loss_cfg(
        cfg,
        weights=DETAIL_WEIGHTS,
        exp_name=(
            "staged_reliability_stage2_frozen"
            if stage == 2
            else "staged_reliability_stage3_joint"
        ),
    )


def train_stage(cfg, *, stage: int):
    cfg = prepare_cfg(cfg, stage=stage)
    fe.build_event_model = _build_model
    fe.build_event_loader = build_staged_reliability_loader
    fe.configure_trainable_params = _configure_trainable
    fe.build_optimizer_params = _optimizer_groups
    fe.EventSupervisedLoss = make_staged_reliability_loss(cfg)
    print(
        f"Reliability stage {stage}: freeze_reliability={cfg.model.freeze_reliability}, "
        f"reliability_loss_weight={cfg.loss.staged_reliability_weight}, lr={cfg.lr}"
    )
    fe.train(cfg)
