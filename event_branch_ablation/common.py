"""Shared full-img-reliability configuration for additive-event experiments."""

from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_additive_decomposition_detail import (
    StreamVGGT as AdditiveDecompositionStreamVGGT,
)
from eventvggt.models.streamvggt_geometry_motion_detail import (
    StreamVGGT as GeometryMotionStreamVGGT,
)
from mul_loss_fine.image_guided_event_reliability_loss import (
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg

from event_branch_ablation.data import (
    build_full_decomposition_loader,
    build_geometry_motion_loader,
)
from event_branch_ablation.loss import make_additive_token_loss


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT_DIR / "abl_event_exp"

FULL_RELIABILITY_WEIGHTS = {
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
    "detail_gt_salient_hf_weight": 0.0,
    "detail_gt_salient_mag_weight": 0.0,
    "detail_gt_salient_presence_weight": 0.0,
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
    "img_event_reliability_weight": 0.30,
    "img_event_reject_weight": 0.12,
    "img_event_geometry_threshold": 0.02,
    "img_event_event_threshold": 0.20,
    "img_event_image_support_floor": 0.35,
    "img_event_saturation_reject_boost": 1.5,
}


def _set_paths(cfg, exp_name: str) -> None:
    cfg.exp_name = exp_name
    cfg.save_dir = str(OUTPUT_ROOT)
    cfg.output_dir = str(OUTPUT_ROOT / exp_name)
    cfg.logdir = str(OUTPUT_ROOT / exp_name / "logs")


def _safe_save_current_code(outdir: str):
    """Snapshot code without recursively copying abl_event_exp into itself."""
    timestamp = datetime.datetime.now().strftime("%m_%d-%H-%M-%S")
    destination = Path(outdir) / "code" / timestamp
    shutil.copytree(
        ROOT_DIR,
        destination,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "abl_event_exp*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return os.fspath(destination)


def _prepare_cfg(cfg, *, exp_name: str, variant: str):
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data"):
        OmegaConf.set_struct(getattr(cfg, branch_name), False)
    _set_paths(cfg, exp_name)
    original = ROOT_DIR / "ckpt" / "model.pt"
    current = str(getattr(cfg, "pretrained", "") or "")
    if current in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        cfg.pretrained = str(original) if original.exists() else "./ckpt/model.pt"

    cfg.model.variant = variant
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.035
    cfg.model.event_delta_highpass_kernel = 9
    cfg.model.event_delta_patch_zero_mean = True
    cfg.model.event_delta_patch_size = 14
    cfg.model.event_delta_abs_limit = 0.025
    cfg.model.event_reliability_gate_enabled = True
    cfg.model.event_reliability_gate_floor = 0.20
    cfg.model.event_reliability_init_bias = 0.0
    cfg.model.decomposition_hidden_dim = int(getattr(cfg.model, "decomposition_hidden_dim", 24))
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.data.return_normal_gt = True
    cfg.data.random_train_ldr = True
    cfg.data.eval_ldr_event_id = str(
        getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "ev_5"))
    )
    cfg.data.ldr_event_id = "random"
    cfg.data.additive_event_root = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    cfg.epochs = max(int(getattr(cfg, "epochs", 10)), 20)
    cfg.lr = min(float(getattr(cfg, "lr", 1.0e-4)), 4.0e-5)
    return configure_mul_loss_cfg(cfg, weights=FULL_RELIABILITY_WEIGHTS, exp_name=exp_name)


def _model_kwargs(cfg) -> dict:
    return {
        "img_size": cfg.model.img_size,
        "patch_size": cfg.model.patch_size,
        "embed_dim": cfg.model.embed_dim,
        "event_hidden_dim": cfg.model.event_hidden_dim,
        "head_frames_chunk_size": int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        "event_num_bins": int(cfg.model.event_num_bins),
        "event_count_cmax": float(getattr(cfg.model, "event_count_cmax", 3.0)),
        "residual_scale": float(cfg.model.refiner_residual_scale),
        "residual_highpass_kernel": int(cfg.model.event_delta_highpass_kernel),
        "residual_patch_zero_mean": bool(cfg.model.event_delta_patch_zero_mean),
        "residual_patch_size": int(cfg.model.event_delta_patch_size),
        "residual_abs_limit": float(cfg.model.event_delta_abs_limit),
        "reliability_gate_enabled": True,
        "reliability_gate_floor": float(cfg.model.event_reliability_gate_floor),
        "reliability_init_bias": float(cfg.model.event_reliability_init_bias),
        "refine_points": bool(getattr(cfg.model, "refiner_refine_points", True)),
        "use_checkpoint": bool(getattr(cfg.model, "refiner_use_checkpoint", True)),
    }


def _build_geometry_model(cfg):
    return GeometryMotionStreamVGGT(**_model_kwargs(cfg))


def _build_decomposition_model(cfg):
    kwargs = _model_kwargs(cfg)
    kwargs["decomposition_hidden_dim"] = int(cfg.model.decomposition_hidden_dim)
    return AdditiveDecompositionStreamVGGT(**kwargs)


def _configure_trainable(model, cfg) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith("event_detail_refiner.") or name.startswith("event_branch_decomposer."):
            parameter.requires_grad = True


def _experiment_name(cfg, default: str) -> str:
    current = str(getattr(cfg, "exp_name", "") or "")
    return default if current in {"", "event_finetune_LDR5"} else current
    for module_name in ("depth_head", "point_head"):
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad = True


def train_geometry_motion(cfg) -> None:
    cfg = _prepare_cfg(
        cfg,
        exp_name=_experiment_name(cfg, "geometry_motion_full_img_reliability"),
        variant="geometry_motion_full_img_reliability",
    )
    fe.build_event_loader = build_geometry_motion_loader
    fe.build_event_model = _build_geometry_model
    fe.configure_trainable_params = _configure_trainable
    fe.EventSupervisedLoss = make_configured_image_guided_event_reliability_loss(cfg)
    fe.save_current_code = _safe_save_current_code
    print(f"[geometry-motion] output={cfg.output_dir}, epochs={cfg.epochs}, lr={cfg.lr}")
    fe.train(cfg)


def train_full_decomposition(cfg) -> None:
    cfg = _prepare_cfg(
        cfg,
        exp_name=_experiment_name(cfg, "full_to_additive_tokens_img_reliability"),
        variant="full_to_additive_tokens_img_reliability",
    )
    cfg.loss.branch_token_weight = float(getattr(cfg.loss, "branch_token_weight", 0.50))
    cfg.loss.branch_geometry_weight = float(getattr(cfg.loss, "branch_geometry_weight", 1.00))
    cfg.loss.branch_material_weight = float(getattr(cfg.loss, "branch_material_weight", 0.75))
    cfg.loss.branch_noise_weight = float(getattr(cfg.loss, "branch_noise_weight", 0.50))
    cfg.loss.branch_consistency_weight = float(getattr(cfg.loss, "branch_consistency_weight", 0.10))
    fe.build_event_loader = build_full_decomposition_loader
    fe.build_event_model = _build_decomposition_model
    fe.configure_trainable_params = _configure_trainable
    fe.EventSupervisedLoss = make_additive_token_loss(cfg)
    fe.save_current_code = _safe_save_current_code
    print(f"[full-decomposition] output={cfg.output_dir}, epochs={cfg.epochs}, lr={cfg.lr}")
    fe.train(cfg)
