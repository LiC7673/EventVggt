"""Paper ablation entry for exposure/detail/reliability experiments.

This script keeps all ablation variants in one Hydra entry so the runner can
launch consistent two-GPU jobs while changing only ``ablation_variant``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
import finetune_no_event as nf  # noqa: E402
from fine_event.finetune_event_random_ldr import build_random_ldr_event_loader  # noqa: E402
from mul_loss_fine.image_guided_event_reliability_loss import (  # noqa: E402
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg, make_configured_loss  # noqa: E402


DETAIL_WEIGHTS = {
    "mv_normal_weight": 0.0,
    "mv_presence_weight": 0.0,
    "mv_hf_weight": 0.0,
    "mv_orient_weight": 0.0,
    "detail_gt_normal_weight": 0.25,
    "detail_gt_hf_weight": 1.00,
    "detail_gt_grad_weight": 1.00,
    "detail_gt_event_boost": 0.0,
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
}

FULL_RELIABILITY_WEIGHTS = {
    **DETAIL_WEIGHTS,
    "detail_gt_event_boost": 1.50,
    "img_event_reliability_weight": 0.30,
    "img_event_reject_weight": 0.12,
    "img_event_geometry_threshold": 0.02,
    "img_event_event_threshold": 0.20,
    "img_event_image_support_floor": 0.35,
    "img_event_saturation_reject_boost": 1.5,
}


def _set_output_paths(cfg, exp_name: str) -> None:
    cfg.exp_name = exp_name
    save_dir = Path(str(getattr(cfg, "save_dir", "./checkpoints")))
    cfg.output_dir = str(save_dir / exp_name)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")


def _resolve_original_pretrained(cfg) -> None:
    current = str(getattr(cfg, "pretrained", "") or "")
    if current not in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        return
    original = ROOT_DIR / "ckpt" / "model.pt"
    cfg.pretrained = str(original) if original.exists() else "./ckpt/model.pt"


def _prepare_common(cfg, exp_name: str):
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data"):
        if hasattr(cfg, branch_name):
            OmegaConf.set_struct(getattr(cfg, branch_name), False)
    _set_output_paths(cfg, exp_name)
    _resolve_original_pretrained(cfg)
    cfg.data.return_normal_gt = True
    cfg.data.return_debug_event_fields = False
    return cfg


def _prepare_temporal_detail_model(cfg, *, reliability: bool) -> None:
    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "ablation_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "ablation_refiner_residual_scale", 0.035))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "ablation_event_delta_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = bool(getattr(cfg.model, "ablation_event_delta_patch_zero_mean", True))
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "ablation_event_delta_patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "ablation_event_delta_abs_limit", 0.025))
    cfg.model.event_reliability_gate_enabled = bool(reliability)
    cfg.model.event_reliability_gate_floor = float(getattr(cfg.model, "ablation_event_reliability_gate_floor", 0.20))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "ablation_event_reliability_init_bias", 0.0))
    cfg.model.refiner_refine_points = True
    cfg.model.refiner_use_checkpoint = bool(getattr(cfg.model, "refiner_use_checkpoint", True))


def _configure_event_refiner_and_depth_heads(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "event_detail_refiner" in name:
            param.requires_grad = True

    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True


def _use_random_ldr_loader(cfg) -> None:
    cfg.data.random_train_ldr = True
    cfg.data.eval_ldr_event_id = str(getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "auto")))
    cfg.data.ldr_event_id = "random"
    fe.build_event_loader = build_random_ldr_event_loader


def _apply_detail_loss_to_event(cfg, exp_name: str) -> None:
    configure_mul_loss_cfg(cfg, weights=DETAIL_WEIGHTS, exp_name=exp_name)
    fe.EventSupervisedLoss = make_configured_loss(cfg)


def _apply_detail_loss_to_rgb(cfg, exp_name: str) -> None:
    configure_mul_loss_cfg(cfg, weights=DETAIL_WEIGHTS, exp_name=exp_name)
    nf.EventSupervisedLoss = make_configured_loss(cfg)


def _apply_full_reliability_loss(cfg, exp_name: str) -> None:
    configure_mul_loss_cfg(cfg, weights=FULL_RELIABILITY_WEIGHTS, exp_name=exp_name)
    fe.EventSupervisedLoss = make_configured_image_guided_event_reliability_loss(cfg)


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    variant = str(getattr(cfg, "ablation_variant", "full_img_reliability")).lower()
    exp_name = str(getattr(cfg, "exp_name", f"ablation_{variant}"))
    if exp_name == "event_finetune_LDR5":
        exp_name = f"ablation_{variant}"

    cfg = _prepare_common(cfg, exp_name)

    if variant == "rgb_baseline":
        cfg.model.variant = "base"
        print(f"[ablation] {variant}: RGB-only baseline, exp={cfg.exp_name}")
        nf.train(cfg)
        return

    if variant == "rgb_detail_gt":
        cfg.model.variant = "base"
        _apply_detail_loss_to_rgb(cfg, cfg.exp_name)
        print(f"[ablation] {variant}: RGB-only + GT geometry-detail supervision, exp={cfg.exp_name}")
        nf.train(cfg)
        return

    if variant == "raw_event":
        cfg.model.variant = "base"
        print(f"[ablation] {variant}: raw event StreamVGGT, exp={cfg.exp_name}")
        fe.train(cfg)
        return

    if variant == "raw_event_detail_gt":
        cfg.model.variant = "base"
        _apply_detail_loss_to_event(cfg, cfg.exp_name)
        print(f"[ablation] {variant}: raw event + GT geometry-detail supervision, exp={cfg.exp_name}")
        fe.train(cfg)
        return

    if variant == "multildr":
        cfg.model.variant = "base"
        _use_random_ldr_loader(cfg)
        print(f"[ablation] {variant}: multi-LDR exposure sampling, exp={cfg.exp_name}")
        fe.train(cfg)
        return

    if variant == "multildr_detail_gt":
        cfg.model.variant = "base"
        _use_random_ldr_loader(cfg)
        _apply_detail_loss_to_event(cfg, cfg.exp_name)
        print(f"[ablation] {variant}: multi-LDR + GT geometry-detail supervision, exp={cfg.exp_name}")
        fe.train(cfg)
        return

    if variant == "full_img_reliability":
        _use_random_ldr_loader(cfg)
        _prepare_temporal_detail_model(cfg, reliability=True)
        cfg.train.unfreeze_heads = False
        cfg.train.unfreeze_aggregator_blocks = False
        if float(getattr(cfg, "lr", 1.0e-4)) > 4.0e-5:
            cfg.lr = 4.0e-5
        _apply_full_reliability_loss(cfg, cfg.exp_name)
        fe.configure_trainable_params = _configure_event_refiner_and_depth_heads
        print(
            f"[ablation] {variant}: multi-LDR + detail GT + image-guided event reliability, "
            f"exp={cfg.exp_name}, pretrained={cfg.pretrained}"
        )
        fe.train(cfg)
        return

    raise ValueError(
        f"Unknown ablation_variant={variant}. Expected one of: "
        "rgb_baseline, rgb_detail_gt, raw_event, raw_event_detail_gt, "
        "multildr, multildr_detail_gt, full_img_reliability"
    )


if __name__ == "__main__":
    run()
