"""Best temporal-detail reliability script with geometry/diffuse events only."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from diffuse_event_ablation.geometry_event_loader import build_geometry_event_loader  # noqa: E402
from mul_loss_fine.image_guided_event_reliability_loss import (  # noqa: E402
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg  # noqa: E402


def _configure_event_refiner_and_depth_heads(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "event_detail_refiner" in name:
            param.requires_grad = True
    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def _resolve_pretrained(cfg) -> None:
    current = str(getattr(cfg, "pretrained", "") or "")
    if current not in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        return
    original = ROOT_DIR / "ckpt" / "model.pt"
    cfg.pretrained = str(original) if original.exists() else "./ckpt/model.pt"


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data"):
        OmegaConf.set_struct(getattr(cfg, branch_name), False)

    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "geom_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "geom_refiner_residual_scale", 0.035))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "geom_event_delta_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = bool(getattr(cfg.model, "geom_event_delta_patch_zero_mean", True))
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "geom_event_delta_patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "geom_event_delta_abs_limit", 0.025))
    cfg.model.event_reliability_gate_enabled = True
    cfg.model.event_reliability_gate_floor = float(getattr(cfg.model, "geom_event_reliability_gate_floor", 0.20))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "geom_event_reliability_init_bias", 0.0))

    cfg.data.random_train_ldr = True
    cfg.data.eval_ldr_event_id = str(
        getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "ev_5"))
    )
    cfg.data.ldr_event_id = "random"
    cfg.data.additive_event_root = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    cfg.data.additive_event_branch = str(getattr(cfg.data, "additive_event_branch", "geometry_motion"))
    cfg.data.geometry_event_mask_dilate_kernel = int(
        getattr(cfg.data, "geometry_event_mask_dilate_kernel", 5)
    )
    cfg.data.return_normal_gt = True

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.loss.pose_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = 1.0
    cfg.epochs = max(int(getattr(cfg, "epochs", 10)), int(getattr(cfg, "geometry_event_epochs", 20)))
    if float(getattr(cfg, "lr", 1.0e-4)) > 4.0e-5:
        cfg.lr = 4.0e-5
    _resolve_pretrained(cfg)
    return cfg


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    weights = {
        "mv_normal_weight": 0.0,
        "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0,
        "mv_orient_weight": 0.0,
        "detail_gt_normal_weight": 0.25,
        "detail_gt_hf_weight": 1.00,
        "detail_gt_grad_weight": 1.00,
        "detail_gt_event_boost": 1.50,
        "detail_gt_threshold": 0.02,
        "detail_gt_weight_power": 1.0,
        "detail_gt_normal_source": "depth",
        "detail_gt_chunk_size": 1,
        "mv_event_support_mode": "temporal_polarity",
        "mv_event_threshold": 0.20,
        "mv_event_dilate_kernel": 3,
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
    cfg = configure_mul_loss_cfg(
        _prepare_cfg(cfg),
        weights=weights,
        exp_name="geometry_event_temporal_detail_img_reliability",
    )
    fe.build_event_loader = build_geometry_event_loader
    fe.configure_trainable_params = _configure_event_refiner_and_depth_heads
    fe.EventSupervisedLoss = make_configured_image_guided_event_reliability_loss(cfg)
    print(
        "Geometry-event oracle ablation: using additive geometry_motion events "
        f"instead of full events; dilation={cfg.data.geometry_event_mask_dilate_kernel}, "
        f"branch={cfg.data.additive_event_branch}, epochs={cfg.epochs}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
