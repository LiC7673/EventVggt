"""Fine-tune RGB-only or image-guided EventVGGT on DSEC train scenes."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
import finetune_no_event as nf
from dsec_exp.common import build_dsec_loader
from mul_loss_fine.image_guided_event_reliability_loss import (
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg


ROOT = Path(__file__).resolve().parents[1]


def _freeze_for_event(model, cfg):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "event_detail_refiner" in name:
            parameter.requires_grad = True
    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def _prepare_common(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.model, cfg.train, cfg.loss, cfg.data):
        OmegaConf.set_struct(branch, False)
    cfg.loss.pose_weight = 0.0
    cfg.loss.points_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.normal_weight = float(getattr(cfg.loss, "normal_weight", 0.10))
    # Preserve the Hydra/config choice so raw-metric and scale-aligned DSEC
    # runs can be compared explicitly. YAML keeps the default disabled.
    cfg.loss.align_depth_scale = bool(getattr(cfg.loss, "align_depth_scale", False))
    cfg.model.head_frames_chunk_size = int(getattr(cfg.model, "head_frames_chunk_size", 1))
    # The held-out DSEC test directory is never exposed to the training loop.
    # The loop's optional monitor loader uses val again; final test evaluation
    # is launched only after training by run_dsec_finetune_and_test.sh.
    cfg.validate_each_epoch = False
    cfg.validation_monitor = "depth_loss"
    return cfg


def _run_rgb(cfg):
    cfg.model.variant = "base"
    cfg.train.unfreeze_heads = True
    cfg.train.unfreeze_aggregator_blocks = bool(getattr(cfg.train, "unfreeze_aggregator_blocks", False))
    nf.build_rgb_loader = lambda local_cfg, split="train": build_dsec_loader(
        local_cfg, "train" if split == "train" else "val", rgb_only=True
    )
    print("DSEC approach=rgb: pure VGGT RGB baseline; train scenes train, test scenes evaluate.")
    nf.train(cfg)


def _run_event(cfg):
    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(cfg.data.event_resize_bins)
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "imgrel_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "imgrel_refiner_residual_scale", 0.035))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "imgrel_event_delta_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = True
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "imgrel_event_delta_abs_limit", 0.025))
    cfg.model.event_reliability_gate_enabled = True
    cfg.model.event_reliability_gate_floor = 0.20
    cfg.model.event_reliability_init_bias = 0.0
    weights = {
        "mv_normal_weight": 0.0,
        "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0,
        "mv_orient_weight": 0.0,
        "detail_gt_normal_weight": 0.25,
        "detail_gt_hf_weight": 1.0,
        "detail_gt_grad_weight": 1.0,
        "detail_gt_event_boost": 1.5,
        "detail_gt_threshold": 0.02,
        "detail_gt_normal_source": "depth",
        "mv_event_support_mode": "temporal_polarity",
        "mv_event_threshold": 0.20,
        "mv_event_dilate_kernel": 1,
        "mv_event_blur_kernel": 1,
        "residual_smooth_weight": 0.02,
        "residual_second_order_weight": 0.02,
        "residual_abs_weight": 0.01,
        "final_grid_weight": 0.02,
        "final_phase_weight": 0.01,
        "final_grid_patch_size": 14,
        "img_event_reliability_weight": 0.30,
        "img_event_reject_weight": 0.12,
        "img_event_geometry_threshold": 0.02,
        "img_event_event_threshold": 0.20,
        "img_event_image_support_floor": 0.35,
        "img_event_saturation_reject_boost": 1.5,
    }
    cfg = configure_mul_loss_cfg(cfg, weights=weights, exp_name=str(cfg.exp_name))
    fe.build_event_loader = lambda local_cfg, split="train": build_dsec_loader(
        local_cfg, "train" if split == "train" else "val", rgb_only=False
    )
    fe.configure_trainable_params = _freeze_for_event
    fe.EventSupervisedLoss = make_configured_image_guided_event_reliability_loss(cfg)
    print("DSEC approach=full_img_reliability: temporal event detail plus GT/image-guided reliability.")
    fe.train(cfg)


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="finetune_dsec_event.yaml")
def run(cfg: OmegaConf):
    cfg = _prepare_common(cfg)
    approach = str(getattr(cfg, "approach", "full_img_reliability")).lower()
    if approach == "rgb":
        _run_rgb(cfg)
    elif approach in {"event", "full", "full_img_reliability"}:
        _run_event(cfg)
    else:
        raise ValueError(f"Unknown DSEC approach: {approach}")


if __name__ == "__main__":
    run()
