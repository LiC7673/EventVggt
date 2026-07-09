"""DSEC repair finetune: image-guided event detail with conservative logging.

This is a DSEC-specific train entry.  It keeps DSEC self-contained because the
paired-token ReliabilityUNet is trained on the reflective indoor dataset and is
not assumed to transfer to driving scenes.
"""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from dsec_exp.common import build_dsec_loader
from mul_loss_fine.image_guided_event_reliability_loss import (
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg


ROOT = Path(__file__).resolve().parents[1]


def _freeze_for_event(model, _cfg):
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "event_detail_refiner" in name:
            parameter.requires_grad = True
    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(True)


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.model, cfg.train, cfg.loss, cfg.data, cfg.vis):
        OmegaConf.set_struct(branch, False)
    cfg.approach = "full_img_reliability"
    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(cfg.data.event_resize_bins)
    cfg.model.head_frames_chunk_size = int(getattr(cfg.model, "head_frames_chunk_size", 1))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "repair_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "repair_residual_scale", 0.05))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "repair_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = True
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "repair_abs_limit", 0.04))
    cfg.model.event_reliability_gate_enabled = True
    cfg.model.event_reliability_gate_floor = float(getattr(cfg.model, "repair_gate_floor", 0.15))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "repair_gate_bias", -0.5))

    cfg.loss.pose_weight = 0.0
    cfg.loss.points_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.normal_weight = float(getattr(cfg.loss, "repair_normal_weight", 0.15))
    cfg.loss.align_depth_scale = bool(getattr(cfg.loss, "align_depth_scale", False))

    cfg.validate_each_epoch = False
    cfg.skip_final_eval = True
    cfg.eval_every_steps = 0
    cfg.save_every_steps = int(getattr(cfg, "save_every_steps", 1000))
    cfg.vis.save_every_steps = int(getattr(cfg.vis, "save_every_steps", 3000))
    if str(getattr(cfg, "exp_name", "")) in {"", "dsec_full_img_reliability"}:
        cfg.exp_name = "dsec_repair_full_img_reliability"
    cfg.output_dir = str(Path(cfg.save_dir) / str(cfg.exp_name))
    cfg.logdir = str(Path(cfg.output_dir) / "logs")
    return cfg


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="finetune_dsec_event.yaml")
def run(cfg: OmegaConf):
    cfg = _prepare_cfg(cfg)
    weights = {
        "mv_normal_weight": 0.0,
        "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0,
        "mv_orient_weight": 0.0,
        "detail_gt_normal_weight": 0.35,
        "detail_gt_hf_weight": 1.2,
        "detail_gt_grad_weight": 1.2,
        "detail_gt_event_boost": 1.5,
        "detail_gt_threshold": 0.02,
        "detail_gt_normal_source": "depth",
        "mv_event_support_mode": "temporal_polarity",
        "mv_event_threshold": 0.20,
        "mv_event_dilate_kernel": 1,
        "mv_event_blur_kernel": 1,
        "residual_smooth_weight": 0.04,
        "residual_second_order_weight": 0.03,
        "residual_abs_weight": 0.02,
        "final_grid_weight": 0.02,
        "final_phase_weight": 0.01,
        "final_grid_patch_size": 14,
        "img_event_reliability_weight": 0.35,
        "img_event_reject_weight": 0.15,
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
    print(
        f"DSEC repair: image-guided event reliability, output={cfg.output_dir}, root={cfg.data.root}",
        flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
