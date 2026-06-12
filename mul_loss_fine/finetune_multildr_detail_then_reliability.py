"""Stage-2 event reliability refinement on a Multi-LDR detail checkpoint.

The Multi-LDR + GT-detail model is treated as a frozen coarse geometry teacher.
Only the temporal event refiner is trained, so event cues can improve local
normals without changing the global depth or camera solution.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from fine_event.finetune_event_random_ldr import build_random_ldr_event_loader  # noqa: E402
from mul_loss_fine.launcher import configure_mul_loss_cfg  # noqa: E402
from mul_loss_fine.scale_preserving_event_reliability_loss import (  # noqa: E402
    make_configured_scale_preserving_event_reliability_loss,
)


DETAIL_RELIABILITY_WEIGHTS = {
    "mv_normal_weight": 0.0,
    "mv_presence_weight": 0.0,
    "mv_hf_weight": 0.0,
    "mv_orient_weight": 0.0,
    "detail_gt_normal_weight": 0.25,
    "detail_gt_hf_weight": 1.0,
    "detail_gt_grad_weight": 1.0,
    "detail_gt_event_boost": 1.25,
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
    "residual_smooth_weight": 0.01,
    "residual_second_order_weight": 0.01,
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
    "scale_preserve_weight": 2.0,
    "low_frequency_preserve_weight": 0.5,
    "non_detail_guard_weight": 0.25,
    "low_frequency_preserve_kernel": 31,
    "non_detail_guard_margin": 0.002,
    "scale_preserve_detail_threshold": 0.02,
}


def _freeze_coarse_train_event_refiner(model, cfg) -> None:
    del cfg
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "event_detail_refiner" in name:
            param.requires_grad = True


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data"):
        OmegaConf.set_struct(getattr(cfg, branch_name), False)

    teacher = ROOT_DIR / "checkpoints" / "ablation_multildr_detail_gt" / "checkpoint-last.pth"
    configured_pretrained = str(getattr(cfg, "pretrained", "") or "")
    if configured_pretrained in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        cfg.pretrained = str(teacher)
    if not Path(str(cfg.pretrained)).exists():
        raise FileNotFoundError(
            f"Multi-LDR detail checkpoint not found: {cfg.pretrained}. "
            "Train ablation_multildr_detail_gt first or pass pretrained=/path/to/checkpoint."
        )

    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "fusion_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "fusion_residual_scale", 0.020))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "fusion_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = True
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "fusion_delta_abs_limit", 0.015))
    cfg.model.event_reliability_gate_enabled = True
    cfg.model.event_reliability_gate_floor = float(getattr(cfg.model, "fusion_gate_floor", 0.10))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "fusion_gate_init_bias", -0.5))
    cfg.model.refiner_refine_points = True
    cfg.model.refiner_use_checkpoint = bool(getattr(cfg.model, "refiner_use_checkpoint", True))

    cfg.data.random_train_ldr = True
    cfg.data.eval_ldr_event_id = str(
        getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "ev_5"))
    )
    cfg.data.ldr_event_id = "random"
    cfg.data.return_normal_gt = True

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.loss.pose_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = 1.0
    cfg.lr = min(float(getattr(cfg, "lr", 2.0e-5)), 2.0e-5)
    cfg.epochs = max(int(getattr(cfg, "epochs", 10)), 10)
    return cfg


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    cfg = _prepare_cfg(cfg)
    cfg = configure_mul_loss_cfg(
        cfg,
        weights=DETAIL_RELIABILITY_WEIGHTS,
        exp_name="multildr_detail_then_reliability",
    )
    fe.build_event_loader = build_random_ldr_event_loader
    fe.configure_trainable_params = _freeze_coarse_train_event_refiner
    fe.EventSupervisedLoss = make_configured_scale_preserving_event_reliability_loss(cfg)
    print(
        "Stage-2 scale-preserving event refinement: "
        f"teacher={cfg.pretrained}, residual_scale={cfg.model.refiner_residual_scale}, "
        f"delta_limit={cfg.model.event_delta_abs_limit}, lr={cfg.lr}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
