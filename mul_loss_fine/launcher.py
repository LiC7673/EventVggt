import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from mul_loss_fine.event_supported_mv_loss import MultiViewEventSupervisedLoss  # noqa: E402


DEFAULT_EVENT_LOSS = {
    "mv_normal_weight": 0.0,
    "mv_presence_weight": 0.0,
    "mv_hf_weight": 0.0,
    "mv_orient_weight": 0.0,
    "mv_presence_margin": 0.04,
    "mv_event_blur_kernel": 5,
    "mv_event_dilate_kernel": 3,
    "mv_event_threshold": 0.02,
    "mv_event_power": 1.0,
    "mv_event_top_fraction": 0.0,
    "mv_event_support_mode": "abs",
    "mv_hf_kernel": 7,
    "mv_bidirectional": False,
    "mv_max_pairs": 4,
    "mv_detach_warp_grid": True,
    "mv_projection_pose": "gt",
    "detail_gt_normal_weight": 0.0,
    "detail_gt_hf_weight": 0.0,
    "detail_gt_grad_weight": 0.0,
    "detail_gt_event_boost": 0.5,
    "detail_gt_threshold": 0.03,
    "detail_gt_weight_power": 1.0,
    "detail_gt_normal_source": "auto",
    "detail_gt_salient_hf_weight": 0.0,
    "detail_gt_salient_mag_weight": 0.0,
    "detail_gt_salient_presence_weight": 0.0,
    "detail_gt_salient_threshold": 0.35,
    "detail_gt_salient_power": 2.0,
    "detail_gt_salient_presence_ratio": 0.8,
    "detail_gt_chunk_size": 1,
}


def _set_if_missing(cfg, key, value):
    if not hasattr(cfg, key):
        setattr(cfg, key, value)


def configure_mul_loss_cfg(cfg, *, weights, exp_name):
    OmegaConf.set_struct(cfg, False)
    for key, value in DEFAULT_EVENT_LOSS.items():
        _set_if_missing(cfg.loss, key, value)

    # By default each ablation script must own its weights. Otherwise a shared
    # yaml/CLI value such as loss.mv_hf_weight can silently make all scripts run
    # with the same loss, which defeats the ablation.
    respect_existing = bool(getattr(cfg.loss, "respect_existing_mul_loss_weights", False))
    for key, value in weights.items():
        if not respect_existing:
            setattr(cfg.loss, key, value)
        elif not hasattr(cfg.loss, key):
            setattr(cfg.loss, key, value)

    if str(getattr(cfg, "exp_name", "")) == "event_finetune_LDR5":
        cfg.exp_name = exp_name

    OmegaConf.resolve(cfg)
    return cfg


def make_configured_loss(cfg):
    class ConfiguredMultiViewEventLoss(MultiViewEventSupervisedLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                mv_normal_weight=float(getattr(cfg.loss, "mv_normal_weight", 0.0)),
                mv_presence_weight=float(getattr(cfg.loss, "mv_presence_weight", 0.0)),
                mv_hf_weight=float(getattr(cfg.loss, "mv_hf_weight", 0.0)),
                mv_orient_weight=float(getattr(cfg.loss, "mv_orient_weight", 0.0)),
                mv_presence_margin=float(getattr(cfg.loss, "mv_presence_margin", 0.04)),
                mv_event_blur_kernel=int(getattr(cfg.loss, "mv_event_blur_kernel", 5)),
                mv_event_dilate_kernel=int(getattr(cfg.loss, "mv_event_dilate_kernel", 3)),
                mv_event_threshold=float(getattr(cfg.loss, "mv_event_threshold", 0.02)),
                mv_event_power=float(getattr(cfg.loss, "mv_event_power", 1.0)),
                mv_event_top_fraction=float(getattr(cfg.loss, "mv_event_top_fraction", 0.0)),
                mv_event_support_mode=str(getattr(cfg.loss, "mv_event_support_mode", "abs")),
                mv_hf_kernel=int(getattr(cfg.loss, "mv_hf_kernel", 7)),
                mv_bidirectional=bool(getattr(cfg.loss, "mv_bidirectional", False)),
                mv_max_pairs=int(getattr(cfg.loss, "mv_max_pairs", 4)),
                mv_detach_warp_grid=bool(getattr(cfg.loss, "mv_detach_warp_grid", True)),
                mv_projection_pose=str(getattr(cfg.loss, "mv_projection_pose", "gt")),
                detail_gt_normal_weight=float(getattr(cfg.loss, "detail_gt_normal_weight", 0.0)),
                detail_gt_hf_weight=float(getattr(cfg.loss, "detail_gt_hf_weight", 0.0)),
                detail_gt_grad_weight=float(getattr(cfg.loss, "detail_gt_grad_weight", 0.0)),
                detail_gt_event_boost=float(getattr(cfg.loss, "detail_gt_event_boost", 0.5)),
                detail_gt_threshold=float(getattr(cfg.loss, "detail_gt_threshold", 0.03)),
                detail_gt_weight_power=float(getattr(cfg.loss, "detail_gt_weight_power", 1.0)),
                detail_gt_normal_source=str(getattr(cfg.loss, "detail_gt_normal_source", "auto")),
                detail_gt_salient_hf_weight=float(getattr(cfg.loss, "detail_gt_salient_hf_weight", 0.0)),
                detail_gt_salient_mag_weight=float(getattr(cfg.loss, "detail_gt_salient_mag_weight", 0.0)),
                detail_gt_salient_presence_weight=float(
                    getattr(cfg.loss, "detail_gt_salient_presence_weight", 0.0)
                ),
                detail_gt_salient_threshold=float(getattr(cfg.loss, "detail_gt_salient_threshold", 0.35)),
                detail_gt_salient_power=float(getattr(cfg.loss, "detail_gt_salient_power", 2.0)),
                detail_gt_salient_presence_ratio=float(
                    getattr(cfg.loss, "detail_gt_salient_presence_ratio", 0.8)
                ),
                detail_gt_chunk_size=int(getattr(cfg.loss, "detail_gt_chunk_size", 1)),
                **kwargs,
            )

    return ConfiguredMultiViewEventLoss


def launch_mul_loss(cfg, *, weights, exp_name):
    cfg = configure_mul_loss_cfg(cfg, weights=weights, exp_name=exp_name)
    fe.EventSupervisedLoss = make_configured_loss(cfg)
    print(
        "Mul-loss ablation "
        f"{exp_name}: "
        f"mv_normal={float(getattr(cfg.loss, 'mv_normal_weight', 0.0)):.4f} "
        f"mv_presence={float(getattr(cfg.loss, 'mv_presence_weight', 0.0)):.4f} "
        f"mv_hf={float(getattr(cfg.loss, 'mv_hf_weight', 0.0)):.4f} "
        f"mv_orient={float(getattr(cfg.loss, 'mv_orient_weight', 0.0)):.4f} "
        f"detail_gt_normal={float(getattr(cfg.loss, 'detail_gt_normal_weight', 0.0)):.4f} "
        f"detail_gt_hf={float(getattr(cfg.loss, 'detail_gt_hf_weight', 0.0)):.4f} "
        f"detail_gt_grad={float(getattr(cfg.loss, 'detail_gt_grad_weight', 0.0)):.4f} "
        f"detail_gt_salient_hf={float(getattr(cfg.loss, 'detail_gt_salient_hf_weight', 0.0)):.4f} "
        f"detail_gt_salient_mag={float(getattr(cfg.loss, 'detail_gt_salient_mag_weight', 0.0)):.4f} "
        f"detail_gt_salient_presence={float(getattr(cfg.loss, 'detail_gt_salient_presence_weight', 0.0)):.4f} "
        f"detail_gt_event_boost={float(getattr(cfg.loss, 'detail_gt_event_boost', 0.0)):.4f} "
        f"detail_gt_normal_source={str(getattr(cfg.loss, 'detail_gt_normal_source', 'auto'))} "
        f"event_support_mode={str(getattr(cfg.loss, 'mv_event_support_mode', 'abs'))} "
        f"event_top_fraction={float(getattr(cfg.loss, 'mv_event_top_fraction', 0.0)):.3f} "
        f"max_pairs={int(getattr(cfg.loss, 'mv_max_pairs', 4))} "
        f"projection_pose={str(getattr(cfg.loss, 'mv_projection_pose', 'gt'))}"
    )
    fe.train(cfg)
