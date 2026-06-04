"""Simple temporal-event detail residual with high-pass constraints.

This is the "no-v2" path: no reverse/swap reliability, no geo teacher
counterfactual objective. Events may write a bounded high-frequency residual,
while patch-zero-mean filtering prevents low-frequency drag and token-grid
imprints from re-entering the prediction.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from mul_loss_fine.launcher import configure_mul_loss_cfg, make_configured_loss  # noqa: E402


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


def _resolve_pretrained(cfg) -> None:
    current = str(getattr(cfg, "pretrained", "") or "")
    if current not in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        return

    candidates = [
        ROOT_DIR / "checkpoints" / "mul_loss_detail_gt_head_degrid" / "checkpoint-last.pth",
        ROOT_DIR / "checkpoints" / "mul_loss_detail_gt_temporal_detail" / "checkpoint-last.pth",
        ROOT_DIR / "checkpoints" / "mul_loss_detail_gt_uniform" / "checkpoint-last.pth",
        ROOT_DIR / "ckpt" / "model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            cfg.pretrained = str(candidate)
            return


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.model, cfg.train, cfg.loss):
        OmegaConf.set_struct(branch, False)

    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "hf_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "hf_refiner_residual_scale", 0.035))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "hf_event_delta_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = bool(getattr(cfg.model, "hf_event_delta_patch_zero_mean", True))
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "hf_event_delta_patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "hf_event_delta_abs_limit", 0.025))

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.loss.pose_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = 1.0
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

    cfg = configure_mul_loss_cfg(
        _prepare_cfg(cfg),
        weights=weights,
        exp_name="mul_loss_detail_gt_temporal_detail_hf",
    )
    fe.configure_trainable_params = _configure_event_refiner_and_depth_heads
    fe.EventSupervisedLoss = make_configured_loss(cfg)
    print(
        "Temporal-detail-HF training: no v2 reliability/counterfactual; "
        "events write bounded high-pass patch-zero-mean residuals; "
        f"pretrained={cfg.pretrained}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
