import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from mul_loss_fine.launcher import configure_mul_loss_cfg, make_configured_loss  # noqa: E402


EVENT_KEYS = {
    "event_xy",
    "event_t",
    "event_p",
    "event_time_range",
    "event_voxel",
    "event_voxel_raw",
    "event_abs",
    "event_signed",
    "event_time_grad",
    "event_pos",
    "event_neg",
    "events",
    "debug_event_xy",
    "debug_event_t",
    "debug_event_p",
}


def _drop_event_fields(views):
    for view in views:
        for key in EVENT_KEYS:
            view.pop(key, None)
    return views


def _install_rgb_only_collate() -> None:
    original_collate = fe.event_multiview_collate

    def rgb_only_collate(batch):
        return _drop_event_fields(original_collate(batch))

    fe.event_multiview_collate = rgb_only_collate


def _configure_depth_point_heads_only(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    enabled = []
    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
        enabled.append(module_name)

    if not enabled:
        raise RuntimeError("No depth/point head module found to train.")


def _maybe_set_pretrained(cfg) -> None:
    current = str(getattr(cfg, "pretrained", "") or "")
    if current not in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        return

    candidates = [
        ROOT_DIR / "checkpoints" / "mul_loss_detail_gt_uniform" / "checkpoint-last.pth",
        ROOT_DIR / "checkpoints" / "mul_loss_detail_gt_temporal_detail" / "checkpoint-last.pth",
        ROOT_DIR / "ckpt" / "model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            cfg.pretrained = str(candidate)
            return


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg.model.variant = "base"
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False

    # Camera head is frozen in this diagnostic run, so keep pose out of the
    # optimized objective while still retaining depth and point supervision.
    cfg.loss.pose_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = 1.0
    cfg.loss.depth_second_order_weight = float(getattr(cfg.loss, "depth_second_order_weight", 0.0) or 0.0)
    cfg.loss.grid_suppress_weight = float(getattr(cfg.loss, "grid_suppress_weight", 0.0) or 0.0)

    # A slightly smaller LR than the event-refiner runs keeps the DPT heads from
    # overfitting patch phase noise during the first diagnostic pass.
    if float(getattr(cfg, "lr", 1.0e-4)) > 5.0e-5:
        cfg.lr = 5.0e-5

    _maybe_set_pretrained(cfg)
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
        "detail_gt_hf_weight": 0.8,
        "detail_gt_grad_weight": 0.8,
        "detail_gt_event_boost": 0.0,
        "detail_gt_threshold": 0.02,
        "detail_gt_weight_power": 1.0,
        "detail_gt_normal_source": "depth",
        "detail_gt_salient_hf_weight": 0.0,
        "detail_gt_salient_mag_weight": 0.0,
        "detail_gt_salient_presence_weight": 0.0,
        "detail_gt_chunk_size": 1,
        "residual_smooth_weight": 0.0,
        "residual_second_order_weight": 0.0,
        "residual_abs_weight": 0.0,
        "final_grid_weight": 0.10,
        "final_phase_weight": 0.05,
        "final_grid_patch_size": 14,
        "final_grid_band": 1,
        "final_grid_detail_threshold": 0.02,
        "mv_event_support_mode": "temporal_polarity",
        "mv_event_threshold": 0.20,
        "mv_event_dilate_kernel": 1,
        "mv_event_blur_kernel": 1,
        "mv_event_power": 2.0,
        "mv_event_top_fraction": 0.20,
    }

    cfg = configure_mul_loss_cfg(
        _prepare_cfg(cfg),
        weights=weights,
        exp_name="mul_loss_detail_gt_head_degrid",
    )
    fe.EventSupervisedLoss = make_configured_loss(cfg)
    fe.configure_trainable_params = _configure_depth_point_heads_only
    _install_rgb_only_collate()

    print(
        "Head-degrid diagnostic: RGB-only views, trainable modules=depth_head+point_head, "
        "frozen aggregator/camera/event modules, losses=detail_gt+final_antigrid."
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
