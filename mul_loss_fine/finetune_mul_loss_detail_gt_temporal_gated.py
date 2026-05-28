from pathlib import Path

import hydra
from omegaconf import OmegaConf

from launcher import launch_mul_loss


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    cfg.model.variant = "temporal_gated_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.01
    cfg.model.event_gate_downsample = 4

    # Preserve the strong uniform geometry model; only the event-gated detail
    # adapter is allowed to add bounded improvements.
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    if str(getattr(cfg, "pretrained", "")) in {"", "./ckpt/model.pt"}:
        cfg.pretrained = "./checkpoints/mul_loss_detail_gt_uniform/checkpoint-last.pth"

    launch_mul_loss(
        cfg,
        exp_name="mul_loss_detail_gt_temporal_gated",
        weights={
            "mv_normal_weight": 0.0,
            "mv_presence_weight": 0.0,
            "mv_hf_weight": 0.0,
            "mv_orient_weight": 0.0,
            "detail_gt_normal_weight": 0.2,
            "detail_gt_hf_weight": 0.35,
            "detail_gt_grad_weight": 0.35,
            "detail_gt_event_boost": 0.0,
            "detail_gt_threshold": 0.02,
            "detail_gt_normal_source": "depth",
            "detail_gt_salient_hf_weight": 0.0,
            "detail_gt_salient_mag_weight": 0.0,
            "detail_gt_salient_presence_weight": 0.0,
            "mv_event_support_mode": "temporal_polarity",
            "mv_event_threshold": 0.20,
            "mv_event_dilate_kernel": 1,
            "mv_event_blur_kernel": 1,
            "mv_event_power": 2.0,
            "mv_event_top_fraction": 0.20,
            "residual_smooth_weight": 0.15,
            "residual_second_order_weight": 0.20,
            "residual_abs_weight": 0.01,
            "residual_smooth_alpha": 10.0,
            "final_grid_weight": 0.08,
            "final_phase_weight": 0.04,
            "final_grid_patch_size": 14,
            "final_grid_band": 1,
            "final_grid_detail_threshold": 0.02,
        },
    )


if __name__ == "__main__":
    run()
