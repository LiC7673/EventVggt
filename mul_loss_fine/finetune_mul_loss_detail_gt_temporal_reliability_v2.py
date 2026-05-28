"""Train temporal reliability V2 with paired LDR inputs and GT residual teaching."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from launcher import configure_mul_loss_cfg

import finetune_event as fe
from finetune_mul_ldr_event import _to_list, build_mul_ldr_loader
from reliability_residual_v2_loss import make_configured_reliability_v2_loss


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.data, cfg.model, cfg.train, cfg.loss, cfg.vis):
        OmegaConf.set_struct(branch, False)

    eval_ldr = getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "5"))
    cfg.data.ldr_event_id = "random"
    cfg.data.eval_ldr_event_id = eval_ldr
    cfg.data.mul_ldr_train_ids = getattr(cfg.data, "mul_ldr_train_ids", ["ev_2", "ev_5", "ev_10"])
    cfg.data.mul_ldr_exposures_per_sample = int(getattr(cfg.data, "mul_ldr_exposures_per_sample", 2))
    cfg.data.mul_ldr_scenes_per_batch = int(getattr(cfg.data, "mul_ldr_scenes_per_batch", 1))
    cfg.data.num_views = int(getattr(cfg.data, "mul_ldr_num_views", 4))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "test_max_batches", 2))
    cfg.vis.test_num_views = int(getattr(cfg.vis, "test_num_views", cfg.data.num_views))

    cfg.model.variant = "temporal_reliability_v2"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.02
    cfg.model.event_gate_downsample = 4
    cfg.model.event_reliability_floor = float(getattr(cfg.model, "event_reliability_floor", 0.0))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "event_reliability_init_bias", 0.0))
    cfg.model.exposure_forward_batch_chunk = int(getattr(cfg.model, "exposure_forward_batch_chunk", 1))

    # Keep the strong coarse predictor fixed; the event-conditioned correction
    # is the only trainable geometry adjustment in this ablation.
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    if str(getattr(cfg, "pretrained", "")) in {"", "./ckpt/model.pt"}:
        cfg.pretrained = "./checkpoints/mul_loss_detail_gt_temporal_gated/checkpoint-last.pth"

    weights = {
        "mv_normal_weight": 0.0,
        "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0,
        "mv_orient_weight": 0.0,
        "detail_gt_normal_weight": 0.20,
        "detail_gt_hf_weight": 0.30,
        "detail_gt_grad_weight": 0.30,
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
        # V2 smoothness is masked away from real GT details; do not apply the
        # old uniform residual smoother a second time.
        "residual_smooth_weight": 0.0,
        "residual_second_order_weight": 0.0,
        "residual_abs_weight": 0.0,
        "final_grid_weight": 0.08,
        "final_phase_weight": 0.04,
        "final_grid_patch_size": 14,
        "final_grid_band": 1,
        "final_grid_detail_threshold": 0.02,
        "v2_residual_target_weight": 0.70,
        "v2_gate_reliability_weight": 0.20,
        "v2_gate_need_floor": 0.10,
        "v2_gate_positive_boost": 2.0,
        "v2_temporal_quality_floor": 0.25,
        "v2_counterfactual_weight": 0.20,
        "v2_counterfactual_margin": 0.08,
        "v2_ldr_final_depth_weight": 0.10,
        "v2_ldr_final_normal_weight": 0.10,
        "v2_ldr_correction_weight": 0.20,
        "v2_ldr_base_weight": 0.10,
        "v2_non_detail_smooth_weight": 0.03,
        "v2_non_detail_second_order_weight": 0.03,
        "v2_target_detail_threshold": 0.02,
    }
    cfg = configure_mul_loss_cfg(
        cfg,
        weights=weights,
        exp_name="mul_loss_detail_gt_temporal_reliability_v2",
    )
    fe.build_event_loader = build_mul_ldr_loader
    fe.EventSupervisedLoss = make_configured_reliability_v2_loss(cfg)
    print(
        "Temporal reliability V2: "
        f"train_ids={_to_list(cfg.data.mul_ldr_train_ids)}, "
        f"exposures_per_sample={cfg.data.mul_ldr_exposures_per_sample}, "
        f"num_views={cfg.data.num_views}, eval_ldr={cfg.data.eval_ldr_event_id}, "
        f"residual_scale={cfg.model.refiner_residual_scale}, pretrained={cfg.pretrained}"
    )
    print(
        "Event-dependence guarantee: final log-depth residual is multiplied by "
        "an event-derived gate; zero event voxel yields zero V2 correction."
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
