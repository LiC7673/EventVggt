"""Train exposure-invariant temporal event gating with paired LDR observations."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from launcher import configure_mul_loss_cfg

import finetune_event as fe
from exposure_invariant_ldr_loss import make_configured_exposure_invariant_loss
from finetune_mul_ldr_event import _to_list, build_mul_ldr_loader


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.train, False)
    OmegaConf.set_struct(cfg.loss, False)
    OmegaConf.set_struct(cfg.vis, False)

    eval_ldr = getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "5"))
    cfg.data.ldr_event_id = "random"
    cfg.data.eval_ldr_event_id = eval_ldr
    cfg.data.mul_ldr_train_ids = getattr(cfg.data, "mul_ldr_train_ids", ["ev_2", "ev_5", "ev_10"])
    cfg.data.mul_ldr_exposures_per_sample = int(getattr(cfg.data, "mul_ldr_exposures_per_sample", 2))
    cfg.data.mul_ldr_scenes_per_batch = int(getattr(cfg.data, "mul_ldr_scenes_per_batch", 1))
    # Two LDR copies already double the per-step sample count. Four views and
    # sequential coarse forwards keep this experiment practical on two cards.
    cfg.data.num_views = int(getattr(cfg.data, "mul_ldr_num_views", 4))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "test_max_batches", 2))
    cfg.vis.test_num_views = int(getattr(cfg.vis, "test_num_views", cfg.data.num_views))

    cfg.model.variant = "temporal_exposure_invariant"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.01
    cfg.model.event_gate_downsample = 4
    cfg.model.exposure_match_dim = int(getattr(cfg.model, "exposure_match_dim", 8))
    cfg.model.exposure_agreement_floor = float(getattr(cfg.model, "exposure_agreement_floor", 0.25))
    cfg.model.exposure_forward_batch_chunk = int(getattr(cfg.model, "exposure_forward_batch_chunk", 1))

    # The coarse RGB geometry stays fixed. The inherited gated refiner and new
    # agreement branch adapt on top of the strongest single-LDR checkpoint.
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    if str(getattr(cfg, "pretrained", "")) in {"", "./ckpt/model.pt"}:
        cfg.pretrained = "./checkpoints/mul_loss_detail_gt_temporal_gated/checkpoint-last.pth"

    weights = {
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
        "ldr_feature_consistency_weight": 0.10,
        "ldr_event_match_weight": 0.05,
        "ldr_output_depth_weight": 0.10,
        "ldr_output_normal_weight": 0.10,
        "ldr_event_reliability_weight": 0.05,
        "ldr_output_base_weight": 0.10,
        "ldr_reliability_detail_threshold": 0.02,
        "ldr_reliability_target_floor": 0.25,
    }
    cfg = configure_mul_loss_cfg(
        cfg,
        weights=weights,
        exp_name="mul_loss_detail_gt_temporal_gated_multildr",
    )
    fe.build_event_loader = build_mul_ldr_loader
    fe.EventSupervisedLoss = make_configured_exposure_invariant_loss(cfg)
    print(
        "Multi-LDR exposure-invariant temporal gate: "
        f"train_ids={_to_list(cfg.data.mul_ldr_train_ids)}, "
        f"exposures_per_sample={cfg.data.mul_ldr_exposures_per_sample}, "
        f"num_views={cfg.data.num_views}, eval_ldr={cfg.data.eval_ldr_event_id}, "
        f"pretrained={cfg.pretrained}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
