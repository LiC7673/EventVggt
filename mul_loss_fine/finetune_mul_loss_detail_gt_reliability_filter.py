"""Train event reliability as a filter, not as a residual generator."""

from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe
from finetune_mul_loss_detail_gt_geo_event_teacher import build_geo_teacher_loader
from geo_contribution_event_loss import make_configured_geo_contribution_loss
from launcher import configure_mul_loss_cfg


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.data, cfg.model, cfg.train, cfg.loss, cfg.vis):
        OmegaConf.set_struct(branch, False)

    eval_ldr = getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "ev_5"))
    cfg.data.ldr_event_id = "random"
    cfg.data.eval_ldr_event_id = eval_ldr
    cfg.data.geo_teacher_ldr_id = getattr(cfg.data, "geo_teacher_ldr_id", "ev_10")
    cfg.data.geo_student_ldr_ids = getattr(cfg.data, "geo_student_ldr_ids", ["ev_2", "ev_5"])
    cfg.data.geo_exposures_per_sample = int(getattr(cfg.data, "geo_exposures_per_sample", 2))
    cfg.data.geo_scenes_per_batch = int(getattr(cfg.data, "geo_scenes_per_batch", 1))
    cfg.data.num_views = int(getattr(cfg.data, "geo_num_views", 4))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "test_max_batches", 2))
    cfg.vis.test_num_views = int(getattr(cfg.vis, "test_num_views", cfg.data.num_views))

    cfg.model.variant = "reliability_filter_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.03
    cfg.model.event_gate_downsample = 2
    cfg.model.event_reliability_floor = float(getattr(cfg.model, "event_reliability_floor", 0.30))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "event_reliability_init_bias", 0.5))

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    if str(getattr(cfg, "pretrained", "")) in {"", "./ckpt/model.pt"}:
        preferred = Path("./checkpoints/mul_loss_detail_gt_temporal_detail/checkpoint-last.pth")
        fallback = Path("./checkpoints/mul_loss_detail_gt_uniform/checkpoint-last.pth")
        cfg.pretrained = str(preferred if preferred.exists() else fallback)

    weights = {
        "mv_normal_weight": 0.0,
        "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0,
        "mv_orient_weight": 0.0,
        # Match the strong detail supervision that made temporal_detail work,
        # while keeping events as a reliability filter rather than a writer.
        "detail_gt_normal_weight": 0.20,
        "detail_gt_hf_weight": 0.80,
        "detail_gt_grad_weight": 0.80,
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
        "residual_smooth_weight": 0.0,
        "residual_second_order_weight": 0.0,
        "residual_abs_weight": 0.0,
        "final_grid_weight": 0.02,
        "final_phase_weight": 0.01,
        "final_grid_patch_size": 14,
        "final_grid_band": 1,
        "final_grid_detail_threshold": 0.02,
        "v2_residual_target_weight": 0.0,
        "v2_gate_reliability_weight": 0.0,
        "v2_gate_need_floor": 0.10,
        "v2_gate_positive_boost": 2.0,
        "v2_temporal_quality_floor": 0.25,
        "v2_counterfactual_weight": 0.0,
        "v2_counterfactual_margin": 0.08,
        "v2_ldr_final_depth_weight": 0.0,
        "v2_ldr_final_normal_weight": 0.0,
        "v2_ldr_correction_weight": 0.0,
        "v2_ldr_base_weight": 0.10,
        "v2_non_detail_smooth_weight": 0.0,
        "v2_non_detail_second_order_weight": 0.0,
        "v2_target_detail_threshold": 0.02,
        "geo_teacher_ldr_id": cfg.data.geo_teacher_ldr_id,
        "geo_event_target_weight": 0.30,
        "geo_event_reject_weight": 0.02,
        # Reliability sees the same event voxel across LDRs, so teacher/student
        # consistency is not informative here.
        "geo_teacher_consistency_weight": 0.0,
        "geo_event_delta_weight": 0.0,
        "geo_teacher_boost": 0.5,
        "geo_detail_threshold": 0.02,
        "geo_positive_floor": 0.20,
        "geo_negative_margin": 0.25,
    }
    cfg = configure_mul_loss_cfg(
        cfg,
        weights=weights,
        exp_name="mul_loss_detail_gt_reliability_filter",
    )
    fe.build_event_loader = build_geo_teacher_loader
    fe.EventSupervisedLoss = make_configured_geo_contribution_loss(cfg)
    print(
        "Reliability-filter training: events only learn reliability; "
        f"teacher={cfg.data.geo_teacher_ldr_id}, students={cfg.data.geo_student_ldr_ids}, "
        f"eval_ldr={cfg.data.eval_ldr_event_id}, pretrained={cfg.pretrained}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
