"""Add events after a de-gridded RGB/depth head warmup.

This experiment separates two failure modes:
1. DPT head patch phase artifacts are first handled by the head-degrid warmup.
2. Events are then introduced as a reliability/detail gate, not as a global
   residual writer.
"""

import inspect
import sys
from pathlib import Path

import hydra
from accelerate import Accelerator as HFAccelerator
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from mul_loss_fine.finetune_mul_loss_detail_gt_geo_event_teacher import (  # noqa: E402
    build_geo_teacher_loader,
)
from mul_loss_fine.geo_contribution_event_loss import make_configured_geo_contribution_loss  # noqa: E402
from mul_loss_fine.launcher import configure_mul_loss_cfg  # noqa: E402


class EventBatchAccelerator(HFAccelerator):
    def __init__(self, *args, **kwargs):
        signature = inspect.signature(HFAccelerator.__init__)
        if "even_batches" in signature.parameters:
            kwargs.setdefault("even_batches", False)
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
            # Older accelerate versions do not expose even_batches in the
            # constructor, but prepare_data_loader still checks the instance
            # attribute when a custom batch_sampler has no batch_size.
            try:
                self.even_batches = False
            except Exception:
                pass
            dataloader_config = getattr(self, "dataloader_config", None)
            if dataloader_config is not None and hasattr(dataloader_config, "even_batches"):
                dataloader_config.even_batches = False


def _configure_event_plus_heads(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    train_tokens = ("event_encoder", "event_detail_refiner")
    for name, param in model.named_parameters():
        if any(token in name for token in train_tokens):
            param.requires_grad = True

    # Keep the de-gridded head adaptive. If this is frozen, event residuals have
    # to fight the head output instead of being absorbed into a stable predictor.
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
        ROOT_DIR / "checkpoints" / "mul_loss_detail_gt_uniform" / "checkpoint-last.pth",
        ROOT_DIR / "ckpt" / "model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            cfg.pretrained = str(candidate)
            return


def _prepare_cfg(cfg):
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

    cfg.model.variant = "temporal_reliability_v2"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "after_head_event_hidden_dim", 32))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "after_head_residual_scale", 0.035))
    cfg.model.event_gate_downsample = int(getattr(cfg.model, "after_head_event_gate_downsample", 1))
    cfg.model.event_gate_smooth_kernel = int(getattr(cfg.model, "after_head_event_gate_smooth_kernel", 3))
    cfg.model.event_reliability_floor = float(getattr(cfg.model, "after_head_event_reliability_floor", 0.18))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "after_head_event_reliability_init_bias", -0.5))
    cfg.model.proposal_depth_lowpass = bool(getattr(cfg.model, "after_head_proposal_depth_lowpass", True))
    cfg.model.proposal_use_depth_hf = bool(getattr(cfg.model, "after_head_proposal_use_depth_hf", False))
    cfg.model.event_proposal_weight = float(getattr(cfg.model, "after_head_event_proposal_weight", 0.65))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "after_head_event_delta_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = bool(getattr(cfg.model, "after_head_event_delta_patch_zero_mean", True))
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "after_head_event_delta_patch_size", 14))
    cfg.model.final_degrid_strength = float(getattr(cfg.model, "after_head_final_degrid_strength", 0.0))
    cfg.model.final_degrid_kernel = int(getattr(cfg.model, "after_head_final_degrid_kernel", 9))
    cfg.model.exposure_forward_batch_chunk = int(getattr(cfg.model, "after_head_forward_batch_chunk", 1))

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
        "residual_smooth_weight": 0.0,
        "residual_second_order_weight": 0.0,
        "residual_abs_weight": 0.0,
        "final_grid_weight": 0.02,
        "final_phase_weight": 0.01,
        "final_grid_patch_size": 14,
        "final_grid_band": 1,
        "final_grid_detail_threshold": 0.02,
        "v2_residual_target_weight": 0.20,
        "v2_gate_reliability_weight": 0.25,
        "v2_gate_need_floor": 0.08,
        "v2_gate_positive_boost": 2.5,
        "v2_temporal_quality_floor": 0.50,
        "v2_counterfactual_weight": 0.30,
        "v2_counterfactual_margin": 0.12,
        "v2_counterfactual_negative_weight": 0.40,
        "v2_ldr_final_depth_weight": 0.05,
        "v2_ldr_final_normal_weight": 0.05,
        "v2_ldr_correction_weight": 0.10,
        "v2_ldr_base_weight": 0.05,
        "v2_non_detail_smooth_weight": 0.04,
        "v2_non_detail_second_order_weight": 0.04,
        "v2_target_detail_threshold": 0.02,
        "geo_teacher_ldr_id": getattr(cfg.data, "geo_teacher_ldr_id", "ev_10"),
        "geo_event_target_weight": 0.30,
        "geo_event_reject_weight": 0.15,
        "geo_teacher_consistency_weight": 0.10,
        "geo_event_delta_weight": 0.40,
        "geo_teacher_boost": 0.8,
        "geo_detail_threshold": 0.02,
        "geo_positive_floor": 0.18,
        "geo_negative_margin": 0.20,
    }

    cfg = configure_mul_loss_cfg(
        _prepare_cfg(cfg),
        weights=weights,
        exp_name="mul_loss_detail_gt_event_after_head_degrid",
    )
    fe.Accelerator = EventBatchAccelerator
    fe.build_event_loader = build_geo_teacher_loader
    fe.configure_trainable_params = _configure_event_plus_heads
    fe.EventSupervisedLoss = make_configured_geo_contribution_loss(cfg)

    print(
        "Event-after-head-degrid training: initialize from de-gridded heads; "
        "train event_encoder/event_detail_refiner plus depth_head/point_head; "
        "events write high-pass zero-mean detail residuals through reliability gates."
    )
    print(
        f"teacher={cfg.data.geo_teacher_ldr_id}, students={cfg.data.geo_student_ldr_ids}, "
        f"eval_ldr={cfg.data.eval_ldr_event_id}, pretrained={cfg.pretrained}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
