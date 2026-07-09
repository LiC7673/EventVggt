from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_paired_token_reliability_repair import StreamVGGT
from mul_loss_fine.launcher import configure_mul_loss_cfg
from paired_token_reliability.finetune_stage2 import (
    _build_scene_disjoint_loader,
    _prepare_cfg as _prepare_paired_cfg,
)
from real_reliability_stage.stage2_loss import make_stage2_reliability_weighted_loss


ROOT = Path(__file__).resolve().parents[1]


def _build_model(cfg):
    return StreamVGGT(
        img_size=int(cfg.model.img_size),
        patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        event_hidden_dim=int(cfg.model.event_hidden_dim),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        event_num_bins=int(cfg.model.event_num_bins),
        event_count_cmax=float(cfg.model.event_count_cmax),
        residual_scale=float(cfg.model.refiner_residual_scale),
        residual_highpass_kernel=int(cfg.model.event_delta_highpass_kernel),
        residual_patch_zero_mean=bool(cfg.model.event_delta_patch_zero_mean),
        residual_patch_size=int(cfg.model.event_delta_patch_size),
        residual_abs_limit=float(cfg.model.event_delta_abs_limit),
        refine_points=True,
        use_checkpoint=bool(getattr(cfg.model, "refiner_use_checkpoint", True)),
        reliability_checkpoint=str(cfg.model.reliability_checkpoint),
        reliability_base_channels=int(cfg.model.reliability_base_channels),
        reliability_gate_floor=float(cfg.model.reliability_gate_floor),
        reliability_dilate_kernel=1,
        reliability_frame_chunk_size=int(cfg.model.reliability_frame_chunk_size),
        reliability_rgb_input_range=str(cfg.model.reliability_rgb_input_range),
        repair_reliability_threshold=float(cfg.model.repair_reliability_threshold),
        repair_reliability_temperature=float(cfg.model.repair_reliability_temperature),
        repair_reliability_top_fraction=float(cfg.model.repair_reliability_top_fraction),
        repair_event_support_threshold=float(cfg.model.repair_event_support_threshold),
        repair_event_support_dilate_kernel=int(cfg.model.repair_event_support_dilate_kernel),
        repair_event_support_floor=float(cfg.model.repair_event_support_floor),
        repair_residual_gain=float(cfg.model.repair_residual_gain),
        repair_output_abs_limit=float(cfg.model.repair_output_abs_limit),
    )


def _configure_trainable(model, _cfg):
    model.requires_grad_(False)
    model.event_detail_refiner.base_refiner.requires_grad_(True)
    model.event_detail_refiner.base_refiner.reliability_head.requires_grad_(False)
    model.depth_head.requires_grad_(True)
    model.point_head.requires_grad_(True)
    model.event_detail_refiner.reliability_net.requires_grad_(False).eval()


def _snapshot(outdir):
    destination = Path(outdir) / "code" / "repair_reliability"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        ROOT / "repair_reliability" / "finetune_stage2_repair.py",
        ROOT / "repair_reliability" / "evaluate_stage2_repair.py",
        ROOT / "repair_reliability" / "run_repair_pipeline.sh",
        ROOT / "eventvggt" / "models" / "streamvggt_paired_token_reliability_repair.py",
        ROOT / "real_reliability_stage" / "stage2_loss.py",
    )
    for source in sources:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    return str(destination)


def _prepare_cfg(cfg):
    cfg = _prepare_paired_cfg(cfg)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.loss, False)
    cfg.model.variant = "paired_token_reliability_repair"
    cfg.exp_name = str(getattr(cfg, "exp_name", "paired_token_reliability_repair"))
    cfg.save_dir = str(ROOT / "abl_event_exp/paired_token_reliability_repair")
    cfg.output_dir = str(Path(cfg.save_dir) / cfg.exp_name)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")

    # Make reliability more selective.  The previous positive ratio near 0.9
    # diluted event supervision and made the residual branch timid.
    cfg.model.reliability_gate_floor = float(getattr(cfg.model, "reliability_gate_floor", 0.20))
    cfg.model.repair_reliability_threshold = float(
        getattr(cfg.model, "repair_reliability_threshold", 0.45)
    )
    cfg.model.repair_reliability_temperature = float(
        getattr(cfg.model, "repair_reliability_temperature", 0.18)
    )
    cfg.model.repair_reliability_top_fraction = float(
        getattr(cfg.model, "repair_reliability_top_fraction", 0.80)
    )
    cfg.model.repair_event_support_threshold = float(
        getattr(cfg.model, "repair_event_support_threshold", 0.0)
    )
    cfg.model.repair_event_support_dilate_kernel = int(
        getattr(cfg.model, "repair_event_support_dilate_kernel", 5)
    )
    cfg.model.repair_event_support_floor = float(
        getattr(cfg.model, "repair_event_support_floor", 0.25)
    )
    cfg.model.repair_residual_gain = float(getattr(cfg.model, "repair_residual_gain", 2.0))
    cfg.model.repair_output_abs_limit = float(
        getattr(cfg.model, "repair_output_abs_limit", 0.06)
    )

    # The last run predicted only about 24% of the target delta.  Increase the
    # target and gradient terms, but keep smooth/abs regularizers active below.
    cfg.loss.stage2_residual_target_weight = float(
        getattr(cfg.loss, "stage2_residual_target_weight", 2.0)
    )
    cfg.loss.stage2_residual_gradient_weight = float(
        getattr(cfg.loss, "stage2_residual_gradient_weight", 3.0)
    )
    cfg.loss.stage2_target_reliability_floor = float(
        getattr(cfg.loss, "stage2_target_reliability_floor", 0.10)
    )
    cfg.loss.stage2_target_abs_limit = float(getattr(cfg.loss, "stage2_target_abs_limit", 0.06))
    cfg.epochs = max(int(getattr(cfg, "epochs", 20)), 20)
    cfg.eval_every_steps = int(getattr(cfg, "eval_every_steps", 0))
    cfg.vis.save_every_steps = int(getattr(cfg.vis, "save_every_steps", 3000))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "test_max_batches", 1))
    return cfg


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="finetune_event.yaml")
def run(cfg):
    cfg = _prepare_cfg(cfg)
    weights = {
        "mv_normal_weight": 0.0,
        "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0,
        "mv_orient_weight": 0.0,
        "detail_gt_normal_weight": 0.25,
        "detail_gt_hf_weight": 1.0,
        "detail_gt_grad_weight": 1.0,
        "detail_gt_event_boost": 1.5,
        "detail_gt_threshold": 0.02,
        "detail_gt_weight_power": 1.0,
        "detail_gt_normal_source": "depth",
        "detail_gt_chunk_size": 1,
        "residual_smooth_weight": 0.04,
        "residual_second_order_weight": 0.02,
        "residual_abs_weight": 0.02,
        "final_grid_weight": 0.01,
        "final_phase_weight": 0.005,
        "final_grid_patch_size": 14,
        "final_grid_band": 1,
        "final_grid_detail_threshold": 0.02,
    }
    cfg = configure_mul_loss_cfg(cfg, weights=weights, exp_name=cfg.exp_name)
    if not Path(cfg.model.reliability_checkpoint).is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint missing: {cfg.model.reliability_checkpoint}")
    fe.build_event_loader = _build_scene_disjoint_loader
    fe.build_event_model = _build_model
    fe.configure_trainable_params = _configure_trainable
    fe.save_current_code = _snapshot
    fe.EventSupervisedLoss = make_stage2_reliability_weighted_loss(cfg)
    print(
        "[repair Stage 2] selective reliability gate + stronger event residual; "
        f"R={cfg.model.reliability_checkpoint} output={cfg.output_dir}",
        flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
