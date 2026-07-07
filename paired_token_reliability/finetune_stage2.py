from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_paired_token_reliability_detail import StreamVGGT
from mul_loss_fine.launcher import configure_mul_loss_cfg
from real_reliability_stage.finetune_stage2_vggt import (
    _build_scene_disjoint_loader,
    _prepare_cfg as _prepare_base_stage2_cfg,
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
        reliability_dilate_kernel=int(cfg.model.reliability_dilate_kernel),
        reliability_frame_chunk_size=int(cfg.model.reliability_frame_chunk_size),
        reliability_rgb_input_range=str(cfg.model.reliability_rgb_input_range),
    )


def _configure_trainable(model, _cfg):
    model.requires_grad_(False)
    model.event_detail_refiner.base_refiner.requires_grad_(True)
    # The internal head is unused as a gate, but its features remain part of the
    # original refiner implementation; keep its output layer frozen.
    model.event_detail_refiner.base_refiner.reliability_head.requires_grad_(False)
    model.depth_head.requires_grad_(True)
    model.point_head.requires_grad_(True)
    model.event_detail_refiner.reliability_net.requires_grad_(False).eval()


def _snapshot(outdir):
    destination = Path(outdir) / "code" / "paired_token_reliability"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (
        ROOT / "paired_token_reliability" / "finetune_stage2.py",
        ROOT / "paired_token_reliability" / "train_reliability.py",
        ROOT / "paired_token_reliability" / "export_targets.py",
        ROOT / "eventvggt" / "models" / "streamvggt_paired_token_reliability_detail.py",
    )
    for source in sources:
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    return str(destination)


def _prepare_cfg(cfg):
    cfg = _prepare_base_stage2_cfg(cfg)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.variant = "paired_token_reliability_detail"
    checkpoint = Path(str(getattr(
        cfg.model,
        "reliability_checkpoint",
        ROOT / "abl_event_exp/paired_token_reliability/reliability_net/checkpoint-best.pth",
    ))).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    cfg.model.reliability_checkpoint = str(checkpoint.resolve())
    cfg.model.reliability_gate_floor = float(getattr(cfg.model, "reliability_gate_floor", 0.15))
    cfg.model.reliability_dilate_kernel = int(getattr(cfg.model, "reliability_dilate_kernel", 3))
    cfg.model.residual_postfilter_kernel = 1
    cfg.model.residual_postfilter_strength = 0.0
    cfg.model.causal_output_gate = False
    cfg.exp_name = str(getattr(cfg, "exp_name", "paired_token_reliability_stage2"))
    if cfg.exp_name in {"event_finetune_LDR5", "stage2_reliability_residual_train12_test4"}:
        cfg.exp_name = "paired_token_reliability_stage2"
    cfg.save_dir = str(ROOT / "abl_event_exp/paired_token_reliability")
    cfg.output_dir = str(Path(cfg.save_dir) / cfg.exp_name)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")
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
        "residual_smooth_weight": 0.02,
        "residual_second_order_weight": 0.01,
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
        "[paired-token reliability Stage 2] full voxel -> refiner -> residual gate; "
        f"R frozen={cfg.model.reliability_checkpoint} output={cfg.output_dir}",
        flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()

