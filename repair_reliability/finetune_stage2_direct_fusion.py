"""Stage 2: reliability-weighted events and RGB tokens share VGGT heads."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_reliability_direct_fusion import StreamVGGT
from mul_loss_fine.launcher import configure_mul_loss_cfg, make_configured_loss
from paired_token_reliability.finetune_stage2 import (
    _build_scene_disjoint_loader,
    _prepare_cfg as _prepare_paired_cfg,
)


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
        reliability_checkpoint=str(cfg.model.reliability_checkpoint),
        reliability_base_channels=int(cfg.model.reliability_base_channels),
        reliability_frame_chunk_size=int(cfg.model.reliability_frame_chunk_size),
        reliability_rgb_input_range=str(cfg.model.reliability_rgb_input_range),
        direct_event_hidden_dim=int(cfg.model.direct_event_hidden_dim),
        direct_fusion_scale=float(cfg.model.direct_fusion_scale),
        refine_points=False,
        use_checkpoint=bool(getattr(cfg.model, "refiner_use_checkpoint", True)),
    )


def _configure_trainable(model, _cfg):
    model.requires_grad_(False)
    model.event_token_encoder.requires_grad_(True)
    model.camera_head.requires_grad_(True)
    model.depth_head.requires_grad_(True)
    model.point_head.requires_grad_(True)
    model.input_reliability.reliability_net.requires_grad_(False).eval()


def _snapshot(outdir):
    destination = Path(outdir) / "code" / "direct_fusion"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (
        ROOT / "repair_reliability" / "finetune_stage2_direct_fusion.py",
        ROOT / "eventvggt" / "models" / "streamvggt_reliability_direct_fusion.py",
        ROOT / "repair_reliability" / "evaluate_stage2_direct_fusion.py",
    ):
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    return str(destination)


def _prepare_cfg(cfg):
    cfg = _prepare_paired_cfg(cfg)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.loss, False)
    cfg.model.variant = "reliability_direct_fusion"
    cfg.model.direct_event_hidden_dim = int(getattr(cfg.model, "direct_event_hidden_dim", 32))
    cfg.model.direct_fusion_scale = float(getattr(cfg.model, "direct_fusion_scale", 0.10))
    cfg.loss.pose_weight = float(getattr(cfg.loss, "direct_pose_weight", 1.0))
    cfg.loss.normal_weight = float(getattr(cfg.loss, "direct_normal_weight", 0.30))
    save_root = Path(str(getattr(cfg, "repair_save_dir", ROOT / "abl_event_exp/direct_fusion")))
    if not save_root.is_absolute():
        save_root = ROOT / save_root
    cfg.save_dir = str(save_root.resolve())
    cfg.output_dir = str(Path(cfg.save_dir) / cfg.exp_name)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")
    return cfg


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="finetune_event.yaml")
def run(cfg):
    cfg = _prepare_cfg(cfg)
    zero_weights = {
        "mv_normal_weight": 0.0, "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0, "mv_orient_weight": 0.0,
        "detail_gt_normal_weight": 0.0, "detail_gt_hf_weight": 0.0,
        "detail_gt_grad_weight": 0.0, "residual_smooth_weight": 0.0,
        "residual_second_order_weight": 0.0, "residual_abs_weight": 0.0,
        "final_grid_weight": 0.0, "final_phase_weight": 0.0,
    }
    cfg = configure_mul_loss_cfg(cfg, weights=zero_weights, exp_name=cfg.exp_name)
    if not Path(cfg.model.reliability_checkpoint).is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint missing: {cfg.model.reliability_checkpoint}")
    fe.build_event_loader = _build_scene_disjoint_loader
    fe.build_event_model = _build_model
    fe.configure_trainable_params = _configure_trainable
    fe.save_current_code = _snapshot
    fe.EventSupervisedLoss = make_configured_loss(cfg)
    print(
        "[direct fusion Stage 2] R*V -> polarity/temporal tokens + RGB tokens -> shared heads; "
        f"R={cfg.model.reliability_checkpoint} output={cfg.output_dir}", flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
