"""Train the same geometry-adapter Stage 2 under a selected ablation mode."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from ab_st1_st2 import METHODS
from ab_st1_st2.model import AblationStreamVGGT
from real_reliability_stage.finetune_stage2_vggt import _build_scene_disjoint_loader
from stage2_geometry_adapter import finetune as base
from stage2_geometry_adapter.loss import make_geometry_adapter_loss


ROOT = Path(__file__).resolve().parents[1]


def _build_model(cfg):
    return AblationStreamVGGT(
        img_size=int(cfg.model.img_size),
        patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        event_hidden_dim=int(cfg.model.adapter_event_hidden_dim),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        event_num_bins=int(cfg.model.event_num_bins),
        event_count_cmax=float(cfg.model.event_count_cmax),
        stage1_checkpoint=str(cfg.model.stage1_contribution_checkpoint),
        event_pyramid_channels=int(cfg.model.adapter_event_pyramid_channels),
        adapter_hidden_channels=int(cfg.model.adapter_hidden_channels),
        ablation_method=str(cfg.model.ablation_method),
        saturation_threshold=float(cfg.model.ablation_saturation_threshold),
    )


def _snapshot(outdir):
    destination = Path(outdir) / "code" / "ab_st1_st2"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "ab_st1_st2").glob("*"):
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    return str(destination)


def _prepare(cfg):
    cfg = base._prepare_cfg(cfg)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    method = str(getattr(cfg.model, "ablation_method", "ours"))
    if method not in METHODS:
        raise ValueError(f"Unknown method={method!r}; choices={METHODS}")
    if method == "rgb_only":
        raise ValueError("rgb_only uses the frozen base checkpoint and must not be finetuned")
    cfg.model.ablation_method = method
    cfg.model.ablation_saturation_threshold = float(
        getattr(cfg.model, "ablation_saturation_threshold", 0.98)
    )
    cfg.data.train_initial_scene_idx = int(
        getattr(cfg.data, "ablation_train_initial_scene_idx", 0)
    )
    cfg.data.train_scene_count = int(getattr(cfg.data, "ablation_train_scene_count", 20))
    cfg.data.test_initial_scene_idx = int(
        getattr(cfg.data, "ablation_test_initial_scene_idx", 20)
    )
    cfg.data.test_scene_count = int(getattr(cfg.data, "ablation_test_scene_count", 5))
    cfg.data.train_holdout_frame_count = 0
    cfg.data.heldout_test_frame_count = int(
        getattr(cfg.data, "heldout_test_frame_count", 120)
    )
    cfg.train.adapter_phase = "A"
    cfg.train.adapter_train_contribution = False
    output = ROOT / "experiments" / "ablation" / method
    cfg.exp_name = "stage2"
    cfg.save_dir = str(output)
    cfg.output_dir = str(output / "stage2")
    cfg.logdir = str(output / "stage2" / "logs")
    return cfg


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="finetune_event.yaml")
def run(cfg):
    cfg = _prepare(cfg)
    fe.build_event_loader = _build_scene_disjoint_loader
    fe.build_event_model = _build_model
    fe.configure_trainable_params = base._configure_trainable
    fe.build_optimizer_params = base._optimizer_groups
    fe.save_current_code = _snapshot
    fe.EventSupervisedLoss = make_geometry_adapter_loss(cfg)
    print(
        f"[fast ablation Stage2] method={cfg.model.ablation_method} "
        f"train scenes=20 test scenes=5 output={cfg.output_dir}",
        flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
