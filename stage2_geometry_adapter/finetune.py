"""Finetune contribution-guided DPT geometry adapters without depth residuals."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from paired_token_reliability.common import torch_load
from paired_token_reliability.contribution_stage1 import MultiLdrEventContributionModel
from real_reliability_stage.finetune_stage2_vggt import _build_scene_disjoint_loader
from stage2_geometry_adapter.loss import make_geometry_adapter_loss
from stage2_geometry_adapter.model import StreamVGGT


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _build_model(cfg):
    return StreamVGGT(
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
    )


def _enable_adapter_modules(model) -> None:
    model.event_encoder.requires_grad_(True)
    model.depth_head.geometry_adapters.requires_grad_(True)
    model.point_head.geometry_adapters.requires_grad_(True)


def _configure_trainable(model, cfg) -> None:
    model.requires_grad_(False)
    _enable_adapter_modules(model)
    phase = str(cfg.train.adapter_phase).upper()
    if phase == "A":
        pass
    elif phase == "B":
        model.depth_head.requires_grad_(True)
        model.point_head.requires_grad_(True)
        last_blocks = max(int(cfg.train.adapter_unfreeze_last_blocks), 0)
        if last_blocks > 0:
            for block in model.aggregator.frame_blocks[-last_blocks:]:
                block.requires_grad_(True)
            for block in model.aggregator.global_blocks[-last_blocks:]:
                block.requires_grad_(True)
        if bool(cfg.train.adapter_train_contribution):
            model.contribution_net.requires_grad_(True)
    else:
        raise ValueError(f"Unknown adapter phase {phase!r}; expected A or B")

    # Pose must remain RGB-only and frozen in both phases.
    model.camera_head.requires_grad_(False)
    if not bool(cfg.train.adapter_train_contribution) or phase == "A":
        model.contribution_net.requires_grad_(False).eval()


def _optimizer_groups(model, cfg):
    adapter_ids = {
        id(parameter)
        for module in (
            model.event_encoder,
            model.depth_head.geometry_adapters,
            model.point_head.geometry_adapters,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    contribution_ids = {
        id(parameter) for parameter in model.contribution_net.parameters() if parameter.requires_grad
    }
    adapter_parameters = []
    contribution_parameters = []
    vggt_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in adapter_ids:
            adapter_parameters.append(parameter)
        elif id(parameter) in contribution_ids:
            contribution_parameters.append(parameter)
        else:
            vggt_parameters.append(parameter)
    groups = []
    if adapter_parameters:
        groups.append(
            {"params": adapter_parameters, "lr_scale": float(cfg.train.adapter_lr_scale)}
        )
    if contribution_parameters:
        groups.append(
            {
                "params": contribution_parameters,
                "lr_scale": float(cfg.train.adapter_contribution_lr_scale),
            }
        )
    if vggt_parameters:
        groups.append(
            {"params": vggt_parameters, "lr_scale": float(cfg.train.adapter_vggt_lr_scale)}
        )
    if not groups:
        raise RuntimeError("No trainable Stage-2 adapter parameters were enabled.")
    return groups


def _snapshot(outdir):
    destination = Path(outdir) / "code" / "stage2_geometry_adapter"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in (
        "stage2_geometry_adapter/model.py",
        "stage2_geometry_adapter/loss.py",
        "stage2_geometry_adapter/finetune.py",
        "stage2_geometry_adapter/evaluate.py",
        "stage2_geometry_adapter/run_gpu2_7.sh",
        "stage2_geometry_adapter/test_model.py",
        "stage2_geometry_adapter/README.md",
        "paired_token_reliability/contribution_stage1.py",
    ):
        source = ROOT / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return str(destination)


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.model, cfg.train, cfg.loss, cfg.data, cfg.vis):
        OmegaConf.set_struct(branch, False)

    cfg.model.variant = "stage2_geometry_adapter"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.stage1_contribution_checkpoint = _resolve_path(
        getattr(
            cfg.model,
            "stage1_contribution_checkpoint",
            "abl_event_exp/event_contribution_stage1/checkpoint-best.pth",
        )
    )
    cfg.model.adapter_event_hidden_dim = int(
        getattr(cfg.model, "adapter_event_hidden_dim", 48)
    )
    cfg.model.event_hidden_dim = cfg.model.adapter_event_hidden_dim
    cfg.model.adapter_event_pyramid_channels = int(
        getattr(cfg.model, "adapter_event_pyramid_channels", 64)
    )
    cfg.model.adapter_hidden_channels = int(
        getattr(cfg.model, "adapter_hidden_channels", 128)
    )

    cfg.train.adapter_phase = str(getattr(cfg.train, "adapter_phase", "A")).upper()
    cfg.train.adapter_unfreeze_last_blocks = int(
        getattr(cfg.train, "adapter_unfreeze_last_blocks", 2)
    )
    cfg.train.adapter_train_contribution = bool(
        getattr(cfg.train, "adapter_train_contribution", False)
    )
    cfg.train.adapter_lr_scale = float(getattr(cfg.train, "adapter_lr_scale", 1.0))
    cfg.train.adapter_contribution_lr_scale = float(
        getattr(cfg.train, "adapter_contribution_lr_scale", 0.2)
    )
    cfg.train.adapter_vggt_lr_scale = float(
        getattr(cfg.train, "adapter_vggt_lr_scale", 0.1)
    )

    cfg.loss.pose_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = float(getattr(cfg.loss, "adapter_points_weight", 1.0))
    cfg.loss.normal_weight = float(getattr(cfg.loss, "adapter_normal_weight", 0.25))
    cfg.loss.points_loss_type = str(getattr(cfg.loss, "adapter_points_loss_type", "l1"))
    cfg.loss.depth_second_order_weight = 0.0
    cfg.loss.grid_suppress_weight = 0.0
    cfg.loss.adapter_update_weight = float(
        getattr(cfg.loss, "adapter_update_weight", 0.01)
    )
    cfg.loss.adapter_saturation_boost = float(
        getattr(cfg.loss, "adapter_saturation_boost", 1.0)
    )
    cfg.loss.adapter_saturation_threshold = float(
        getattr(cfg.loss, "adapter_saturation_threshold", 0.98)
    )
    cfg.loss.adapter_saturation_normal_weight = float(
        getattr(cfg.loss, "adapter_saturation_normal_weight", cfg.loss.normal_weight)
    )

    cfg.data.train_initial_scene_idx = int(getattr(cfg.data, "train_initial_scene_idx", 0))
    cfg.data.train_scene_count = int(getattr(cfg.data, "train_scene_count", 12))
    cfg.data.train_holdout_frame_count = int(
        getattr(cfg.data, "train_holdout_frame_count", 0)
    )
    cfg.data.train_min_start_id = int(getattr(cfg.data, "train_min_start_id", 2))
    cfg.data.test_initial_scene_idx = int(
        getattr(
            cfg.data,
            "test_initial_scene_idx",
            cfg.data.train_initial_scene_idx + cfg.data.train_scene_count,
        )
    )
    cfg.data.test_scene_count = int(getattr(cfg.data, "test_scene_count", 4))
    cfg.data.heldout_test_frame_count = int(
        getattr(cfg.data, "heldout_test_frame_count", 120)
    )
    cfg.data.return_normal_gt = True

    cfg.eval_every_steps = int(getattr(cfg, "adapter_eval_every_steps", 0))
    cfg.save_every_steps = int(getattr(cfg, "adapter_checkpoint_every_steps", 2000))
    cfg.print_freq = int(getattr(cfg, "adapter_print_freq", 50))
    cfg.log_freq = int(getattr(cfg, "adapter_log_freq", 100))
    cfg.vis.save_every_steps = int(getattr(cfg.vis, "adapter_save_every_steps", 2000))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "adapter_test_max_batches", 1))

    if not Path(cfg.model.stage1_contribution_checkpoint).is_file():
        raise FileNotFoundError(
            f"Stage-1 contribution checkpoint missing: {cfg.model.stage1_contribution_checkpoint}"
        )
    stage1 = torch_load(cfg.model.stage1_contribution_checkpoint)
    if stage1.get("schema") != MultiLdrEventContributionModel.checkpoint_schema:
        raise ValueError(
            "Stage 2 requires the new multi_ldr_event_contribution_v1 checkpoint; "
            "legacy ReliabilityUNet checkpoints are not accepted."
        )

    current_pretrained = str(getattr(cfg, "pretrained", "") or "")
    if current_pretrained in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        cfg.pretrained = _resolve_path("ckpt/model.pt")
    phase = cfg.train.adapter_phase
    default_name = f"geometry_adapter_stage2_{phase.lower()}"
    if str(getattr(cfg, "exp_name", "")) == "event_finetune_LDR5":
        cfg.exp_name = default_name
    output_root = Path(
        str(getattr(cfg, "adapter_output_root", ROOT / "abl_event_exp" / "stage2_geometry_adapter"))
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg.save_dir = str(output_root.resolve())
    cfg.output_dir = str((output_root / str(cfg.exp_name)).resolve())
    cfg.logdir = str(Path(cfg.output_dir) / "logs")
    return cfg


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="finetune_event.yaml")
def run(cfg):
    cfg = _prepare_cfg(cfg)
    fe.build_event_loader = _build_scene_disjoint_loader
    fe.build_event_model = _build_model
    fe.configure_trainable_params = _configure_trainable
    fe.build_optimizer_params = _optimizer_groups
    fe.save_current_code = _snapshot
    fe.EventSupervisedLoss = make_geometry_adapter_loss(cfg)
    print(
        "[Stage 2 geometry adapter] "
        f"phase={cfg.train.adapter_phase} Stage1={cfg.model.stage1_contribution_checkpoint} "
        f"output={cfg.output_dir}; no depth residual; pose=RGB-only",
        flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
