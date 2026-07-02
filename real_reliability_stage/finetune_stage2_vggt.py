"""Stage 2: finetune VGGT with a frozen pretrained event ReliabilityNet.

Default split:
  * train: scene indices [0, 11], all frames;
  * held-out prediction/test: scene indices [12, 15], all frames.

The frozen ReliabilityNet filters event voxels before temporal-detail
refinement. Event-aware supervision is separately weighted by R.detach().
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from eventvggt.models.streamvggt_pretrained_reliability_detail import StreamVGGT  # noqa: E402
from mul_loss_fine.launcher import configure_mul_loss_cfg  # noqa: E402
from real_reliability_stage.stage2_loss import make_stage2_reliability_weighted_loss  # noqa: E402


_BASE_BUILD_LOADER = fe.build_event_loader
_SPLIT_SCENES = {}


def _resolve_path(value: str) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path.resolve())


def _build_scene_disjoint_loader(cfg, split="train"):
    local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(local_cfg, False)
    OmegaConf.set_struct(local_cfg.data, False)
    local_cfg.data.scene_names = None
    if split == "train":
        local_cfg.data.initial_scene_idx = int(cfg.data.train_initial_scene_idx)
        local_cfg.data.active_scene_count = int(cfg.data.train_scene_count)
        local_cfg.data.test_frame_count = int(cfg.data.train_holdout_frame_count)
    else:
        local_cfg.data.initial_scene_idx = int(cfg.data.test_initial_scene_idx)
        local_cfg.data.active_scene_count = int(cfg.data.test_scene_count)
        # Setting this to the full sequence length makes split=test evaluate
        # every valid window of each entirely unseen scene.
        local_cfg.data.test_frame_count = int(cfg.data.heldout_test_frame_count)

    loader = _BASE_BUILD_LOADER(local_cfg, split=split)
    scenes = tuple(loader.dataset.get_active_scenes())
    _SPLIT_SCENES[split] = scenes
    print(f"[stage2 data] split={split} scenes={list(scenes)} samples={len(loader.dataset)}", flush=True)
    if "train" in _SPLIT_SCENES and "test" in _SPLIT_SCENES:
        overlap = sorted(set(_SPLIT_SCENES["train"]) & set(_SPLIT_SCENES["test"]))
        if overlap:
            raise RuntimeError(
                f"Scene-disjoint split failed; train/test overlap: {overlap}. "
                "Check train/test scene indices and available scene count."
            )
    return loader


def _build_stage2_model(cfg):
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
        refine_points=bool(getattr(cfg.model, "refiner_refine_points", True)),
        use_checkpoint=bool(getattr(cfg.model, "refiner_use_checkpoint", True)),
        reliability_checkpoint=str(cfg.model.reliability_checkpoint),
        reliability_base_channels=int(cfg.model.reliability_base_channels),
        reliability_gate_floor=float(cfg.model.reliability_gate_floor),
        reliability_frame_chunk_size=int(cfg.model.reliability_frame_chunk_size),
        reliability_rgb_input_range=str(cfg.model.reliability_rgb_input_range),
    )


def _configure_stage2_trainable_params(model, cfg) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if "event_detail_refiner.base_refiner" in name and ".reliability_head." not in name:
            parameter.requires_grad = True

    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    # Guard against accidental unfreezing through broad name matching.
    model.event_detail_refiner.reliability_net.requires_grad_(False)
    model.event_detail_refiner.reliability_net.eval()


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.model, cfg.train, cfg.loss, cfg.data, cfg.vis):
        OmegaConf.set_struct(branch, False)

    cfg.model.variant = "pretrained_reliability_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "stage2_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "stage2_residual_scale", 0.035))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "stage2_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = bool(getattr(cfg.model, "stage2_patch_zero_mean", True))
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "stage2_patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "stage2_abs_limit", 0.025))
    cfg.model.reliability_checkpoint = _resolve_path(
        getattr(
            cfg.model,
            "reliability_checkpoint",
            "abl_event_exp/real_reliability_stage/reliability_net/checkpoint-best.pth",
        )
    )
    cfg.model.reliability_base_channels = int(getattr(cfg.model, "reliability_base_channels", 32))
    cfg.model.reliability_gate_floor = float(getattr(cfg.model, "reliability_gate_floor", 0.20))
    cfg.model.reliability_frame_chunk_size = int(getattr(cfg.model, "reliability_frame_chunk_size", 1))
    cfg.model.reliability_rgb_input_range = str(
        getattr(cfg.model, "reliability_rgb_input_range", "minus_one_one")
    )

    cfg.data.train_initial_scene_idx = int(getattr(cfg.data, "train_initial_scene_idx", 0))
    cfg.data.train_scene_count = int(getattr(cfg.data, "train_scene_count", 12))
    cfg.data.train_holdout_frame_count = int(getattr(cfg.data, "train_holdout_frame_count", 0))
    cfg.data.test_initial_scene_idx = int(
        getattr(cfg.data, "test_initial_scene_idx", cfg.data.train_initial_scene_idx + cfg.data.train_scene_count)
    )
    cfg.data.test_scene_count = int(getattr(cfg.data, "test_scene_count", 4))
    cfg.data.heldout_test_frame_count = int(getattr(cfg.data, "heldout_test_frame_count", 120))
    cfg.data.num_views = int(getattr(cfg.data, "num_views", 4))
    cfg.data.return_normal_gt = True

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.loss.pose_weight = 0.0
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = 1.0
    cfg.epochs = max(int(getattr(cfg, "epochs", 10)), 20)
    if float(getattr(cfg, "lr", 1.0e-4)) > 4.0e-5:
        cfg.lr = 4.0e-5
    cfg.eval_every_steps = int(getattr(cfg, "eval_every_steps", 500))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "test_max_batches", 4))
    cfg.vis.test_num_views = int(getattr(cfg.vis, "test_num_views", cfg.data.num_views))

    current_pretrained = str(getattr(cfg, "pretrained", "") or "")
    if current_pretrained in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        cfg.pretrained = _resolve_path("ckpt/model.pt")

    if str(getattr(cfg, "exp_name", "")) == "event_finetune_LDR5":
        cfg.exp_name = "stage2_frozen_real_reliability_train12_test4"
    output_root = ROOT_DIR / "abl_event_exp"
    cfg.save_dir = str(output_root)
    cfg.output_dir = str(output_root / str(cfg.exp_name))
    cfg.logdir = str(Path(cfg.output_dir) / "logs")
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
        "residual_smooth_weight": 0.02,
        "residual_second_order_weight": 0.02,
        "residual_abs_weight": 0.01,
        "residual_smooth_alpha": 10.0,
        "final_grid_weight": 0.02,
        "final_phase_weight": 0.01,
        "final_grid_patch_size": 14,
        "final_grid_band": 1,
        "final_grid_detail_threshold": 0.02,
    }
    cfg = configure_mul_loss_cfg(
        _prepare_cfg(cfg),
        weights=weights,
        exp_name="stage2_frozen_real_reliability_train12_test4",
    )

    reliability_checkpoint = Path(str(cfg.model.reliability_checkpoint))
    if not reliability_checkpoint.is_file():
        raise FileNotFoundError(
            f"Stage-1 ReliabilityNet checkpoint is missing: {reliability_checkpoint}"
        )

    fe.build_event_loader = _build_scene_disjoint_loader
    fe.build_event_model = _build_stage2_model
    fe.configure_trainable_params = _configure_stage2_trainable_params
    fe.EventSupervisedLoss = make_stage2_reliability_weighted_loss(cfg)
    print(
        "Stage-2 frozen reliability VGGT finetune: "
        f"train_scenes={cfg.data.train_scene_count}, test_scenes={cfg.data.test_scene_count}, "
        f"test_start={cfg.data.test_initial_scene_idx}, gate_floor={cfg.model.reliability_gate_floor}, "
        f"reliability={cfg.model.reliability_checkpoint}, output={cfg.output_dir}",
        flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
