"""Strict RGB + full-event baseline.

This entry intentionally contains none of the proposed method components:

* no geometry-event teacher;
* no contribution/reliability map;
* no Multi-LDR/HDR token alignment;
* no event-normal or normal-derivative supervision;
* no pixel depth refiner.

The full event stream is encoded once and added directly to the frozen/pretrained
RGB token stream by the original ``eventvggt.models.streamvggt.StreamVGGT``.
Training uses only the ordinary final pose/depth/point geometry losses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import finetune_event as fe  # noqa: E402
from eventvggt.datasets.my_event_dataset import (  # noqa: E402
    event_multiview_collate,
    get_combined_dataset,
)


def _safe_snapshot(output_dir):
    """Avoid recursively copying an experiment directory into itself."""
    path = Path(output_dir) / "code"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_full_event_loader(cfg, split="train"):
    data = cfg.data
    if split == "train":
        initial_scene_idx = int(getattr(data, "train_initial_scene_idx", 0))
        active_scene_count = int(getattr(data, "train_scene_count", 12))
        test_frame_count = int(getattr(data, "train_holdout_frame_count", 0))
    else:
        initial_scene_idx = int(getattr(data, "test_initial_scene_idx", 12))
        active_scene_count = int(getattr(data, "test_scene_count", 4))
        test_frame_count = int(getattr(data, "heldout_test_frame_count", 120))

    dataset = get_combined_dataset(
        root=str(data.root),
        num_views=int(data.num_views),
        resolution=tuple(data.resolution),
        fps=int(data.fps),
        seed=int(cfg.seed),
        scene_names=data.scene_names if getattr(data, "scene_names", None) else None,
        initial_scene_idx=initial_scene_idx,
        active_scene_count=active_scene_count,
        split=split,
        test_frame_count=test_frame_count,
        min_train_start_id=int(getattr(data, "min_train_start_id", 1)),
        ldr_event_id=str(data.ldr_event_id),
        event_y_flip=getattr(data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(data, "event_spatial_transform", "auto"),
        event_resize_method=str(getattr(data, "event_resize_method", "voxel_linear_time")),
        event_resize_bins=int(getattr(data, "event_resize_bins", 5)),
        event_voxel_cache_size=int(getattr(data, "event_voxel_cache_size", 0)),
        event_source_mode="decomposition_full",
        decomposition_supervision=False,
        decomposition_event_root="events_additive",
        decomposition_geo_branch="geometry_motion",
        decomposition_full_branch="full",
        return_normal_gt=False,
        return_debug_event_fields=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples for split={split}, scenes={dataset.get_active_scenes()}, "
            f"root={data.root}"
        )
    print(
        f"[direct-full] split={split} samples={len(dataset)} "
        f"scenes={dataset.get_active_scenes()} source=events_additive/full",
        flush=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=(split == "train"),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_mem),
        drop_last=(split == "train"),
        collate_fn=event_multiview_collate,
    )


def configure_direct_trainable(model, cfg):
    """Train the direct fusion path; do not leave its patch projection frozen."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    direct_event_prefixes = ("event_encoder.", "event_patch_embed.")
    for name, parameter in model.named_parameters():
        if name.startswith(direct_event_prefixes):
            parameter.requires_grad = True

    if bool(cfg.train.unfreeze_heads):
        for module in (
            model.camera_head,
            model.depth_head,
            model.point_head,
            model.track_head,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    if bool(cfg.train.unfreeze_aggregator_blocks):
        for parameter in model.aggregator.frame_blocks.parameters():
            parameter.requires_grad = True
        for parameter in model.aggregator.global_blocks.parameters():
            parameter.requires_grad = True


def _prepare_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.data, cfg.model, cfg.loss, cfg.train):
        OmegaConf.set_struct(branch, False)

    # The original direct-addition StreamVGGT is the complete model here.
    cfg.model.variant = "base"
    cfg.epochs = 3
    cfg.start_epoch = 0
    cfg.validate_each_epoch = False
    cfg.skip_final_eval = True
    cfg.eval_every_steps = 0

    cfg.data.num_views = int(getattr(cfg.data, "num_views", 4))
    cfg.data.event_resize_method = "voxel_linear_time"
    cfg.data.event_resize_bins = 5
    cfg.data.event_source_mode = "decomposition_full"
    cfg.data.decomposition_supervision = False
    cfg.data.decomposition_event_root = "events_additive"
    cfg.data.decomposition_geo_branch = "geometry_motion"
    cfg.data.decomposition_full_branch = "full"
    cfg.data.train_initial_scene_idx = 0
    cfg.data.train_scene_count = 12
    cfg.data.train_holdout_frame_count = 0
    cfg.data.test_initial_scene_idx = 12
    cfg.data.test_scene_count = 4
    cfg.data.heldout_test_frame_count = 120

    # Only ordinary final geometry supervision.
    cfg.loss.normal_weight = 0.0
    cfg.loss.depth_second_order_weight = 0.0
    cfg.loss.grid_suppress_weight = 0.0
    cfg.loss.align_depth_scale = False
    return cfg


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg):
    cfg = _prepare_cfg(cfg)
    fe.save_current_code = _safe_snapshot
    fe.build_event_loader = build_full_event_loader
    fe.configure_trainable_params = configure_direct_trainable
    print(
        "[DIRECT FULL BASELINE] RGB + E_full -> direct token addition -> "
        "pose/depth/point heads; no C, Multi-LDR, teacher, normal path, or refiner",
        flush=True,
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
