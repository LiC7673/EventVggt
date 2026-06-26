"""Load geometry-related diffuse events for oracle reliability ablations."""

from __future__ import annotations

import os.path as osp
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from fine_event.finetune_event_random_ldr import RandomLdrBatchSampler


def _additive_event_path(scene_dir: str, branch: str, root_name: str) -> str:
    return osp.join(scene_dir, root_name, branch, "events.h5")


def _switch_dataset_event_branch(dataset, *, branch: str, root_name: str) -> None:
    """Replace each active scene event source with an additive event branch."""

    for scene_name in dataset.get_active_scenes():
        record = dataset.scene_records[scene_name]
        scene_dir = record["scene_dir"]
        branch_path = _additive_event_path(scene_dir, branch, root_name)
        if not osp.isfile(branch_path):
            raise FileNotFoundError(
                f"Missing additive event branch for scene={scene_name}: {branch_path}"
            )

        record["event_h5"] = branch_path
        meta = dataset.active_scene_data[scene_name]
        meta["event_h5"] = branch_path
        frame_event_index, event_time_bounds, event_time_info = dataset.build_frame_event_index(
            branch_path,
            meta["frame_count"],
            return_time_info=True,
        )
        meta["frame_event_index"] = frame_event_index
        meta["event_time_bounds"] = event_time_bounds
        meta["event_time_info"] = event_time_info
        meta["event_columns"] = event_time_info.get("columns")
        if event_time_info.get("event_width") and event_time_info.get("event_height"):
            meta["event_resolution"] = (
                int(event_time_info["event_width"]),
                int(event_time_info["event_height"]),
            )


def _dilate_event_voxel(event_voxel: torch.Tensor, *, kernel: int) -> torch.Tensor:
    if kernel <= 1 or event_voxel.numel() == 0:
        return event_voxel
    if event_voxel.ndim != 4:
        return event_voxel
    if event_voxel.shape[1] <= 0:
        return event_voxel
    height, width = event_voxel.shape[-2:]
    # The voxel channels are polarity/time separated and non-negative after
    # rasterization, so max-pooling expands support without mixing signs.
    dilated = F.max_pool2d(event_voxel, kernel_size=kernel, stride=1, padding=kernel // 2)
    return dilated[..., :height, :width]


def _collate_with_dilated_event_mask(batch, *, mask_dilate_kernel: int):
    views = event_multiview_collate(batch)
    if mask_dilate_kernel <= 1:
        return views
    for view in views:
        if "event_voxel" in view and torch.is_tensor(view["event_voxel"]):
            view["event_voxel"] = _dilate_event_voxel(
                view["event_voxel"],
                kernel=int(mask_dilate_kernel),
            )
    return views


def _build_dataset(cfg, split: str, ldr_event_id: str):
    dataset = get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=cfg.data.scene_names if cfg.data.scene_names else None,
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=ldr_event_id,
        event_resize_method=getattr(cfg.data, "event_resize_method", "voxel_antialias"),
        event_resize_bins=getattr(cfg.data, "event_resize_bins", 10),
        return_normal_gt=getattr(cfg.data, "return_normal_gt", False),
    )
    branch = str(getattr(cfg.data, "additive_event_branch", "geometry_motion"))
    root_name = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    _switch_dataset_event_branch(dataset, branch=branch, root_name=root_name)
    return dataset


def build_geometry_event_loader(cfg, split: str = "train"):
    random_train_ldr = bool(getattr(cfg.data, "random_train_ldr", True))
    if split == "train" and random_train_ldr:
        ldr_event_id = getattr(cfg.data, "ldr_event_id", "random")
    else:
        ldr_event_id = getattr(cfg.data, "eval_ldr_event_id", "auto")

    dataset = _build_dataset(cfg, split, ldr_event_id)
    if len(dataset) <= 0:
        raise ValueError(
            f"Dataset has no valid samples under {cfg.data.root}. "
            f"num_views={cfg.data.num_views}, active_scenes={dataset.get_active_scenes()}"
        )

    mask_dilate_kernel = int(getattr(cfg.data, "geometry_event_mask_dilate_kernel", 5))
    collate_fn = lambda batch: _collate_with_dilated_event_mask(  # noqa: E731
        batch,
        mask_dilate_kernel=mask_dilate_kernel,
    )

    if split == "train" and random_train_ldr:
        ldr_event_ids: List[str] = dataset.get_active_ldr_events(common=True)
        if not ldr_event_ids:
            raise ValueError("No common LDR exposure was found across active scenes.")
        batch_sampler = RandomLdrBatchSampler(
            dataset,
            batch_size=cfg.batch_size,
            ldr_event_ids=ldr_event_ids,
            num_views=cfg.data.num_views,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=collate_fn,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            drop_last=False,
            collate_fn=collate_fn,
        )

    fe.printer.info(
        "Geometry-event loader split=%s branch=%s scenes=%s samples=%d dilation=%d",
        split,
        getattr(cfg.data, "additive_event_branch", "geometry_motion"),
        dataset.get_active_scenes(),
        len(dataset),
        mask_dilate_kernel,
    )
    return loader
