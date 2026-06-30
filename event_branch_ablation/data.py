"""Dedicated additive-event loaders for the two event-branch ablations."""

from __future__ import annotations

import os.path as osp
from typing import Dict, Iterable

import numpy as np
from torch.utils.data import DataLoader, Dataset

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from fine_event.finetune_event_random_ldr import RandomLdrBatchSampler


TARGET_BRANCHES = {
    "geometry_motion": "event_geometry_voxel",
    "material_reflection": "event_material_voxel",
    "noise": "event_noise_voxel",
}


def _branch_path(scene_dir: str, root_name: str, branch: str) -> str:
    return osp.join(scene_dir, root_name, branch, "events.h5")


def _event_meta(dataset, path: str, frame_count: int) -> dict:
    frame_index, time_bounds, time_info = dataset.build_frame_event_index(
        path,
        frame_count,
        return_time_info=True,
    )
    return {
        "path": path,
        "frame_index": frame_index,
        "time_bounds": time_bounds,
        "time_info": time_info,
        "columns": time_info.get("columns"),
    }


def switch_event_source(dataset, *, branch: str, root_name: str = "events_additive") -> None:
    """Use one additive branch as the dataset's normal event input."""
    for scene in dataset.get_active_scenes():
        meta = dataset.active_scene_data[scene]
        path = _branch_path(meta["scene_dir"], root_name, branch)
        if not osp.isfile(path):
            raise FileNotFoundError(f"Missing additive event branch: scene={scene}, path={path}")
        branch_meta = _event_meta(dataset, path, meta["frame_count"])
        meta["event_h5"] = path
        meta["frame_event_index"] = branch_meta["frame_index"]
        meta["event_time_bounds"] = branch_meta["time_bounds"]
        meta["event_time_info"] = branch_meta["time_info"]
        meta["event_columns"] = branch_meta["columns"]
        info = branch_meta["time_info"]
        if info.get("event_width") and info.get("event_height"):
            meta["event_resolution"] = (int(info["event_width"]), int(info["event_height"]))


class AdditiveBranchTargetDataset(Dataset):
    """Attach branch voxels as labels while retaining full events as input."""

    def __init__(self, dataset, *, root_name: str = "events_additive") -> None:
        self.dataset = dataset
        self.root_name = str(root_name)
        self.branch_meta: Dict[str, Dict[str, dict]] = {}
        self._prepare_branches()

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def _prepare_branches(self) -> None:
        for scene in self.dataset.get_active_scenes():
            scene_meta = self.dataset.active_scene_data[scene]
            per_scene = {}
            for branch in TARGET_BRANCHES:
                path = _branch_path(scene_meta["scene_dir"], self.root_name, branch)
                if not osp.isfile(path):
                    raise FileNotFoundError(
                        f"Missing additive target branch: scene={scene}, branch={branch}, path={path}"
                    )
                per_scene[branch] = _event_meta(self.dataset, path, scene_meta["frame_count"])
            self.branch_meta[scene] = per_scene

    def _load_branch(self, *, scene: str, frame_idx: int, view: dict, branch: str) -> np.ndarray:
        scene_meta = self.dataset.active_scene_data[scene]
        branch_meta = self.branch_meta[scene][branch]
        start, end = branch_meta["frame_index"][frame_idx]
        event_data = self.dataset.load_event_slice(
            branch_meta["path"],
            start,
            end,
            event_columns=branch_meta["columns"],
            time_origin=branch_meta["time_info"].get("origin", 0.0),
        )
        info = branch_meta["time_info"]
        fallback = scene_meta.get("event_resolution", view["event_source_resolution"])
        src_resolution = (
            int(info.get("event_width") or fallback[0]),
            int(info.get("event_height") or fallback[1]),
        )
        dst_values = np.asarray(view["event_resolution"]).reshape(-1)
        dst_resolution = (int(dst_values[0]), int(dst_values[1]))
        resized = self.dataset._resize_event_data(
            event_data,
            src_resolution=src_resolution,
            dst_resolution=dst_resolution,
            spatial_transform=str(view["event_spatial_transform"]),
            resize_method=self.dataset.event_resize_method,
            resize_bins=self.dataset.event_resize_bins,
        )
        voxel = resized["event_voxel"].astype(np.float32, copy=False)
        if "mask" in view:
            voxel = voxel * np.asarray(view["mask"], dtype=np.float32)[None]
        return voxel

    def __getitem__(self, index):
        views = self.dataset[index]
        for view in views:
            scene, frame_text = str(view["instance"]).rsplit("_", 1)
            frame_idx = int(frame_text)
            for branch, target_key in TARGET_BRANCHES.items():
                view[target_key] = self._load_branch(
                    scene=scene,
                    frame_idx=frame_idx,
                    view=view,
                    branch=branch,
                )
        return views


def _build_base_dataset(cfg, split: str, ldr_event_id: str):
    return get_combined_dataset(
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
        return_normal_gt=True,
    )


def _make_loader(cfg, dataset, *, split: str):
    random_ldr = split == "train" and bool(getattr(cfg.data, "random_train_ldr", True))
    if random_ldr:
        exposures: Iterable[str] = dataset.get_active_ldr_events(common=True)
        exposures = list(exposures)
        if not exposures:
            raise ValueError("No common LDR exposure exists across the selected scenes.")
        sampler = RandomLdrBatchSampler(
            dataset,
            batch_size=cfg.batch_size,
            ldr_event_ids=exposures,
            num_views=cfg.data.num_views,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=event_multiview_collate,
        )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )


def _ldr_for_split(cfg, split: str) -> str:
    if split == "train" and bool(getattr(cfg.data, "random_train_ldr", True)):
        return "random"
    return str(getattr(cfg.data, "eval_ldr_event_id", "ev_5"))


def build_geometry_motion_loader(cfg, split: str = "train"):
    dataset = _build_base_dataset(cfg, split, _ldr_for_split(cfg, split))
    switch_event_source(
        dataset,
        branch="geometry_motion",
        root_name=str(getattr(cfg.data, "additive_event_root", "events_additive")),
    )
    fe.printer.info(
        "Geometry-motion reliability loader split=%s scenes=%s samples=%d",
        split, dataset.get_active_scenes(), len(dataset),
    )
    return _make_loader(cfg, dataset, split=split)


def build_full_decomposition_loader(cfg, split: str = "train"):
    dataset = _build_base_dataset(cfg, split, _ldr_for_split(cfg, split))
    root_name = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    switch_event_source(dataset, branch="full", root_name=root_name)
    # Branch files are labels only. Held-out inference intentionally receives
    # full events and RGB without loading branch targets.
    if split == "train":
        dataset = AdditiveBranchTargetDataset(dataset, root_name=root_name)
    fe.printer.info(
        "Full-to-branch loader split=%s scenes=%s samples=%d branch_targets=%s",
        split, dataset.get_active_scenes(), len(dataset), split == "train",
    )
    return _make_loader(cfg, dataset, split=split)

