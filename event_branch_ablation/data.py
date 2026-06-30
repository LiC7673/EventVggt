"""Dedicated additive-event loaders for the two event-branch ablations."""

from __future__ import annotations

import os.path as osp
from typing import Dict, Iterable

import cv2
import h5py
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


def _common_frame_index(path: str, *, columns: dict, origin: float, dt: float, frame_count: int):
    """Index one branch using the full stream's absolute frame boundaries."""
    boundaries = float(origin) + np.arange(frame_count, dtype=np.float64) * float(dt)
    boundary_indices = np.empty(frame_count, dtype=np.int64)
    boundary_pos = 0
    global_offset = 0
    with h5py.File(path, "r") as handle:
        events = handle["events"]
        total = len(events)
        while global_offset < total and boundary_pos < frame_count:
            chunk_end = min(global_offset + 1_000_000, total)
            timestamps = events[global_offset:chunk_end, int(columns["t"])].astype(np.float64)
            if timestamps.size == 0:
                break
            while boundary_pos < frame_count and boundaries[boundary_pos] <= timestamps[-1]:
                local = np.searchsorted(timestamps, boundaries[boundary_pos], side="left")
                boundary_indices[boundary_pos] = global_offset + local
                boundary_pos += 1
            global_offset = chunk_end
        if boundary_pos < frame_count:
            boundary_indices[boundary_pos:] = total
    frame_index = np.zeros((frame_count, 2), dtype=np.int64)
    for frame_idx in range(1, frame_count):
        frame_index[frame_idx] = (boundary_indices[frame_idx - 1], boundary_indices[frame_idx])
    return frame_index


def _fixed_window_voxel(
    event_data,
    *,
    src_resolution,
    dst_resolution,
    spatial_transform: str,
    num_bins: int,
    t0: float,
    t1: float,
):
    """Voxelize all branches with identical time bins and linear resizing."""
    src_w, src_h = int(src_resolution[0]), int(src_resolution[1])
    dst_w, dst_h = int(dst_resolution[0]), int(dst_resolution[1])
    voxel = np.zeros((2 * num_bins, src_h, src_w), dtype=np.float32)
    xy = event_data["event_xy"].astype(np.float32, copy=True)
    times = event_data["event_t"].astype(np.float64, copy=False)
    polarity = event_data["event_p"].astype(np.float32, copy=False)
    if xy.size:
        if spatial_transform == "hflip":
            xy[:, 0] = (src_w - 1) - xy[:, 0]
        elif spatial_transform == "vflip":
            xy[:, 1] = (src_h - 1) - xy[:, 1]
        elif spatial_transform == "rot180":
            xy[:, 0] = (src_w - 1) - xy[:, 0]
            xy[:, 1] = (src_h - 1) - xy[:, 1]
        elif spatial_transform not in {"none", "None", ""}:
            raise ValueError(f"Unsupported event spatial transform: {spatial_transform}")
        x = np.floor(xy[:, 0]).astype(np.int64)
        y = np.floor(xy[:, 1]).astype(np.int64)
        valid = (
            (x >= 0) & (x < src_w) & (y >= 0) & (y < src_h)
            & np.isfinite(times) & np.isfinite(polarity) & (np.abs(polarity) > 0)
            & (times >= t0) & (times < t1)
        )
        if np.any(valid):
            denom = max(float(t1 - t0), 1e-12)
            bin_idx = np.floor((times[valid] - t0) / denom * num_bins).astype(np.int64)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
            channel = bin_idx + np.where(polarity[valid] > 0, 0, num_bins)
            flat = channel * (src_h * src_w) + y[valid] * src_w + x[valid]
            np.add.at(voxel.reshape(-1), flat, np.abs(polarity[valid]).astype(np.float32))
    interpolation = cv2.INTER_AREA if dst_w <= src_w and dst_h <= src_h else cv2.INTER_LINEAR
    area_scale = float(src_w * src_h) / max(float(dst_w * dst_h), 1.0)
    resized = [
        cv2.resize(channel, (dst_w, dst_h), interpolation=interpolation).astype(np.float32) * area_scale
        for channel in voxel
    ]
    return np.stack(resized, axis=0)


class FixedWindowAdditiveDataset(Dataset):
    """Rebuild full and branch voxels on one shared full-stream time grid."""

    def __init__(
        self,
        dataset,
        *,
        primary_branch: str,
        attach_targets: bool,
        root_name: str = "events_additive",
        mask_dilate_kernel: int = 5,
    ) -> None:
        self.dataset = dataset
        self.primary_branch = str(primary_branch)
        self.attach_targets = bool(attach_targets)
        self.root_name = str(root_name)
        self.mask_dilate_kernel = max(1, int(mask_dilate_kernel))
        self.branch_meta = {}
        self._prepare()

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def _prepare(self):
        required = {self.primary_branch, "full"}
        if self.attach_targets:
            required.update(TARGET_BRANCHES)
        for scene in self.dataset.get_active_scenes():
            scene_meta = self.dataset.active_scene_data[scene]
            full_path = _branch_path(scene_meta["scene_dir"], self.root_name, "full")
            full_meta = _event_meta(self.dataset, full_path, scene_meta["frame_count"])
            reference = full_meta["time_info"]
            per_scene = {}
            for branch in required:
                path = _branch_path(scene_meta["scene_dir"], self.root_name, branch)
                if not osp.isfile(path):
                    raise FileNotFoundError(f"Missing additive branch: {path}")
                own_meta = _event_meta(self.dataset, path, scene_meta["frame_count"])
                own_meta["frame_index"] = _common_frame_index(
                    path,
                    columns=own_meta["columns"],
                    origin=float(reference["origin"]),
                    dt=float(reference["dt"]),
                    frame_count=scene_meta["frame_count"],
                )
                own_meta["common_origin"] = float(reference["origin"])
                own_meta["common_dt"] = float(reference["dt"])
                per_scene[branch] = own_meta
            self.branch_meta[scene] = per_scene

    def _load(self, scene: str, frame_idx: int, view: dict, branch: str):
        meta = self.branch_meta[scene][branch]
        start, end = meta["frame_index"][frame_idx]
        data = self.dataset.load_event_slice(
            meta["path"],
            start,
            end,
            event_columns=meta["columns"],
            time_origin=meta["common_origin"],
        )
        info = meta["time_info"]
        fallback = np.asarray(view["event_source_resolution"]).reshape(-1)
        src = (
            int(info.get("event_width") or fallback[0]),
            int(info.get("event_height") or fallback[1]),
        )
        dst_raw = np.asarray(view["event_resolution"]).reshape(-1)
        dst = (int(dst_raw[0]), int(dst_raw[1]))
        dt = meta["common_dt"]
        t0 = max(frame_idx - 1, 0) * dt
        t1 = frame_idx * dt
        if frame_idx == 0:
            voxel = np.zeros((2 * self.dataset.event_resize_bins, dst[1], dst[0]), dtype=np.float32)
        else:
            voxel = _fixed_window_voxel(
                data,
                src_resolution=src,
                dst_resolution=dst,
                spatial_transform=str(view["event_spatial_transform"]),
                num_bins=int(self.dataset.event_resize_bins),
                t0=t0,
                t1=t1,
            )
        if "mask" in view:
            mask = np.asarray(view["mask"], dtype=np.uint8)
            if self.mask_dilate_kernel > 1:
                kernel = np.ones((self.mask_dilate_kernel, self.mask_dilate_kernel), dtype=np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            voxel = voxel * mask.astype(np.float32)[None]
        return voxel

    def __getitem__(self, index):
        views = self.dataset[index]
        for view in views:
            scene, frame_text = str(view["instance"]).rsplit("_", 1)
            frame_idx = int(frame_text)
            view["event_voxel"] = self._load(scene, frame_idx, view, self.primary_branch)
            if self.attach_targets:
                for branch, key in TARGET_BRANCHES.items():
                    view[key] = self._load(scene, frame_idx, view, branch)
        return views


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
    root_name = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    switch_event_source(
        dataset,
        branch="geometry_motion",
        root_name=root_name,
    )
    dataset = FixedWindowAdditiveDataset(
        dataset,
        primary_branch="geometry_motion",
        attach_targets=False,
        root_name=root_name,
        mask_dilate_kernel=int(getattr(cfg.data, "additive_mask_dilate_kernel", 5)),
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
    dataset = FixedWindowAdditiveDataset(
        dataset,
        primary_branch="full",
        attach_targets=(split == "train"),
        root_name=root_name,
        mask_dilate_kernel=int(getattr(cfg.data, "additive_mask_dilate_kernel", 5)),
    )
    fe.printer.info(
        "Full-to-branch loader split=%s scenes=%s samples=%d branch_targets=%s",
        split, dataset.get_active_scenes(), len(dataset), split == "train",
    )
    return _make_loader(cfg, dataset, split=split)
