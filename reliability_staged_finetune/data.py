"""Full-event VGGT loader augmented with additive geometry reliability labels."""

from __future__ import annotations

import os.path as osp
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, Dataset

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset


class FullEventReliabilityDataset(Dataset):
    def __init__(self, dataset, *, additive_root: str = "events_additive") -> None:
        self.dataset = dataset
        self.additive_root = str(additive_root)
        self.geometry_meta: Dict[str, dict] = {}
        self._configure_additive_sources()

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def _configure_additive_sources(self) -> None:
        for scene in self.dataset.get_active_scenes():
            scene_meta = self.dataset.active_scene_data[scene]
            scene_dir = scene_meta["scene_dir"]
            full_path = osp.join(scene_dir, self.additive_root, "full", "events.h5")
            geo_path = osp.join(scene_dir, self.additive_root, "geometry_motion", "events.h5")
            if not osp.isfile(full_path) or not osp.isfile(geo_path):
                raise FileNotFoundError(
                    f"Missing additive full/geometry branches for scene={scene}: "
                    f"full={full_path}, geometry={geo_path}"
                )

            full_index, full_bounds, full_info = self.dataset.build_frame_event_index(
                full_path, scene_meta["frame_count"], return_time_info=True
            )
            scene_meta["event_h5"] = full_path
            scene_meta["frame_event_index"] = full_index
            scene_meta["event_time_bounds"] = full_bounds
            scene_meta["event_time_info"] = full_info
            scene_meta["event_columns"] = full_info.get("columns")
            if full_info.get("event_width") and full_info.get("event_height"):
                scene_meta["event_resolution"] = (
                    int(full_info["event_width"]), int(full_info["event_height"])
                )

            geo_index, _, geo_info = self.dataset.build_frame_event_index(
                geo_path, scene_meta["frame_count"], return_time_info=True
            )
            self.geometry_meta[scene] = {
                "path": geo_path,
                "index": geo_index,
                "info": geo_info,
                "columns": geo_info.get("columns"),
            }

    def __getitem__(self, index):
        views = self.dataset[index]
        for view in views:
            scene, frame_text = str(view["instance"]).rsplit("_", 1)
            frame_idx = int(frame_text)
            scene_meta = self.dataset.active_scene_data[scene]
            geo_meta = self.geometry_meta[scene]
            start, end = geo_meta["index"][frame_idx]
            event_data = self.dataset.load_event_slice(
                geo_meta["path"],
                start,
                end,
                event_columns=geo_meta["columns"],
                time_origin=geo_meta["info"].get("origin", 0.0),
            )
            src_resolution = (
                int(geo_meta["info"].get("event_width") or scene_meta["event_resolution"][0]),
                int(geo_meta["info"].get("event_height") or scene_meta["event_resolution"][1]),
            )
            dst_resolution = tuple(int(v) for v in np.asarray(view["event_resolution"]).reshape(-1)[:2])
            event_data = self.dataset._resize_event_data(
                event_data,
                src_resolution=src_resolution,
                dst_resolution=dst_resolution,
                spatial_transform=view["event_spatial_transform"],
                resize_method=self.dataset.event_resize_method,
                resize_bins=self.dataset.event_resize_bins,
            )
            geometry = event_data["event_voxel"].astype(np.float32, copy=False)
            if "mask" in view:
                geometry = geometry * np.asarray(view["mask"], dtype=np.float32)[None]
            full = np.asarray(view["event_voxel"], dtype=np.float32)
            geo_energy = geometry.sum(axis=0)
            full_energy = full.sum(axis=0)
            view["event_geometry_voxel"] = geometry
            view["event_reliability_gt"] = np.clip(
                geo_energy / (full_energy + 1.0e-6), 0.0, 1.0
            ).astype(np.float32)
            view["event_full_presence"] = (full_energy > 0).astype(np.float32)
        return views


def build_staged_reliability_loader(cfg, split: str = "train"):
    base = get_combined_dataset(
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
        ldr_event_id=getattr(cfg.data, "ldr_event_id", "ev_5"),
        event_resize_method=getattr(cfg.data, "event_resize_method", "voxel_antialias"),
        event_resize_bins=getattr(cfg.data, "event_resize_bins", 10),
        return_normal_gt=True,
    )
    dataset = FullEventReliabilityDataset(
        base, additive_root=getattr(cfg.data, "additive_event_root", "events_additive")
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=(split == "train"),
        collate_fn=event_multiview_collate,
    )
    fe.printer.info(
        "Staged reliability loader split=%s scenes=%s samples=%d",
        split, base.get_active_scenes(), len(base),
    )
    return loader

