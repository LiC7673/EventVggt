"""Dataset for supervised geometry-event reliability pretraining.

Expected scene layout::

    scene_xxx/
      LDR/ev_5/*.png
      events_additive/
        geometry_motion/events.h5
        material_reflection/events.h5
        noise/events.h5
        full/events.h5

The target is a soft geometry reliability map:

    R_geo = abs(V_geometry) / (abs(V_full) + eps)

All branches are voxelized using identical temporal bins, polarity handling,
spatial transform, and resolution.
"""

from __future__ import annotations

import bisect
import json
import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EVENT_BRANCHES = ("geometry_motion", "material_reflection", "noise", "full")


def _list_images(folder: str) -> List[str]:
    suffixes = {".png", ".jpg", ".jpeg"}
    if not osp.isdir(folder):
        return []

    def key(path: str):
        stem = osp.splitext(osp.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(
        [
            osp.join(folder, name)
            for name in os.listdir(folder)
            if osp.splitext(name)[1].lower() in suffixes
        ],
        key=key,
    )


def _format_ldr(ldr_event_id: str) -> str:
    value = str(ldr_event_id)
    return value if value.startswith("ev_") else f"ev_{value}"


def _read_event_columns(h5: h5py.File) -> Dict[str, int]:
    raw = h5.attrs.get("event_columns", None)
    if raw is None and "events" in h5:
        raw = h5["events"].attrs.get("event_columns", None)
    if raw is None:
        return {"t": 0, "x": 1, "y": 2, "p": 3}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return {str(k): int(v) for k, v in parsed.items()}
        except Exception:
            pass
    return {"t": 0, "x": 1, "y": 2, "p": 3}


def _read_resolution(h5: h5py.File, events: h5py.Dataset) -> Optional[Tuple[int, int]]:
    attrs = {}
    attrs.update(dict(h5.attrs))
    attrs.update(dict(events.attrs))
    for w_key, h_key in (
        ("width", "height"),
        ("event_width", "event_height"),
        ("w", "h"),
    ):
        if w_key in attrs and h_key in attrs:
            return int(attrs[w_key]), int(attrs[h_key])
    if "resolution" in attrs:
        value = np.asarray(attrs["resolution"]).reshape(-1)
        if value.size >= 2:
            return int(value[0]), int(value[1])
    return None


@dataclass
class EventBranchMeta:
    path: str
    event_count: int
    columns: Dict[str, int]
    times: np.ndarray
    resolution: Optional[Tuple[int, int]]


@dataclass
class ReliabilitySceneMeta:
    scene: str
    scene_dir: str
    image_paths: List[str]
    branches: Dict[str, EventBranchMeta]
    src_resolution: Tuple[int, int]
    dst_resolution: Tuple[int, int]
    frame_count: int


class AdditiveEventReliabilityDataset(Dataset):
    def __init__(
        self,
        root: str,
        *,
        ldr_event_id: str = "ev_5",
        resolution: Tuple[int, int] = (518, 392),
        num_bins: int = 5,
        split: str = "train",
        initial_scene_idx: int = 0,
        active_scene_count: int = 12,
        test_scene_count: int = 6,
        scene_names: Optional[Sequence[str]] = None,
        event_root_name: str = "events_additive",
        spatial_transform: str = "none",
        eps: float = 1e-6,
    ) -> None:
        self.root = root
        self.ldr_event_id = _format_ldr(ldr_event_id)
        self.resolution = tuple(int(v) for v in resolution)
        self.num_bins = int(num_bins)
        self.split = str(split)
        self.initial_scene_idx = int(initial_scene_idx)
        self.active_scene_count = int(active_scene_count)
        self.test_scene_count = int(test_scene_count)
        self.scene_names = list(scene_names) if scene_names else None
        self.event_root_name = event_root_name
        self.spatial_transform = str(spatial_transform).lower()
        self.eps = float(eps)
        self.scenes = self._discover_scenes()
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No valid additive-event samples found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_sampled_scene_names(self) -> List[str]:
        seen = []
        for scene_idx, _ in self.samples:
            scene = self.scenes[scene_idx].scene
            if scene not in seen:
                seen.append(scene)
        return seen

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        scene_idx, frame_idx = self.samples[index]
        meta = self.scenes[scene_idx]
        rgb = self._load_rgb(meta.image_paths[frame_idx], meta.dst_resolution)
        branch_voxels = {
            branch: self._load_branch_voxel(meta, branch, frame_idx)
            for branch in EVENT_BRANCHES
        }
        full = branch_voxels["full"]
        geometry = branch_voxels["geometry_motion"]
        material = branch_voxels["material_reflection"]
        noise = branch_voxels["noise"]
        full_abs = np.abs(full).sum(axis=0, keepdims=True)
        geo_abs = np.abs(geometry).sum(axis=0, keepdims=True)
        target = np.clip(geo_abs / (full_abs + self.eps), 0.0, 1.0).astype(np.float32)
        event_presence = (full_abs > 0).astype(np.float32)
        additive_error = np.abs(full - (geometry + material + noise)).mean(dtype=np.float64)

        return {
            "rgb": torch.from_numpy(rgb),
            "event_full": torch.from_numpy(full.astype(np.float32, copy=False)),
            "event_geometry": torch.from_numpy(geometry.astype(np.float32, copy=False)),
            "event_material": torch.from_numpy(material.astype(np.float32, copy=False)),
            "event_noise": torch.from_numpy(noise.astype(np.float32, copy=False)),
            "target_reliability": torch.from_numpy(target),
            "event_presence": torch.from_numpy(event_presence),
            "additive_error": torch.tensor(float(additive_error), dtype=torch.float32),
            "scene": meta.scene,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }

    def _discover_scenes(self) -> List[ReliabilitySceneMeta]:
        if self.scene_names is None:
            raw_scenes = sorted(
                name for name in os.listdir(self.root) if osp.isdir(osp.join(self.root, name))
            )
        else:
            raw_scenes = list(self.scene_names)

        discovered = []
        for scene in raw_scenes:
            scene_dir = osp.join(self.root, scene)
            meta = self._probe_scene(scene, scene_dir)
            if meta is not None:
                discovered.append(meta)
        return discovered

    def _probe_scene(self, scene: str, scene_dir: str) -> Optional[ReliabilitySceneMeta]:
        image_paths = _list_images(osp.join(scene_dir, "LDR", self.ldr_event_id))
        if not image_paths:
            return None

        branches = {}
        for branch in EVENT_BRANCHES:
            path = osp.join(scene_dir, self.event_root_name, branch, "events.h5")
            if not osp.isfile(path):
                return None
            branches[branch] = self._load_branch_meta(path)
        src_resolution = next(
            (meta.resolution for meta in branches.values() if meta.resolution is not None),
            None,
        )
        if src_resolution is None:
            with Image.open(image_paths[0]) as image:
                src_resolution = image.size
        frame_count = min(len(image_paths), 120)
        return ReliabilitySceneMeta(
            scene=scene,
            scene_dir=scene_dir,
            image_paths=image_paths[:frame_count],
            branches=branches,
            src_resolution=(int(src_resolution[0]), int(src_resolution[1])),
            dst_resolution=self.resolution,
            frame_count=frame_count,
        )

    def _load_branch_meta(self, path: str) -> EventBranchMeta:
        with h5py.File(path, "r") as h5:
            if "events" not in h5:
                raise ValueError(f"Missing events dataset: {path}")
            events = h5["events"]
            columns = _read_event_columns(h5)
            t_col = columns["t"]
            times = events[:, t_col].astype(np.float64)
            resolution = _read_resolution(h5, events)
            return EventBranchMeta(
                path=path,
                event_count=int(events.shape[0]),
                columns=columns,
                times=times,
                resolution=resolution,
            )

    def _build_samples(self) -> List[Tuple[int, int]]:
        if self.split == "train":
            start = self.initial_scene_idx
            end = min(start + self.active_scene_count, len(self.scenes))
        elif self.split in {"test", "val"}:
            start = min(self.initial_scene_idx + self.active_scene_count, len(self.scenes))
            end = min(start + self.test_scene_count, len(self.scenes))
        else:
            start, end = 0, len(self.scenes)

        samples = []
        for scene_idx in range(start, end):
            frame_count = self.scenes[scene_idx].frame_count
            for frame_idx in range(1, frame_count):
                samples.append((scene_idx, frame_idx))
        return samples

    @staticmethod
    def _load_rgb(path: str, resolution: Tuple[int, int]) -> np.ndarray:
        image = Image.open(path).convert("RGB").resize(resolution, Image.BICUBIC)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return arr * 2.0 - 1.0

    def _load_branch_voxel(self, meta: ReliabilitySceneMeta, branch: str, frame_idx: int) -> np.ndarray:
        branch_meta = meta.branches[branch]
        t0, t1 = self._frame_time_bounds(branch_meta, meta.frame_count, frame_idx)
        start = bisect.bisect_left(branch_meta.times, t0)
        end = bisect.bisect_left(branch_meta.times, t1)
        if end <= start:
            return np.zeros((2 * self.num_bins, meta.dst_resolution[1], meta.dst_resolution[0]), dtype=np.float32)

        with h5py.File(branch_meta.path, "r") as h5:
            events = h5["events"][start:end]
        cols = branch_meta.columns
        event_t = events[:, cols["t"]].astype(np.float64)
        event_x = events[:, cols["x"]].astype(np.float32)
        event_y = events[:, cols["y"]].astype(np.float32)
        event_p = events[:, cols["p"]].astype(np.float32)
        event_p[event_p == 0] = -1.0
        return self._events_to_voxel(
            event_x,
            event_y,
            event_t,
            event_p,
            t0=t0,
            t1=t1,
            src_resolution=meta.src_resolution,
            dst_resolution=meta.dst_resolution,
        )

    @staticmethod
    def _frame_time_bounds(branch_meta: EventBranchMeta, frame_count: int, frame_idx: int) -> Tuple[float, float]:
        if branch_meta.times.size == 0:
            return 0.0, 0.0
        t_min = float(branch_meta.times[0])
        t_max = float(branch_meta.times[-1])
        if t_max <= t_min:
            return t_min, t_min
        # Events are assumed to span the whole 0..frame_count-1 sequence.
        edges = np.linspace(t_min, t_max, frame_count, dtype=np.float64)
        left = max(0, frame_idx - 1)
        right = min(frame_idx, frame_count - 1)
        return float(edges[left]), float(edges[right])

    def _events_to_voxel(
        self,
        event_x: np.ndarray,
        event_y: np.ndarray,
        event_t: np.ndarray,
        event_p: np.ndarray,
        *,
        t0: float,
        t1: float,
        src_resolution: Tuple[int, int],
        dst_resolution: Tuple[int, int],
    ) -> np.ndarray:
        src_w, src_h = src_resolution
        dst_w, dst_h = dst_resolution
        x = event_x.copy()
        y = event_y.copy()
        if self.spatial_transform in {"vflip", "vertical_flip"}:
            y = float(src_h - 1) - y
        elif self.spatial_transform in {"hflip", "horizontal_flip"}:
            x = float(src_w - 1) - x
        elif self.spatial_transform in {"rot180", "rotate180"}:
            x = float(src_w - 1) - x
            y = float(src_h - 1) - y

        x = np.floor(x * float(dst_w) / max(float(src_w), 1.0)).astype(np.int64)
        y = np.floor(y * float(dst_h) / max(float(src_h), 1.0)).astype(np.int64)
        valid = (x >= 0) & (x < dst_w) & (y >= 0) & (y < dst_h)
        if t1 <= t0:
            bin_id = np.zeros_like(x)
        else:
            bin_id = np.floor((event_t - t0) / max(t1 - t0, 1e-12) * self.num_bins).astype(np.int64)
        bin_id = np.clip(bin_id, 0, self.num_bins - 1)
        polarity = (event_p <= 0).astype(np.int64)
        channel = polarity * self.num_bins + bin_id
        flat = channel * (dst_h * dst_w) + y * dst_w + x
        flat = flat[valid]
        weights = np.abs(event_p[valid]).astype(np.float32, copy=False)
        voxel = np.zeros(2 * self.num_bins * dst_h * dst_w, dtype=np.float32)
        if flat.size > 0:
            np.add.at(voxel, flat, weights)
        voxel = voxel.reshape(2 * self.num_bins, dst_h, dst_w)
        return voxel.astype(np.float32, copy=False)


def build_reliability_dataloader(
    *,
    root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    **dataset_kwargs,
):
    dataset = AdditiveEventReliabilityDataset(root=root, split=split, **dataset_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
    return dataset, loader
