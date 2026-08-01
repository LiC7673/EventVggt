
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameRecord:
    sequence: Path
    frame_id: int
    image_path: Path
    depth_path: Path
    timestamp: float


def linear_polarity_voxel(x, y, timestamp, polarity, height, width,
                          bin_count, start_time, end_time):
                                                                             
    voxel = torch.zeros(2 * bin_count, height, width, dtype=torch.float32)
    if len(timestamp) == 0 or end_time <= start_time:
        return voxel
    x = torch.as_tensor(x, dtype=torch.long)
    y = torch.as_tensor(y, dtype=torch.long)
    t = torch.as_tensor(timestamp, dtype=torch.float32)
    p = torch.as_tensor(polarity)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    normalized = (t - start_time) / max(end_time - start_time, 1.0e-9)
    coordinate = normalized.clamp(0, 1) * (bin_count - 1)
    left = coordinate.floor().long()
    right = (left + 1).clamp_max(bin_count - 1)
    right_weight = coordinate - left.float()
    left_weight = 1.0 - right_weight
    polarity_offset = torch.where(p > 0, 0, bin_count)
    flat = voxel.view(2 * bin_count, -1)
    pixel = y * width + x
    flat.index_put_((left + polarity_offset, pixel), left_weight, accumulate=True)
    flat.index_put_((right + polarity_offset, pixel), right_weight, accumulate=True)
    return voxel


class EventWindowReader:
                                                                               

    def __init__(self, event_file):
        self.event_file = Path(event_file)

    def read(self, start_time, end_time):
        with h5py.File(self.event_file, "r") as handle:
            timestamps = np.asarray(handle["t"])
            begin = int(np.searchsorted(timestamps, start_time, side="right"))
            finish = int(np.searchsorted(timestamps, end_time, side="right"))
            return {
                "x": np.asarray(handle["x"][begin:finish]),
                "y": np.asarray(handle["y"][begin:finish]),
                "t": timestamps[begin:finish],
                "p": np.asarray(handle["polarity"][begin:finish]),
            }


class MultiViewEventGeometryDataset(Dataset):
                                                                                  

    def __init__(self, root, num_views=4, bin_count=5, image_size=None):
        self.root = Path(root)
        self.num_views = int(num_views)
        self.bin_count = int(bin_count)
        self.image_size = image_size
        self.sequences = self._discover_sequences()
        self.windows = self._build_adjacent_windows()

    def _discover_sequences(self):
        sequences = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            calibration = directory / "calibration.json"
            event_file = directory / "events.h5"
            if calibration.is_file() and event_file.is_file():
                sequences.append(directory)
        if not sequences:
            raise RuntimeError("no sequences match the documented public layout")
        return sequences

    def _records(self, sequence):
        metadata = json.loads((sequence / "calibration.json").read_text())
        records = []
        for item in metadata["frames"]:
            frame_id = int(item["id"])
            image = sequence / "rgb" / f"{frame_id:06d}.png"
            depth = sequence / "depth" / f"{frame_id:06d}.npy"
            if image.is_file() and depth.is_file():
                records.append(FrameRecord(sequence, frame_id, image, depth,
                                           float(item["timestamp"])))
        return records

    def _build_adjacent_windows(self):
        windows = []
        for sequence in self.sequences:
            records = self._records(sequence)
            for start in range(1, len(records) - self.num_views + 1):
                selected = records[start:start + self.num_views]
                if all(b.frame_id == a.frame_id + 1
                       for a, b in zip(selected[:-1], selected[1:])):
                    windows.append((records[start - 1], selected))
        return windows

    @staticmethod
    def _camera_metadata(sequence):
        return json.loads((sequence / "calibration.json").read_text())

    def _load_view(self, previous, current, metadata):
        rgb = np.asarray(Image.open(current.image_path).convert("RGB"))
        rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(np.load(current.depth_path)).float()
        height, width = depth.shape
        events = EventWindowReader(current.sequence / "events.h5").read(
            previous.timestamp, current.timestamp
        )
        voxel = linear_polarity_voxel(
            events["x"], events["y"], events["t"], events["p"],
            height, width, self.bin_count, previous.timestamp, current.timestamp,
        )
        camera = metadata["camera_by_frame"][str(current.frame_id)]
        return {
            "image": rgb,
            "depth": depth,
            "valid_depth": torch.isfinite(depth) & (depth > 0),
            "event_voxel": voxel,
            "event_time_range": torch.tensor(
                [previous.timestamp, current.timestamp], dtype=torch.float64
            ),
            "intrinsics": torch.tensor(camera["intrinsics"], dtype=torch.float32),
            "camera_to_world": torch.tensor(camera["camera_to_world"],
                                            dtype=torch.float32),
            "frame_id": current.frame_id,
        }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        previous, records = self.windows[index]
        metadata = self._camera_metadata(records[0].sequence)
        views = []
        for record in records:
            views.append(self._load_view(previous, record, metadata))
            previous = record
        return {"views": views}
