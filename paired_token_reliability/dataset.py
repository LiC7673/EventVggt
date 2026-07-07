from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from eventvggt.datasets.my_event_dataset import get_combined_dataset
from paired_token_reliability.common import read_json


class PairedReliabilityDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = read_json(self.manifest_path)
        self.records = [record for record in self.manifest["records"] if record["split"] == split]
        self.dataset = get_combined_dataset(
            root=self.manifest["dataset_root"],
            num_views=int(self.manifest["num_views"]),
            resolution=tuple(self.manifest["resolution"]),
            fps=int(self.manifest["fps"]),
            seed=0,
            scene_names=self.manifest["scene_names"],
            initial_scene_idx=0,
            active_scene_count=len(self.manifest["scene_names"]),
            split="train",
            test_frame_count=int(self.manifest["test_frame_count"]),
            ldr_event_id="random",
            event_y_flip=self.manifest["event_y_flip"],
            event_spatial_transform=self.manifest["event_spatial_transform"],
            event_resize_method=self.manifest["event_resize_method"],
            event_resize_bins=int(self.manifest["event_resize_bins"]),
            return_normal_gt=False,
            return_debug_event_fields=False,
        )

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _stack(sample, key):
        values = []
        for view in sample:
            value = view[key]
            values.append(torch.from_numpy(value) if isinstance(value, np.ndarray) else value)
        return torch.stack(values)

    def __getitem__(self, index):
        record = self.records[index]
        dataset_index = int(record["dataset_index"])
        num_views = int(self.manifest["num_views"])
        sample_a = self.dataset[(dataset_index, 0, num_views, record["ldr_a"])]
        sample_b = self.dataset[(dataset_index, 0, num_views, record["ldr_b"])]
        event_a = self._stack(sample_a, "event_voxel").float()
        event_b = self._stack(sample_b, "event_voxel").float()
        if not torch.equal(event_a, event_b):
            raise RuntimeError(f"Event mismatch in paired record {index}.")
        rgb_a = self._stack(sample_a, "img").float()
        rgb_b = self._stack(sample_b, "img").float()
        cached = np.load(self.root / record["target"])
        target = torch.from_numpy(cached["target"].astype(np.float32) / 255.0).unsqueeze(1)
        weight = torch.from_numpy(cached["weight"].astype(np.float32) / 255.0).unsqueeze(1)
        return {
            "event": event_a,
            "rgb_a": rgb_a,
            "rgb_b": rgb_b,
            "target": target,
            "weight": weight,
            "scene": record["scene"],
        }

