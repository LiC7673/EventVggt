"""Dataset for rendered real-event reliability labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class RenderedReliabilityDataset(Dataset):
    def __init__(self, data_dir: str, *, split: str = "train") -> None:
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / f"manifest_{split}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing reliability manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.items: List[dict] = list(manifest.get("items", []))
        if not self.items:
            raise RuntimeError(f"No rendered reliability samples in {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        path = Path(item["path"])
        if not path.is_file():
            path = self.data_dir / path
        data = np.load(path)
        return {
            "rgb": torch.from_numpy(data["rgb"].astype(np.float32, copy=False)),
            "event_full": torch.from_numpy(data["event_full"].astype(np.float32, copy=False)),
            "target_reliability": torch.from_numpy(data["target_reliability"].astype(np.float32, copy=False)),
            "weight": torch.from_numpy(data["weight"].astype(np.float32, copy=False)),
            "event_support": torch.from_numpy(data["event_support"].astype(np.float32, copy=False)),
            "geometry_support": torch.from_numpy(data["geometry_support"].astype(np.float32, copy=False)),
            "path": str(path),
            "label": str(item.get("label", path.stem)),
        }
