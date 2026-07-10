"""Scene-disjoint datasets used by every fast ablation condition."""

from __future__ import annotations

from torch.utils.data import Dataset

from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from paired_token_reliability.contribution_dataset import MultiLdrContributionDataset


class SingleExposureGeometryDataset(Dataset):
    """One target exposure only; no reference exposure is loaded or compared."""

    def __init__(self, target_exposures, **kwargs) -> None:
        self.num_views = int(kwargs.pop("num_views"))
        self.target_exposures = tuple(str(value) for value in target_exposures)
        if not self.target_exposures:
            raise ValueError("At least one target exposure is required")
        self.dataset = get_combined_dataset(
            num_views=self.num_views,
            ldr_event_id="random",
            return_normal_gt=False,
            return_debug_event_fields=False,
            **kwargs,
        )

    def __len__(self) -> int:
        # Match the number of optimizer steps used by the Multi-LDR condition
        # without loading a second exposure or creating extra image variants.
        return len(self.dataset) * len(self.target_exposures)

    def __getitem__(self, index: int):
        # The legacy Stage-1 batch interface expects two slots. They point to
        # the same target sample, so no second exposure is ever loaded.
        dataset_index = index // len(self.target_exposures)
        target_exposure = self.target_exposures[index % len(self.target_exposures)]
        sample = self.dataset[(dataset_index, 0, self.num_views, target_exposure)]
        return {
            "sample_a": sample,
            "sample_b": sample,
            "dataset_index": dataset_index,
            "ldr_a": target_exposure,
            "ldr_b": target_exposure,
        }


def single_exposure_collate(batch):
    return {
        "views_a": event_multiview_collate([item["sample_a"] for item in batch]),
        "views_b": event_multiview_collate([item["sample_b"] for item in batch]),
        "dataset_index": [int(item["dataset_index"]) for item in batch],
        "ldr_a": [item["ldr_a"] for item in batch],
        "ldr_b": [item["ldr_b"] for item in batch],
    }


def stage1_dataset(cfg, split: str, pairs, method: str):
    data = cfg.data
    is_train = split == "train"
    initial_scene_idx = int(
        getattr(data, "train_initial_scene_idx", 0)
        if is_train
        else getattr(data, "test_initial_scene_idx", 20)
    )
    scene_count = int(
        getattr(data, "train_scene_count", 20)
        if is_train
        else getattr(data, "test_scene_count", 5)
    )
    common = dict(
        root=str(data.root),
        split=split,
        num_views=int(data.num_views),
        resolution=tuple(data.resolution),
        fps=int(data.fps),
        seed=int(cfg.seed),
        scene_names=None,
        initial_scene_idx=initial_scene_idx,
        active_scene_count=scene_count,
        test_frame_count=int(
            getattr(data, "train_holdout_frame_count", 0)
            if is_train
            else getattr(data, "heldout_test_frame_count", 120)
        ),
        min_train_start_id=int(getattr(data, "train_min_start_id", 2)) if is_train else 0,
        event_y_flip=getattr(data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(data, "event_spatial_transform", "auto"),
        event_resize_method=str(getattr(data, "event_resize_method", "voxel_antialias")),
        event_resize_bins=int(getattr(data, "event_resize_bins", 10)),
    )
    if method == "ours":
        return MultiLdrContributionDataset(ordered_pairs=pairs, **common)
    if method == "no_multildr":
        # Match the exact target-exposure frequency and optimizer-step count of
        # the paired condition, but never load the corresponding reference.
        return SingleExposureGeometryDataset(
            target_exposures=[bad_exposure for _reference, bad_exposure in pairs],
            **common,
        )
    raise ValueError(f"Stage 1 is not trained for method={method!r}")


__all__ = [
    "SingleExposureGeometryDataset",
    "single_exposure_collate",
    "stage1_dataset",
]
