"""Ordered Multi-LDR pair dataset for Stage-1 event-contribution learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from torch.utils.data import Dataset

from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset


def canonical_ldr_id(value: str | int) -> str:
    text = str(value).strip().lower()
    if text.startswith("ev_"):
        return text
    return f"ev_{int(text)}"


def parse_ordered_pairs(values: Iterable[str]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for value in values:
        text = str(value).strip()
        separator = "->" if "->" in text else ":" if ":" in text else None
        if separator is None:
            raise ValueError(f"LDR pair must look like ev_1->ev_5, got {value!r}")
        left, right = text.split(separator, 1)
        pair = (canonical_ldr_id(left), canonical_ldr_id(right))
        if pair[0] == pair[1]:
            raise ValueError(f"An exposure pair must contain two distinct levels: {value!r}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("At least one ordered LDR pair is required.")
    return tuple(pairs)


def parse_exposure_sequence(values: str | Iterable[str | int]) -> tuple[str, ...]:
    """Parse a low-to-high exposure sequence such as ``0,1,2,5,10``."""
    if isinstance(values, str):
        raw_values = [value for value in values.split(",") if value.strip()]
    else:
        raw_values = list(values)
    exposures = tuple(canonical_ldr_id(value) for value in raw_values)
    if len(exposures) < 2:
        raise ValueError(f"At least two exposure levels are required, got {exposures}")
    if len(set(exposures)) != len(exposures):
        raise ValueError(f"Exposure levels must be unique and ordered low-to-high: {exposures}")
    numeric = [float(value.removeprefix("ev_")) for value in exposures]
    if any(right <= left for left, right in zip(numeric, numeric[1:])):
        raise ValueError(f"Exposure levels are not strictly increasing: {exposures}")
    return exposures


def generate_ordered_pairs(
    exposures: Sequence[str | int],
    mode: str = "all",
    explicit_pairs: Iterable[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Generate deterministic lower-to-higher exposure pairs.

    ``all`` creates every combination, ``adjacent`` creates neighboring
    levels, ``anchor`` pairs the lowest exposure with every higher level, and
    ``explicit`` uses repeatable ``--pair`` values while still enforcing the
    declared exposure order.
    """
    ordered = parse_exposure_sequence(exposures)
    index = {value: position for position, value in enumerate(ordered)}
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "all":
        pairs = [
            (ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        ]
    elif normalized_mode == "adjacent":
        pairs = list(zip(ordered[:-1], ordered[1:]))
    elif normalized_mode == "anchor":
        pairs = [(ordered[0], value) for value in ordered[1:]]
    elif normalized_mode == "explicit":
        parsed = parse_ordered_pairs(explicit_pairs or ())
        pairs = []
        for left, right in parsed:
            if left not in index or right not in index:
                raise ValueError(
                    f"Explicit pair {(left, right)} contains a level outside {ordered}"
                )
            if index[left] == index[right]:
                continue
            pairs.append((left, right) if index[left] < index[right] else (right, left))
    else:
        raise ValueError(f"Unknown pair mode {mode!r}; use all, adjacent, anchor, or explicit")
    deduplicated = tuple(dict.fromkeys(pairs))
    if not deduplicated:
        raise ValueError(f"Pair mode {mode!r} generated no exposure pairs")
    return deduplicated


@dataclass(frozen=True)
class PairRecord:
    dataset_index: int
    ldr_a: str
    ldr_b: str


class MultiLdrContributionDataset(Dataset):
    """Load the same views/events at one of a small set of exposure pairs."""

    def __init__(
        self,
        *,
        root: str,
        split: str,
        num_views: int,
        resolution: Sequence[int],
        fps: int,
        seed: int,
        ordered_pairs: Sequence[tuple[str, str]],
        scene_names=None,
        initial_scene_idx: int = 0,
        active_scene_count: int = 3,
        test_frame_count: int = 10,
        min_train_start_id: int = 0,
        event_y_flip="auto",
        event_spatial_transform="auto",
        event_resize_method: str = "voxel_antialias",
        event_resize_bins: int = 10,
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.ordered_pairs = tuple(
            (canonical_ldr_id(a), canonical_ldr_id(b)) for a, b in ordered_pairs
        )
        self.dataset = get_combined_dataset(
            root=root,
            num_views=self.num_views,
            resolution=tuple(resolution),
            fps=int(fps),
            seed=int(seed),
            scene_names=scene_names,
            initial_scene_idx=int(initial_scene_idx),
            active_scene_count=int(active_scene_count),
            split=str(split),
            test_frame_count=int(test_frame_count),
            min_train_start_id=int(min_train_start_id),
            ldr_event_id="random",
            event_y_flip=event_y_flip,
            event_spatial_transform=event_spatial_transform,
            event_resize_method=event_resize_method,
            event_resize_bins=int(event_resize_bins),
            return_normal_gt=False,
            return_debug_event_fields=False,
        )
        available = {canonical_ldr_id(value) for value in self.dataset.get_active_ldr_events(common=True)}
        requested = {value for pair in self.ordered_pairs for value in pair}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested LDR levels {missing} are unavailable; common={sorted(available)}")
        self.records = [
            PairRecord(index, pair[0], pair[1])
            for index in range(len(self.dataset))
            for pair in self.ordered_pairs
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        sample_a = self.dataset[(record.dataset_index, 0, self.num_views, record.ldr_a)]
        sample_b = self.dataset[(record.dataset_index, 0, self.num_views, record.ldr_b)]
        return {
            "sample_a": sample_a,
            "sample_b": sample_b,
            "dataset_index": record.dataset_index,
            "ldr_a": record.ldr_a,
            "ldr_b": record.ldr_b,
        }


def contribution_pair_collate(batch):
    return {
        "views_a": event_multiview_collate([item["sample_a"] for item in batch]),
        "views_b": event_multiview_collate([item["sample_b"] for item in batch]),
        "dataset_index": [int(item["dataset_index"]) for item in batch],
        "ldr_a": [item["ldr_a"] for item in batch],
        "ldr_b": [item["ldr_b"] for item in batch],
    }


__all__ = [
    "MultiLdrContributionDataset",
    "PairRecord",
    "canonical_ldr_id",
    "contribution_pair_collate",
    "generate_ordered_pairs",
    "parse_exposure_sequence",
    "parse_ordered_pairs",
]
