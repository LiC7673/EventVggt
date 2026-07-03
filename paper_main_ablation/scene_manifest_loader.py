"""Fixed-scene loaders shared by every paper module-ablation row."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from mul_loss_fine.finetune_mul_ldr_event import MultiLdrBatchSampler, _format_ldr, _to_list
from paper_main_ablation.common import uses_multildr


def _scene_names(cfg):
    with Path(str(cfg.data.module_scene_manifest)).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scenes = list(manifest.get("training_scenes", []))
    if not scenes:
        raise RuntimeError("module_scene_manifest does not contain training_scenes")
    return scenes


def _dataset(cfg, split, ldr_event_id):
    scenes = _scene_names(cfg)
    dataset = get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=scenes,
        initial_scene_idx=0,
        active_scene_count=len(scenes),
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=ldr_event_id,
        event_y_flip=getattr(cfg.data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(cfg.data, "event_spatial_transform", "auto"),
        event_resize_method=getattr(cfg.data, "event_resize_method", "voxel_antialias"),
        event_resize_bins=getattr(cfg.data, "event_resize_bins", 10),
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    missing = sorted(set(scenes) - set(dataset.scenes))
    if missing:
        raise RuntimeError(f"Fixed training scenes unavailable at LDR={ldr_event_id}: {missing}")
    dataset.set_active_scenes(scenes)
    return dataset


def build_module_scene_loader(cfg, split="train"):
    variant = str(cfg.main_table_variant).lower()
    if split == "train" and uses_multildr(variant):
        dataset = _dataset(cfg, "train", "random")
        requested = [_format_ldr(value) for value in _to_list(cfg.data.mul_ldr_train_ids)]
        available = dataset.get_active_ldr_events(common=True)
        missing = [value for value in requested if value not in available]
        if missing:
            raise ValueError(f"Multi-LDR levels {missing} unavailable; common={available}")
        sampler = MultiLdrBatchSampler(
            dataset,
            scenes_per_batch=1,
            ldr_event_ids=requested,
            num_views=cfg.data.num_views,
            exposures_per_sample=2,
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

    ldr_id = str(
        getattr(cfg.data, "eval_ldr_event_id", "ev_5")
        if split != "train"
        else getattr(cfg.data, "ldr_event_id", "ev_5")
    )
    dataset = _dataset(cfg, split, ldr_id)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=(split == "train"),
        collate_fn=event_multiview_collate,
    )


__all__ = ["build_module_scene_loader"]
