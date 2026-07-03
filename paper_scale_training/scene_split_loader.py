"""Scene-disjoint paired-Multi-LDR train and fixed-LDR validation loaders."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from mul_loss_fine.finetune_mul_ldr_event import MultiLdrBatchSampler, _format_ldr, _to_list


def load_scene_split(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    splits = data.get("splits", {})
    required = {"train", "val", "test"}
    if not required.issubset(splits):
        raise ValueError(f"Scene manifest must contain {sorted(required)}")
    sets = {name: set(splits[name]) for name in required}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Scene manifest contains train/val/test overlap")
    return data


def _dataset(cfg, scene_names, ldr_event_id):
    requested_scenes = list(scene_names)
    dataset = get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=list(scene_names),
        initial_scene_idx=0,
        active_scene_count=len(scene_names),
        split="all",
        test_frame_count=0,
        ldr_event_id=ldr_event_id,
        event_y_flip=getattr(cfg.data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(cfg.data, "event_spatial_transform", "auto"),
        event_resize_method=getattr(cfg.data, "event_resize_method", "voxel_antialias"),
        event_resize_bins=getattr(cfg.data, "event_resize_bins", 10),
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    missing = sorted(set(requested_scenes) - set(dataset.scenes))
    if missing:
        raise ValueError(
            f"The requested split contains scenes unavailable at LDR={ldr_event_id}: {missing}"
        )
    dataset.set_active_scenes(requested_scenes)
    return dataset


def build_scene_disjoint_loader(cfg, split="train"):
    manifest = load_scene_split(cfg.data.scene_split_manifest)
    if split == "train":
        scene_names = manifest["splits"]["train"]
        dataset = _dataset(cfg, scene_names, "random")
        requested = [_format_ldr(value) for value in _to_list(cfg.data.mul_ldr_train_ids)]
        available = dataset.get_active_ldr_events(common=True)
        ldr_ids = requested or available
        missing = [value for value in ldr_ids if value not in available]
        if missing:
            raise ValueError(
                f"Paired Multi-LDR levels {missing} are unavailable in all 60 train scenes. "
                f"Common levels: {available}"
            )
        sampler = MultiLdrBatchSampler(
            dataset,
            scenes_per_batch=1,
            ldr_event_ids=ldr_ids,
            num_views=cfg.data.num_views,
            exposures_per_sample=2,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=event_multiview_collate,
        )
        print(
            f"[scene split] train scenes={len(scene_names)} windows={len(dataset)} "
            f"paired_LDR={ldr_ids} batches={len(loader)}"
        )
        return loader

    scene_names = manifest["splits"]["val"]
    eval_ldr = str(getattr(cfg.data, "eval_ldr_event_id", "ev_5"))
    dataset = _dataset(cfg, scene_names, eval_ldr)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    print(
        f"[scene split] validation scenes={len(scene_names)} windows={len(dataset)} "
        f"LDR={eval_ldr} batches={len(loader)}"
    )
    return loader
