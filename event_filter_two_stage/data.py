"""Full-event loaders for decomposer pretraining and frozen Stage-2 use."""

from __future__ import annotations

from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate
from fine_event.finetune_event_random_ldr import RandomLdrBatchSampler

from event_branch_ablation.data import (
    FixedWindowAdditiveDataset,
    _build_base_dataset,
    switch_event_source,
)


def build_full_event_dataset(cfg, *, split: str, attach_targets: bool):
    random_train = split == "train" and bool(getattr(cfg.data, "random_train_ldr", True))
    ldr_id = "random" if random_train else str(getattr(cfg.data, "eval_ldr_event_id", "ev_5"))
    dataset = _build_base_dataset(cfg, split, ldr_id)
    root_name = str(getattr(cfg.data, "additive_event_root", "events_additive"))
    switch_event_source(dataset, branch="full", root_name=root_name)
    dataset = FixedWindowAdditiveDataset(
        dataset,
        primary_branch="full",
        attach_targets=attach_targets,
        root_name=root_name,
        mask_dilate_kernel=int(getattr(cfg.data, "additive_mask_dilate_kernel", 5)),
    )
    return dataset


def make_loader(cfg, dataset, *, split: str):
    if split == "train" and bool(getattr(cfg.data, "random_train_ldr", True)):
        exposures = list(dataset.get_active_ldr_events(common=True))
        if not exposures:
            raise ValueError("No common LDR exposure exists across selected scenes.")
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


def build_stage2_full_event_loader(cfg, split: str = "train"):
    dataset = build_full_event_dataset(cfg, split=split, attach_targets=False)
    fe.printer.info(
        "Two-stage full-event loader split=%s scenes=%s samples=%d",
        split, dataset.get_active_scenes(), len(dataset),
    )
    return make_loader(cfg, dataset, split=split)
