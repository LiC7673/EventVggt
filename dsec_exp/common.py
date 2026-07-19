from __future__ import annotations

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from eventvggt.datasets.dsec_event_dataset import get_dsec_dataset
from eventvggt.datasets.my_event_dataset import event_multiview_collate


def _plain(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def build_dsec_loader(cfg, split="train", *, rgb_only=False):
    data = cfg.data
    dataset = get_dsec_dataset(
        root=str(data.root),
        split=split,
        num_views=int(data.num_views),
        resolution=tuple(data.resolution),
        seed=int(cfg.seed),
        sequence_names=_plain(getattr(data, "sequence_names", None)),
        event_window_ms=float(getattr(data, "event_window_ms", 50.0)),
        event_resize_bins=int(getattr(data, "event_resize_bins", 10)),
        clip_stride=int(getattr(data, "train_clip_stride" if split == "train" else "test_clip_stride", 4)),
        allow_unaligned_rgb=bool(getattr(data, "allow_unaligned_rgb", False)),
        depth_scale=float(getattr(data, "depth_scale", 1.0)),
        disparity_fx=getattr(data, "disparity_fx", None),
        disparity_baseline=getattr(data, "disparity_baseline", None),
        max_depth=float(getattr(cfg.loss, "depth_max", 80.0) or 80.0),
    )

    def collate(batch):
        views = event_multiview_collate(batch)
        if rgb_only:
            for view in views:
                for key in ("event_xy", "event_t", "event_p", "event_voxel", "event_time_range", "has_event"):
                    view.pop(key, None)
        return views

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=(split == "train"),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_mem),
        drop_last=(split == "train"),
        collate_fn=collate,
    )
    print(
        f"DSEC {split}: root={data.root}/{('val' if split == 'train' else 'test')} "
        f"scenes={dataset.get_active_scenes()} clips={len(dataset)} rgb_only={rgb_only}"
    )
    return loader

