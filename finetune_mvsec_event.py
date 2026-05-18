from pathlib import Path

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.mvsec_event_dataset import get_mvsec_dataset
from eventvggt.datasets.my_event_dataset import event_multiview_collate


def _none_if_empty(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _as_plain(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def build_mvsec_event_loader(cfg, split="train"):
    data_cfg = cfg.data
    sequence_names = _none_if_empty(getattr(data_cfg, "sequence_names", None))
    intrinsics = _none_if_empty(getattr(data_cfg, "intrinsics", None))

    dataset = get_mvsec_dataset(
        root=data_cfg.root,
        num_views=data_cfg.num_views,
        resolution=tuple(data_cfg.resolution),
        fps=getattr(data_cfg, "fps", 20),
        seed=cfg.seed,
        split=split,
        sequence_names=_as_plain(sequence_names),
        camera=getattr(data_cfg, "camera", "left"),
        davis_group=getattr(data_cfg, "davis_group", "davis"),
        depth_key=getattr(data_cfg, "depth_key", "depth_image_rect"),
        pose_key=getattr(data_cfg, "pose_key", "pose"),
        event_format=getattr(data_cfg, "event_format", "xytp"),
        intrinsics=_as_plain(intrinsics),
        spatial_transform=getattr(data_cfg, "event_spatial_transform", "none"),
        test_frame_count=getattr(data_cfg, "test_frame_count", 20),
        event_resize_method=getattr(data_cfg, "event_resize_method", "voxel_antialias"),
        event_resize_bins=getattr(data_cfg, "event_resize_bins", 10),
        return_debug_event_fields=getattr(data_cfg, "return_debug_event_fields", False),
    )
    if len(dataset) <= 0:
        scene_stats = [
            {
                "scene": scene_name,
                "frame_count": int(meta["frame_count"]),
                "num_start_ids": int(len(meta.get("start_ids", []))),
                "event_index_source": meta.get("event_index_source", "unknown"),
            }
            for scene_name, meta in getattr(dataset, "scene_data", {}).items()
        ]
        raise ValueError(
            f"MVSEC dataset has no valid samples under {data_cfg.root}. "
            f"camera={getattr(data_cfg, 'camera', 'left')}, "
            f"num_views={data_cfg.num_views}, scenes={dataset.get_active_scenes()}, "
            f"scene_stats={scene_stats}"
        )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=(split == "train"),
        collate_fn=event_multiview_collate,
    )

    source_counts = {}
    for meta in getattr(dataset, "scene_data", {}).values():
        source = meta.get("event_index_source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    fe.printer.info(
        "MVSEC %s dataset loaded from %s: %d sequences, %d samples, event_index_sources=%s",
        split,
        data_cfg.root,
        len(dataset.get_active_scenes()),
        len(dataset),
        source_counts,
    )
    return data_loader


def launch(cfg):
    OmegaConf.resolve(cfg)
    fe.build_event_loader = build_mvsec_event_loader
    fe.train(cfg)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "config"),
    config_name="finetune_mvsec_event.yaml",
)
def run(cfg: OmegaConf):
    launch(cfg)


if __name__ == "__main__":
    run()
