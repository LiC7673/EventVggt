import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_no_event as rgb_fe  # noqa: E402
from fine_rgb.rgb_ldr_dataset import get_rgb_ldr_dataset  # noqa: E402


def normalize_ldr_id(value):
    if value is None:
        return "auto"
    value = str(value)
    if value.lower() == "auto":
        return "auto"
    return value if value.startswith("ev_") else f"ev_{value}"


def build_fixed_ldr_rgb_loader(cfg, split="train"):
    ldr_event_id = normalize_ldr_id(getattr(cfg.data, "ldr_event_id", "auto"))
    dataset = get_rgb_ldr_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=cfg.data.scene_names if cfg.data.scene_names else None,
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=ldr_event_id,
        return_normal_gt=getattr(cfg.data, "return_normal_gt", False),
    )
    if len(dataset) <= 0:
        raise ValueError(
            f"RGB LDR dataset has no valid samples under {cfg.data.root}. "
            f"ldr_event_id={ldr_event_id}, num_views={cfg.data.num_views}, "
            f"active_scenes={dataset.get_active_scenes()}"
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=(split == "train"),
        collate_fn=rgb_fe.rgb_multiview_collate,
    )
    print(
        f"Pure-RGB {split} loader: ldr={ldr_event_id}, "
        f"active_scenes={dataset.get_active_scenes()}, samples={len(dataset)}"
    )
    return loader


def configure_rgb_ldr_cfg(cfg):
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    ldr_event_id = normalize_ldr_id(getattr(cfg.data, "ldr_event_id", "auto"))
    cfg.data.ldr_event_id = ldr_event_id

    default_names = {"finetune_no_event_rgb", "event_finetune_LDR5"}
    if str(getattr(cfg, "exp_name", "")) in default_names:
        safe_ldr = ldr_event_id.replace("/", "_")
        cfg.exp_name = f"fine_rgb_{safe_ldr}"
    cfg.logdir = f"{cfg.save_dir}/{cfg.exp_name}/logs"
    cfg.output_dir = f"{cfg.save_dir}/{cfg.exp_name}"
    OmegaConf.resolve(cfg)
    return cfg


def launch_rgb_ldr(cfg):
    cfg = configure_rgb_ldr_cfg(cfg)
    rgb_fe.build_rgb_loader = build_fixed_ldr_rgb_loader
    print(f"Launching pure RGB LDR finetune: ldr_event_id={cfg.data.ldr_event_id}")
    rgb_fe.train(cfg)
