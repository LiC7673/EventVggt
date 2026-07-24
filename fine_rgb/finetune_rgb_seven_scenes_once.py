"""One-epoch RGB-only finetuning on seven fixed scenes.

Unlike the default sliding-window loader, the training loader below keeps
non-overlapping multi-view clips.  Therefore a source RGB frame is consumed at
most once during the epoch (an incomplete tail shorter than ``num_views`` is
discarded).  Test loading remains unchanged and covers the complete held-out
tail of every scene.
"""

import sys
import datetime
import shutil
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_no_event as rgb_fe  # noqa: E402
from fine_rgb.launcher import configure_rgb_ldr_cfg, normalize_ldr_id  # noqa: E402
from fine_rgb.rgb_ldr_dataset import get_rgb_ldr_dataset  # noqa: E402


SEVEN_SCENES = [
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
    "DH2_Socrates and Seneca_Car_Paint_Midnight",
    "Dragon_1_Car_Paint_Midnight",
    "NAPOLEON_fix_Anodized_Red",
]


def save_minimal_code_snapshot(outdir):
    """Snapshot only this experiment's dependencies, never the whole workspace.

    The generic RGB trainer copies ``Path.cwd()``.  With five concurrent
    experiments inside ``exp_f`` that makes every job copy the other jobs'
    output trees and causes recursive growth.
    """
    stamp = datetime.datetime.now().strftime("%m_%d-%H-%M-%S")
    destination = Path(outdir).resolve() / "code" / stamp
    source_files = [
        ROOT_DIR / "finetune_no_event.py",
        ROOT_DIR / "config" / "finetune_no_event.yaml",
        ROOT_DIR / "fine_rgb" / "finetune_rgb_seven_scenes_once.py",
        ROOT_DIR / "fine_rgb" / "launcher.py",
        ROOT_DIR / "fine_rgb" / "rgb_ldr_dataset.py",
        ROOT_DIR / "eventvggt" / "datasets" / "my_event_dataset.py",
    ]
    for source in source_files:
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT_DIR)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return str(destination)


def _non_overlapping_starts(start_img_ids, num_views):
    """Greedily retain disjoint consecutive clips independently per scene."""
    by_scene = defaultdict(list)
    for scene_name, start_id in start_img_ids:
        by_scene[scene_name].append(int(start_id))

    selected = []
    for scene_name, starts in by_scene.items():
        previous_end = None
        for start_id in sorted(set(starts)):
            if previous_end is None or start_id >= previous_end:
                selected.append((scene_name, start_id))
                previous_end = start_id + int(num_views)
    return selected


def build_once_rgb_loader(cfg, split="train"):
    ldr_event_id = normalize_ldr_id(getattr(cfg.data, "ldr_event_id", "auto"))
    dataset = get_rgb_ldr_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=list(cfg.data.scene_names),
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 120),
        ldr_event_id=ldr_event_id,
        return_normal_gt=getattr(cfg.data, "return_normal_gt", False),
    )

    if split == "train":
        original_windows = len(dataset.start_img_ids)
        dataset.start_img_ids = _non_overlapping_starts(
            dataset.start_img_ids, cfg.data.num_views
        )
        used_frames = set()
        for scene_name, start_id in dataset.start_img_ids:
            for frame_id in range(start_id, start_id + int(cfg.data.num_views)):
                key = (scene_name, frame_id)
                if key in used_frames:
                    raise RuntimeError(f"duplicate training RGB frame detected: {key}")
                used_frames.add(key)
        print(
            "[RGB-once] "
            f"{original_windows} sliding windows -> {len(dataset.start_img_ids)} "
            f"disjoint clips, {len(used_frames)} unique frames"
        )

    if len(dataset) <= 0:
        raise ValueError(
            f"No RGB samples under {cfg.data.root}; exposure={ldr_event_id}, "
            f"split={split}, scenes={list(cfg.data.scene_names)}"
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
        f"Pure-RGB {split}: exposure={ldr_event_id}, "
        f"scenes={dataset.get_active_scenes()}, clips={len(dataset)}"
    )
    return loader


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_no_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.scene_names = list(SEVEN_SCENES)
    cfg.data.initial_scene_idx = 0
    cfg.data.active_scene_count = len(SEVEN_SCENES)
    cfg.epochs = 1
    cfg.start_epoch = 0
    cfg = configure_rgb_ldr_cfg(cfg)

    rgb_fe.build_rgb_loader = build_once_rgb_loader
    rgb_fe.save_current_code = save_minimal_code_snapshot
    print(
        "Launching seven-scene RGB-only one-pass finetune: "
        f"exposure={cfg.data.ldr_event_id}"
    )
    rgb_fe.train(cfg)


if __name__ == "__main__":
    run()
