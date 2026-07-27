"""Three-epoch RGB-only finetuning on four fixed evaluation scenes."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_no_event as rgb_fe  # noqa: E402
from fine_rgb.finetune_rgb_seven_scenes_once import (  # noqa: E402
    build_once_rgb_loader,
    save_minimal_code_snapshot,
)
from fine_rgb.launcher import configure_rgb_ldr_cfg  # noqa: E402


FOUR_SCENES = [
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
]


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_no_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.scene_names = list(FOUR_SCENES)
    cfg.data.initial_scene_idx = 0
    cfg.data.active_scene_count = len(FOUR_SCENES)
    cfg.epochs = 3
    cfg.start_epoch = 0
    cfg = configure_rgb_ldr_cfg(cfg)

    # Clips within one epoch do not overlap. Repeating three epochs means that
    # every retained source frame is visited at most three times in total.
    rgb_fe.build_rgb_loader = build_once_rgb_loader
    rgb_fe.save_current_code = save_minimal_code_snapshot
    print(
        "Launching four-scene RGB-only finetune: "
        f"exposure={cfg.data.ldr_event_id}, epochs={cfg.epochs}, "
        f"pose_weight={float(cfg.loss.pose_weight):.4f}, "
        f"align_depth_scale={bool(cfg.loss.align_depth_scale)}, "
        f"unfreeze_heads={bool(cfg.train.unfreeze_heads)}, "
        f"unfreeze_aggregator={bool(cfg.train.unfreeze_aggregator_blocks)}, "
        f"lr={float(cfg.lr):.3g}"
    )
    if float(cfg.loss.pose_weight) <= 0:
        raise ValueError("pose_weight must be positive for this experiment")
    rgb_fe.train(cfg)


if __name__ == "__main__":
    run()
