"""Full image-guided reliability trained with geometry_motion events only."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf

from event_branch_ablation.common import train_geometry_motion


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    train_geometry_motion(cfg)


if __name__ == "__main__":
    run()

