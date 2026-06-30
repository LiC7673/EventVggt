"""Predict additive branch tokens from full events and RGB, then refine geometry."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf

from event_branch_ablation.common import train_full_decomposition


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    train_full_decomposition(cfg)


if __name__ == "__main__":
    run()

