"""Stage 3: jointly finetune ReliabilityNet and geometry modules at low LR."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf

from reliability_staged_finetune.training import train_stage


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    train_stage(cfg, stage=3)


if __name__ == "__main__":
    run()
