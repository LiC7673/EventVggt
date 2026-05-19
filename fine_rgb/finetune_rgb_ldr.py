from pathlib import Path

import hydra
from omegaconf import OmegaConf

from launcher import launch_rgb_ldr


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_no_event.yaml",
)
def run(cfg: OmegaConf):
    launch_rgb_ldr(cfg)


if __name__ == "__main__":
    run()
