from pathlib import Path

import hydra
from omegaconf import OmegaConf

from launcher import launch_mul_loss


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    launch_mul_loss(
        cfg,
        exp_name="mul_loss_mv_presence",
        weights={
            "mv_presence_weight": 0.05,
            "mv_presence_margin": 0.04,
        },
    )


if __name__ == "__main__":
    run()
