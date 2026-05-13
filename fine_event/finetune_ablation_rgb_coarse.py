from pathlib import Path

import hydra
from omegaconf import OmegaConf

from fine_event.finetune_event_global_local import launch


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event_global_local.yaml",
)
def run(cfg: OmegaConf):
    cfg.train.unfreeze_heads = True
    launch(cfg, exp_name="ablation_rgb_coarse", branch_mode="rgb_coarse")


if __name__ == "__main__":
    run()
