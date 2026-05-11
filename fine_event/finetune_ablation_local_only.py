from pathlib import Path

import hydra
from omegaconf import OmegaConf

from finetune_event_global_local import launch


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "config"),
    config_name="finetune_event_global_local.yaml",
)
def run(cfg: OmegaConf):
    launch(cfg, exp_name="ablation_local_only", branch_mode="local_only")


if __name__ == "__main__":
    run()
