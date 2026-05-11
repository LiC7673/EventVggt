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
    launch(cfg, exp_name="ablation_tokens_16", branch_mode="global_local", num_global_tokens=16)


if __name__ == "__main__":
    run()
