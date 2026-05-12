from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf

from fine_event.finetune_event_two_stage_residual import launch


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event_two_stage_residual.yaml",
)
def run(cfg: OmegaConf):
    launch(
        cfg,
        exp_name="ablation_twostage_event_current_rgb",
        residual_input_mode="event_current_rgb",
    )


if __name__ == "__main__":
    run()
