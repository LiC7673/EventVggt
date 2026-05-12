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
    cfg.model.disable_second_stage = False

    cfg.train.two_stage_train_mode = "stage2"
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.train.unfreeze_aggregator_last_blocks = 0

    cfg.loss.pose_weight = 0.0
    cfg.loss.points_weight = 0.0
    cfg.loss.normal_weight = 0.0
    cfg.loss.residual_depth_weight = 1.0
    cfg.loss.coarse_depth_weight = 0.0

    launch(
        cfg,
        exp_name="ablation_twostage_stage2_freeze_stage1",
        residual_input_mode="global_rgb_current_event",
        train_mode="stage2",
    )


if __name__ == "__main__":
    run()
