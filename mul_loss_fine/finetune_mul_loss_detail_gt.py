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
        exp_name="mul_loss_detail_gt",
        weights={
            "detail_gt_normal_weight": 0.2,
            "detail_gt_hf_weight": 0.5,
            "detail_gt_grad_weight": 0.5,
            "detail_gt_event_boost": 0.5,
            "detail_gt_threshold": 0.02,
        },
    )


if __name__ == "__main__":
    run()
