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
            "detail_gt_hf_weight": 0.8,
            "detail_gt_grad_weight": 0.8,
            "detail_gt_event_boost": 0.75,
            "detail_gt_threshold": 0.02,
            "mv_event_support_mode": "temporal_polarity",
            "mv_event_threshold": 0.05,
            "mv_event_dilate_kernel": 1,
            "mv_event_blur_kernel": 3,
        },
    )


if __name__ == "__main__":
    run()
