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
        exp_name="mul_loss_detail_gt_salient",
        weights={
            "detail_gt_normal_weight": 0.1,
            "detail_gt_hf_weight": 0.2,
            "detail_gt_grad_weight": 0.0,
            "detail_gt_event_boost": 0.75,
            "detail_gt_threshold": 0.02,
            "detail_gt_salient_hf_weight": 0.6,
            "detail_gt_salient_mag_weight": 0.25,
            "detail_gt_salient_presence_weight": 0.25,
            "detail_gt_salient_threshold": 0.35,
            "detail_gt_salient_power": 2.0,
            "detail_gt_salient_presence_ratio": 0.8,
            "detail_gt_chunk_size": 1,
            "mv_hf_kernel": 7,
        },
    )


if __name__ == "__main__":
    run()
