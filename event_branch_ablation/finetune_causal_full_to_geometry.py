"""Train additive decomposition with an event-causal detail residual."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from event_branch_ablation.causal_loss import make_causal_additive_loss
from event_branch_ablation.common import (
    _configure_trainable,
    _model_kwargs,
    _prepare_cfg,
    _safe_save_current_code,
)
from event_branch_ablation.data import build_full_decomposition_loader
from event_branch_ablation.plots import install_event_plot_hook
from event_branch_ablation.visualization import install_event_bin_visualization_hook
from eventvggt.models.streamvggt_causal_additive_detail import StreamVGGT


def _build_model(cfg):
    kwargs = _model_kwargs(cfg)
    kwargs["decomposition_hidden_dim"] = int(cfg.model.decomposition_hidden_dim)
    kwargs["event_support_tau"] = float(cfg.model.event_support_tau)
    kwargs["event_support_dilate_kernel"] = int(cfg.model.event_support_dilate_kernel)
    kwargs["event_support_blur_kernel"] = int(cfg.model.event_support_blur_kernel)
    return StreamVGGT(**kwargs)


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    cfg = _prepare_cfg(
        cfg,
        exp_name="causal_full_to_geometry_scene12",
        variant="causal_full_to_geometry",
    )
    cfg.model.event_support_tau = float(getattr(cfg.model, "event_support_tau", 0.50))
    cfg.model.event_support_dilate_kernel = int(
        getattr(cfg.model, "event_support_dilate_kernel", 5)
    )
    cfg.model.event_support_blur_kernel = int(
        getattr(cfg.model, "event_support_blur_kernel", 3)
    )
    cfg.loss.img_event_reliability_weight = 0.0
    cfg.loss.img_event_reject_weight = 0.0
    cfg.loss.causal_reliability_weight = float(
        getattr(cfg.loss, "causal_reliability_weight", 0.50)
    )
    cfg.loss.causal_partition_weight = float(
        getattr(cfg.loss, "causal_partition_weight", 0.50)
    )
    cfg.loss.branch_token_weight = float(getattr(cfg.loss, "branch_token_weight", 0.50))
    cfg.loss.branch_geometry_weight = 1.0
    cfg.loss.branch_material_weight = 0.75
    cfg.loss.branch_noise_weight = 0.50
    cfg.loss.branch_consistency_weight = 0.0

    fe.build_event_loader = build_full_decomposition_loader
    fe.build_event_model = _build_model
    fe.configure_trainable_params = _configure_trainable
    fe.EventSupervisedLoss = make_causal_additive_loss(cfg)
    fe.save_current_code = _safe_save_current_code
    install_event_plot_hook(fe)
    install_event_bin_visualization_hook(fe)
    print(f"[causal-additive] output={cfg.output_dir}, epochs={cfg.epochs}, lr={cfg.lr}")
    fe.train(cfg)


if __name__ == "__main__":
    run()
