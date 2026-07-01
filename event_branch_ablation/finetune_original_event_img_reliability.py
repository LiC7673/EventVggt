"""Train the best image-guided reliability path with the original event stream.

This is a control experiment for the additive-event branch experiments. It uses
the normal finetune_event.py dataloader, i.e. the original LDR event stream, and
does not read events_additive/{full,geometry_motion,material_reflection,noise}.
Outputs are written under abl_event_exp instead of checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from event_branch_ablation.common import (  # noqa: E402
    FULL_RELIABILITY_WEIGHTS,
    OUTPUT_ROOT,
    _safe_save_current_code,
)
from event_branch_ablation.plots import install_event_plot_hook  # noqa: E402
from event_branch_ablation.visualization import install_event_bin_visualization_hook  # noqa: E402
from mul_loss_fine.image_guided_event_reliability_loss import (  # noqa: E402
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg  # noqa: E402


def _configure_event_refiner_and_depth_heads(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "event_detail_refiner" in name:
            param.requires_grad = True

    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True


def _resolve_pretrained(cfg) -> None:
    current = str(getattr(cfg, "pretrained", "") or "")
    if current not in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        return
    original = ROOT_DIR / "ckpt" / "model.pt"
    cfg.pretrained = str(original) if original.exists() else "./ckpt/model.pt"


def _experiment_name(cfg) -> str:
    current = str(getattr(cfg, "exp_name", "") or "")
    if current in {"", "event_finetune_LDR5"}:
        return "original_event_img_reliability_scene12"
    return current


def _set_output_paths(cfg, exp_name: str) -> None:
    cfg.exp_name = exp_name
    cfg.save_dir = str(OUTPUT_ROOT)
    cfg.output_dir = str(OUTPUT_ROOT / exp_name)
    cfg.logdir = str(OUTPUT_ROOT / exp_name / "logs")


def _prepare_cfg(cfg: OmegaConf) -> OmegaConf:
    OmegaConf.set_struct(cfg, False)
    for branch in ("model", "train", "loss", "data", "vis"):
        OmegaConf.set_struct(getattr(cfg, branch), False)

    _set_output_paths(cfg, _experiment_name(cfg))

    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = int(getattr(cfg.model, "imgrel_event_hidden_dim", 16))
    cfg.model.refiner_residual_scale = float(getattr(cfg.model, "imgrel_refiner_residual_scale", 0.035))
    cfg.model.event_delta_highpass_kernel = int(getattr(cfg.model, "imgrel_event_delta_highpass_kernel", 9))
    cfg.model.event_delta_patch_zero_mean = bool(getattr(cfg.model, "imgrel_event_delta_patch_zero_mean", True))
    cfg.model.event_delta_patch_size = int(getattr(cfg.model, "imgrel_event_delta_patch_size", 14))
    cfg.model.event_delta_abs_limit = float(getattr(cfg.model, "imgrel_event_delta_abs_limit", 0.025))
    cfg.model.event_reliability_gate_enabled = True
    cfg.model.event_reliability_gate_floor = float(getattr(cfg.model, "imgrel_event_reliability_gate_floor", 0.20))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "imgrel_event_reliability_init_bias", 0.0))

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.loss.pose_weight = float(getattr(cfg.loss, "pose_weight", 1.0))
    cfg.loss.depth_weight = 1.0
    cfg.loss.points_weight = 1.0
    cfg.data.return_normal_gt = True

    cfg.vis.event_bins_enabled = bool(getattr(cfg.vis, "event_bins_enabled", True))
    cfg.vis.event_bins_count = int(getattr(cfg.vis, "event_bins_count", cfg.model.event_num_bins))
    cfg.vis.event_bins_num_views = int(getattr(cfg.vis, "event_bins_num_views", min(4, int(cfg.data.num_views))))
    cfg.vis.event_bin_panel_width = int(getattr(cfg.vis, "event_bin_panel_width", 160))

    cfg.epochs = max(int(getattr(cfg, "epochs", 10)), int(getattr(cfg, "imgrel_epochs", 20)))
    if float(getattr(cfg, "lr", 1.0e-4)) > 4.0e-5:
        cfg.lr = 4.0e-5

    _resolve_pretrained(cfg)
    return configure_mul_loss_cfg(
        cfg,
        weights=FULL_RELIABILITY_WEIGHTS,
        exp_name=str(cfg.exp_name),
    )


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    cfg = _prepare_cfg(cfg)
    fe.configure_trainable_params = _configure_event_refiner_and_depth_heads
    fe.EventSupervisedLoss = make_configured_image_guided_event_reliability_loss(cfg)
    fe.save_current_code = _safe_save_current_code
    install_event_plot_hook(fe)
    install_event_bin_visualization_hook(fe)
    print(
        "[original-event-img-reliability] "
        "normal finetune_event loader; no events_additive branches; "
        f"output={cfg.output_dir}, epochs={cfg.epochs}, lr={cfg.lr}, pretrained={cfg.pretrained}"
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
