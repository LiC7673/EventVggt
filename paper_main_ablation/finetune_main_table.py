"""Train one strictly controlled row of the paper main ablation table."""

from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

import hydra
from accelerate import Accelerator as HFAccelerator
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import finetune_event as fe  # noqa: E402
from fine_event.finetune_event_random_ldr import build_random_ldr_event_loader  # noqa: E402
from mul_loss_fine.launcher import configure_mul_loss_cfg, make_configured_loss  # noqa: E402
from paper_main_ablation.common import (  # noqa: E402
    VARIANTS,
    build_model,
    configure_trainable_params,
    uses_detail_loss,
    uses_multildr,
)


COMMON_LOSS = {
    "mv_normal_weight": 0.0,
    "mv_presence_weight": 0.0,
    "mv_hf_weight": 0.0,
    "mv_orient_weight": 0.0,
    "detail_gt_normal_weight": 0.0,
    "detail_gt_hf_weight": 0.0,
    "detail_gt_grad_weight": 0.0,
    "detail_gt_event_boost": 0.0,
    "detail_gt_threshold": 0.02,
    "detail_gt_weight_power": 1.0,
    "detail_gt_normal_source": "depth",
    "detail_gt_chunk_size": 1,
    # These regularizers are identical for all rows and are not ablated.
    "residual_smooth_weight": 0.02,
    "residual_second_order_weight": 0.02,
    "residual_abs_weight": 0.01,
    "residual_smooth_alpha": 10.0,
    "final_grid_weight": 0.02,
    "final_phase_weight": 0.01,
    "final_grid_patch_size": 14,
    "final_grid_band": 1,
    "final_grid_detail_threshold": 0.02,
}

DETAIL_LOSS = {
    **COMMON_LOSS,
    "detail_gt_normal_weight": 0.25,
    "detail_gt_hf_weight": 1.0,
    "detail_gt_grad_weight": 1.0,
}


class UnevenBatchAccelerator(HFAccelerator):
    """Allow the Random-LDR custom batch sampler on old and new accelerate."""

    def __init__(self, *args, **kwargs):
        signature = inspect.signature(HFAccelerator.__init__)
        if "even_batches" in signature.parameters:
            kwargs.setdefault("even_batches", False)
        super().__init__(*args, **kwargs)
        try:
            self.even_batches = False
        except Exception:
            pass
        dataloader_config = getattr(self, "dataloader_config", None)
        if dataloader_config is not None and hasattr(dataloader_config, "even_batches"):
            dataloader_config.even_batches = False


def _safe_code_snapshot(outdir: str):
    destination = Path(outdir) / "code_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    relative_files = (
        "paper_main_ablation/finetune_main_table.py",
        "paper_main_ablation/common.py",
        "eventvggt/models/streamvggt_causal_temporal_detail.py",
        "eventvggt/models/streamvggt_pretrained_reliability_detail.py",
        "mul_loss_fine/event_supported_mv_loss.py",
        "mul_loss_fine/launcher.py",
        "finetune_event.py",
    )
    for relative in relative_files:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def _prepare_cfg(cfg, variant: str):
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data", "vis"):
        branch = getattr(cfg, branch_name, None)
        if branch is not None:
            OmegaConf.set_struct(branch, False)

    cfg.main_table_variant = variant
    cfg.model.variant = variant
    cfg.model.main_event_hidden_dim = int(getattr(cfg.model, "main_event_hidden_dim", 16))
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_count_cmax = float(getattr(cfg.model, "event_count_cmax", 3.0))
    cfg.model.refiner_residual_scale = 0.035
    cfg.model.event_delta_highpass_kernel = 9
    cfg.model.event_delta_patch_zero_mean = True
    cfg.model.event_delta_patch_size = int(cfg.model.patch_size)
    cfg.model.event_delta_abs_limit = 0.025
    cfg.model.causal_support_threshold = 0.01
    cfg.model.causal_support_dilate_kernel = 5
    cfg.model.causal_support_blur_kernel = 3
    cfg.model.reliability_checkpoint = str(
        getattr(
            cfg.model,
            "reliability_checkpoint",
            ROOT / "abl_event_exp/real_reliability_stage/reliability_net/checkpoint-best.pth",
        )
    )
    cfg.model.reliability_base_channels = 32
    cfg.model.reliability_gate_floor = 0.10
    cfg.model.reliability_frame_chunk_size = 1
    # Keep M3 -> M4 as a reliability-only change. The historical full model
    # used an additional residual post-filter, but that would confound this
    # main-table row with a second unlabelled module.
    cfg.model.residual_postfilter_kernel = 1
    cfg.model.residual_postfilter_strength = 0.0

    # This is the fairness contract shared by every row.
    cfg.lr = float(getattr(cfg, "main_table_lr", 4.0e-5))
    cfg.loss.normal_weight = float(getattr(cfg.loss, "main_table_normal_weight", 0.05))
    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.data.return_normal_gt = True
    cfg.data.return_debug_event_fields = False
    cfg.eval_every_steps = 0
    cfg.skip_final_eval = True

    output_root = Path(str(getattr(cfg, "main_table_output_root", "abl_event_exp/paper_main_table")))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg.exp_name = variant
    cfg.output_dir = str(output_root / variant)
    cfg.logdir = str(output_root / variant / "logs")
    cfg.save_dir = str(output_root)

    pretrained = str(getattr(cfg, "pretrained", "") or "")
    if pretrained in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        cfg.pretrained = str(ROOT / "ckpt/model.pt")

    if uses_multildr(variant):
        cfg.data.random_train_ldr = True
        cfg.data.eval_ldr_event_id = str(getattr(cfg.data, "main_table_eval_ldr", "ev_5"))
        cfg.data.ldr_event_id = "random"

    weights = DETAIL_LOSS if uses_detail_loss(variant) else COMMON_LOSS
    configure_mul_loss_cfg(cfg, weights=weights, exp_name=variant)
    cfg.ablation_contract = {
        "variant": variant,
        "pretrained": str(cfg.pretrained),
        "trainable_shared": ["camera_head", "depth_head", "point_head"],
        "aggregator_frozen": True,
        "event_refiner_trainable": variant != "m0_matched_rgb",
        "detail_gt": uses_detail_loss(variant),
        "multi_ldr": uses_multildr(variant),
        "frozen_reliability": variant == "m4_full_reliability",
    }
    return cfg


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg):
    variant = str(getattr(cfg, "main_table_variant", "m4_full_reliability")).lower()
    if variant not in VARIANTS:
        raise ValueError(f"main_table_variant must be one of {VARIANTS}, got {variant}")
    cfg = _prepare_cfg(cfg, variant)
    if variant == "m4_full_reliability" and not Path(cfg.model.reliability_checkpoint).is_file():
        raise FileNotFoundError(
            f"Frozen ReliabilityNet checkpoint missing: {cfg.model.reliability_checkpoint}"
        )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(cfg.output_dir) / "ablation_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(OmegaConf.to_container(cfg.ablation_contract, resolve=True), handle, indent=2)

    fe.Accelerator = UnevenBatchAccelerator
    fe.save_current_code = _safe_code_snapshot
    fe.build_event_model = build_model
    fe.configure_trainable_params = configure_trainable_params
    fe.EventSupervisedLoss = make_configured_loss(cfg)
    if uses_multildr(variant):
        fe.build_event_loader = build_random_ldr_event_loader

    print(f"[main-table] variant={variant}")
    print(f"[main-table] contract={OmegaConf.to_container(cfg.ablation_contract, resolve=True)}")
    fe.train(cfg)


if __name__ == "__main__":
    run()
