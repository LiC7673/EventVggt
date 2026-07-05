"""Controlled ablation of the full image-guided event reliability model."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import finetune_event as fe  # noqa: E402
from ablation.finetune_paper_ablation import FULL_RELIABILITY_WEIGHTS  # noqa: E402
from mul_loss_fine.image_guided_event_reliability_loss import (  # noqa: E402
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg, make_configured_loss  # noqa: E402
from multildr_token_exp import common as multi  # noqa: E402
from multildr_token_exp.token_loss import wrap_token_consistency  # noqa: E402


VARIANTS = (
    "temporal_detail_no_gate",
    "gate_no_img_supervision",
    "img_reliability_no_detail_gt",
    "full_img_reliability",
    "full_img_reliability_token_multildr",
)


def _weights(*, direct_detail: bool, image_reliability: bool):
    weights = dict(FULL_RELIABILITY_WEIGHTS)
    if not direct_detail:
        weights.update(
            {
                "detail_gt_normal_weight": 0.0,
                "detail_gt_hf_weight": 0.0,
                "detail_gt_grad_weight": 0.0,
                "detail_gt_event_boost": 0.0,
            }
        )
    if not image_reliability:
        weights.update(
            {
                "img_event_reliability_weight": 0.0,
                "img_event_reject_weight": 0.0,
            }
        )
    return weights


def _variant_flags(variant: str):
    return {
        "gate": variant != "temporal_detail_no_gate",
        "image_reliability": variant
        in {
            "img_reliability_no_detail_gt",
            "full_img_reliability",
            "full_img_reliability_token_multildr",
        },
        "direct_detail": variant != "img_reliability_no_detail_gt",
        "token_multildr": variant == "full_img_reliability_token_multildr",
    }


def _safe_snapshot(outdir: str):
    destination = Path(outdir) / "code_snapshot"
    relative_files = (
        "full_img_core_ablation/finetune_core_ablation.py",
        "full_img_core_ablation/run_core_ablation_gpus_234567.sh",
        "multildr_token_exp/common.py",
        "multildr_token_exp/token_model.py",
        "multildr_token_exp/token_loss.py",
        "ablation/finetune_paper_ablation.py",
        "mul_loss_fine/image_guided_event_reliability_loss.py",
        "mul_loss_fine/event_supported_mv_loss.py",
        "eventvggt/models/streamvggt_temporal_detail.py",
        "eventvggt/datasets/my_event_dataset.py",
        "finetune_event.py",
        "config/finetune_event.yaml",
    )
    for relative in relative_files:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def _configure_trainable(model, cfg):
    multi.configure_trainable(model, cfg)
    if str(cfg.core_ablation_variant) == "temporal_detail_no_gate":
        for name, parameter in model.named_parameters():
            if "event_detail_refiner.reliability_head" in name:
                parameter.requires_grad = False


def _prepare(cfg, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"core_ablation_variant must be one of {VARIANTS}, got {variant}")
    flags = _variant_flags(variant)
    loader_strategy = "paired_token_full" if flags["token_multildr"] else "random_ldr_full"

    # Reuse the exact historical Full model/data defaults and then change only
    # the ablated switches below.
    cfg = multi.prepare_cfg(cfg, loader_strategy)
    OmegaConf.set_struct(cfg, False)
    cfg.core_ablation_variant = variant
    cfg.multildr_strategy = loader_strategy
    cfg.model.event_reliability_gate_enabled = bool(flags["gate"])

    output_root = Path(
        str(getattr(cfg, "core_ablation_output_root", ROOT / "abl_event_exp/full_img_core_ablation"))
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg.exp_name = variant
    cfg.save_dir = str(output_root)
    cfg.output_dir = str(output_root / variant)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")

    configure_mul_loss_cfg(
        cfg,
        weights=_weights(
            direct_detail=flags["direct_detail"],
            image_reliability=flags["image_reliability"],
        ),
        exp_name=variant,
    )
    cfg.core_ablation_contract = {
        "variant": variant,
        "event_residual": True,
        "reliability_gate": flags["gate"],
        "image_guided_reliability_supervision": flags["image_reliability"],
        "direct_gt_detail_supervision": flags["direct_detail"],
        "exposure_strategy": "paired_token" if flags["token_multildr"] else "random_ldr",
        "training_scene_count": 12,
    }
    return cfg, flags


@hydra.main(
    version_base=None,
    config_path=str(ROOT / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg):
    variant = str(getattr(cfg, "core_ablation_variant", "full_img_reliability"))
    cfg, flags = _prepare(cfg, variant)

    if flags["image_reliability"]:
        loss_class = make_configured_image_guided_event_reliability_loss(cfg)
    else:
        loss_class = make_configured_loss(cfg)
    if flags["token_multildr"]:
        loss_class = wrap_token_consistency(loss_class, cfg)

    fe.Accelerator = multi.UnevenBatchAccelerator
    fe.save_current_code = _safe_snapshot
    fe.build_event_loader = multi.build_strategy_loader
    fe.configure_trainable_params = _configure_trainable
    fe.EventSupervisedLoss = loss_class
    if flags["token_multildr"]:
        fe.build_event_model = multi._build_token_model

    print(f"[core ablation] {cfg.core_ablation_contract}")
    fe.train(cfg)


if __name__ == "__main__":
    run()
