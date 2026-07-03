"""Shared model and optimization policy for the clean main-table ablation."""

from __future__ import annotations

from eventvggt.models.streamvggt_causal_temporal_detail import (
    StreamVGGT as CausalTemporalDetailVGGT,
)
from eventvggt.models.streamvggt_pretrained_reliability_detail import (
    StreamVGGT as PretrainedReliabilityVGGT,
)
from streamvggt.models.streamvggt import StreamVGGT as RGBStreamVGGT


VARIANTS = (
    "a0_rgb_only",
    "a1_direct_event",
    "a2_wo_reliability",
    "a3_wo_multildr",
    "a4_wo_detail",
    "a5_full",
)


VARIANT_MODULES = {
    "a0_rgb_only": {"event": False, "detail": False, "multildr": False, "reliability": False},
    "a1_direct_event": {"event": True, "detail": False, "multildr": False, "reliability": False},
    "a2_wo_reliability": {"event": True, "detail": True, "multildr": True, "reliability": False},
    "a3_wo_multildr": {"event": True, "detail": True, "multildr": False, "reliability": True},
    "a4_wo_detail": {"event": True, "detail": False, "multildr": True, "reliability": True},
    "a5_full": {"event": True, "detail": True, "multildr": True, "reliability": True},
}


def is_event_variant(variant: str) -> bool:
    return bool(VARIANT_MODULES[str(variant).lower()]["event"])


def uses_detail_loss(variant: str) -> bool:
    return bool(VARIANT_MODULES[str(variant).lower()]["detail"])


def uses_multildr(variant: str) -> bool:
    return bool(VARIANT_MODULES[str(variant).lower()]["multildr"])


def uses_reliability(variant: str) -> bool:
    return bool(VARIANT_MODULES[str(variant).lower()]["reliability"])


def build_model(cfg):
    variant = str(cfg.main_table_variant).lower()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown main-table variant: {variant}")
    common = {
        "img_size": int(cfg.model.img_size),
        "patch_size": int(cfg.model.patch_size),
        "embed_dim": int(cfg.model.embed_dim),
    }
    if not is_event_variant(variant):
        return RGBStreamVGGT(**common)

    event_common = {
        **common,
        "event_hidden_dim": int(cfg.model.main_event_hidden_dim),
        "head_frames_chunk_size": int(cfg.model.head_frames_chunk_size),
        "event_num_bins": int(cfg.model.event_num_bins),
        "event_count_cmax": float(cfg.model.event_count_cmax),
        "residual_scale": float(cfg.model.refiner_residual_scale),
        "residual_highpass_kernel": int(cfg.model.event_delta_highpass_kernel),
        "residual_patch_zero_mean": bool(cfg.model.event_delta_patch_zero_mean),
        "residual_patch_size": int(cfg.model.event_delta_patch_size),
        "residual_abs_limit": float(cfg.model.event_delta_abs_limit),
        "refine_points": True,
        "use_checkpoint": bool(cfg.model.refiner_use_checkpoint),
        "forward_batch_chunk": int(getattr(cfg.model, "exposure_forward_batch_chunk", 1)),
    }
    support = {
        "causal_support_threshold": float(cfg.model.causal_support_threshold),
        "causal_support_dilate_kernel": int(cfg.model.causal_support_dilate_kernel),
        "causal_support_blur_kernel": int(cfg.model.causal_support_blur_kernel),
    }
    if uses_reliability(variant):
        return PretrainedReliabilityVGGT(
            **event_common,
            reliability_checkpoint=str(cfg.model.reliability_checkpoint),
            reliability_base_channels=int(cfg.model.reliability_base_channels),
            reliability_gate_floor=float(cfg.model.reliability_gate_floor),
            reliability_frame_chunk_size=int(cfg.model.reliability_frame_chunk_size),
            reliability_rgb_input_range="minus_one_one",
            residual_postfilter_kernel=int(cfg.model.residual_postfilter_kernel),
            residual_postfilter_strength=float(cfg.model.residual_postfilter_strength),
            causal_output_gate=True,
            **support,
        )
    return CausalTemporalDetailVGGT(
        **event_common,
        support_threshold=support["causal_support_threshold"],
        support_dilate_kernel=support["causal_support_dilate_kernel"],
        support_blur_kernel=support["causal_support_blur_kernel"],
    )


def configure_trainable_params(model, _cfg) -> None:
    """Identical shared trainable policy for every row in the main table."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module_name in ("camera_head", "depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(True)

    for name, parameter in model.named_parameters():
        if "event_detail_refiner" not in name:
            continue
        if "reliability_net" in name or ".reliability_head." in name:
            continue
        parameter.requires_grad = True

    refiner = getattr(model, "event_detail_refiner", None)
    reliability_net = getattr(refiner, "reliability_net", None)
    if reliability_net is not None:
        reliability_net.requires_grad_(False)
        reliability_net.eval()


def trainable_parameter_summary(model):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return count, names
