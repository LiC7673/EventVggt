"""Load and evaluate a normal-oriented checkpoint with the unified protocol."""

from __future__ import annotations

from pathlib import Path

import torch

import finetune_event as fe
from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
from paired_token_reliability.normal_oriented_model import NormalOrientedGeometryContributionModel


def build_model(checkpoint: Path, _unused_override: str | None, device: torch.device):
    raw = torch_load(checkpoint)
    expected = NormalOrientedGeometryContributionModel.checkpoint_schema
    if raw.get("schema") != expected:
        raise ValueError(f"Expected {expected!r}, got {raw.get('schema')!r}")
    cfg = cfg_from_checkpoint(raw, None)
    model_cfg, data_cfg = cfg.model, cfg.data
    training_args = raw.get("training_args", {})
    model = NormalOrientedGeometryContributionModel(
        img_size=int(getattr(model_cfg, "img_size", 518)),
        patch_size=int(getattr(model_cfg, "patch_size", 14)),
        embed_dim=int(getattr(model_cfg, "embed_dim", 1024)),
        event_hidden_dim=int(getattr(model_cfg, "adapter_event_hidden_dim", 48)),
        head_frames_chunk_size=int(getattr(model_cfg, "head_frames_chunk_size", 2)),
        event_num_bins=int(getattr(data_cfg, "event_resize_bins", 10)),
        event_count_cmax=float(getattr(model_cfg, "event_count_cmax", 3.0)),
        event_pyramid_channels=int(getattr(model_cfg, "adapter_event_pyramid_channels", 64)),
        adapter_hidden_channels=int(getattr(model_cfg, "adapter_hidden_channels", 128)),
        contribution_channels=int(getattr(model_cfg, "contribution_channels", 32)),
        contribution_initial_value=0.95,
        contribution_use_geometry_prior=not bool(training_args.get("no_geometry_prior", False)),
        event_adapter_levels=tuple(getattr(model_cfg, "event_adapter_levels", [0, 1])),
        support_dilation_kernel=int(getattr(model_cfg, "support_dilation_kernel", 5)),
        enable_event_depth_residual=False,
    )
    model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(raw)), strict=True)
    return model.to(device).eval(), cfg


def main():
    # The all-exposure driver uses its module-global loader. Replace only that
    # loader and retain the established held-out metrics/visualization protocol.
    import paired_token_reliability.evaluate_unified_all_exposures as driver
    driver.build_model = build_model
    driver.main()


if __name__ == "__main__":
    main()
