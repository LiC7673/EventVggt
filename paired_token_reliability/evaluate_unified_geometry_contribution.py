"""Evaluate a unified contribution+adapter checkpoint with the held-out protocol."""

from __future__ import annotations

from pathlib import Path

import torch

import finetune_event as fe
from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
import real_reliability_stage.evaluate_stage2_heldout as evaluator
from paired_token_reliability.unified_model import UnifiedGeometryContributionModel


def build_model(checkpoint: Path, _unused_override: str | None, device: torch.device):
    raw = torch_load(checkpoint)
    if raw.get("schema") != UnifiedGeometryContributionModel.checkpoint_schema:
        raise ValueError(f"Not a unified checkpoint: {raw.get('schema')!r}")
    cfg = cfg_from_checkpoint(raw, None)
    model_cfg = cfg.model
    data_cfg = cfg.data
    training_args = raw.get("training_args", {})
    model = UnifiedGeometryContributionModel(
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
    )
    state = strip_module_prefix(fe.unwrap_state_dict(raw))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg


if __name__ == "__main__":
    evaluator.StreamVGGT = UnifiedGeometryContributionModel
    evaluator.build_model = build_model
    evaluator.main()
