"""Held-out causal evaluation for contribution-guided geometry adapters."""

from __future__ import annotations

from pathlib import Path

import torch

import finetune_event as fe
from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
import real_reliability_stage.evaluate_stage2_heldout as evaluator
from stage2_geometry_adapter.model import StreamVGGT


ROOT = Path(__file__).resolve().parents[1]


def build_model(checkpoint: Path, stage1_override: str | None, device: torch.device):
    raw_checkpoint = torch_load(checkpoint)
    cfg = cfg_from_checkpoint(raw_checkpoint, None)
    model_cfg = cfg.model
    stage1_path = stage1_override or str(model_cfg.stage1_contribution_checkpoint)
    stage1_path = Path(stage1_path).expanduser()
    if not stage1_path.is_absolute():
        stage1_path = ROOT / stage1_path
    model = StreamVGGT(
        img_size=int(getattr(model_cfg, "img_size", 518)),
        patch_size=int(getattr(model_cfg, "patch_size", 14)),
        embed_dim=int(getattr(model_cfg, "embed_dim", 1024)),
        event_hidden_dim=int(getattr(model_cfg, "adapter_event_hidden_dim", 48)),
        head_frames_chunk_size=int(getattr(model_cfg, "head_frames_chunk_size", 2)),
        event_num_bins=int(getattr(model_cfg, "event_num_bins", 10)),
        event_count_cmax=float(getattr(model_cfg, "event_count_cmax", 3.0)),
        stage1_checkpoint=str(stage1_path.resolve()),
        event_pyramid_channels=int(getattr(model_cfg, "adapter_event_pyramid_channels", 64)),
        adapter_hidden_channels=int(getattr(model_cfg, "adapter_hidden_channels", 128)),
    )
    state = strip_module_prefix(fe.unwrap_state_dict(raw_checkpoint))
    message = model.load_state_dict(state, strict=False)
    adapter_missing = [key for key in message.missing_keys if "adapter" in key or "event_encoder" in key]
    if adapter_missing:
        raise RuntimeError(f"Stage-2 checkpoint is missing adapter weights: {adapter_missing[:10]}")
    print(
        f"[load adapter Stage2] missing={len(message.missing_keys)} "
        f"unexpected={len(message.unexpected_keys)} stage1={stage1_path}"
    )
    model.to(device).eval()
    return model, cfg


if __name__ == "__main__":
    evaluator.StreamVGGT = StreamVGGT
    evaluator.build_model = build_model
    evaluator.main()
