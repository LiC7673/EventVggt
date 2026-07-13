"""Evaluator for dense scale-warmup linear-voxel checkpoints."""
from pathlib import Path
import finetune_event as fe

from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_conditioned_dense_scale_warmup_model import (
    ConditionedDenseScaleWarmupLinearVoxelModel,
)
from paired_token_reliability import evaluate_linear_voxel_multiscale as base
import real_reliability_stage.evaluate_stage2_heldout as protocol


def build_model(checkpoint: Path, _override, device):
    raw = torch_load(checkpoint)
    expected = ConditionedDenseScaleWarmupLinearVoxelModel.checkpoint_schema
    if raw.get("schema") != expected:
        raise ValueError(f"checkpoint schema={raw.get('schema')!r}, expected {expected!r}")
    cfg = cfg_from_checkpoint(raw, None); m = cfg.model
    model = ConditionedDenseScaleWarmupLinearVoxelModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)), voxel_bins=5,
        pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        event_dc_limit=float(getattr(m, "event_dc_limit", .50)),
        event_residual_target_limit=float(getattr(m, "event_residual_target_limit", .50)),
        scale_warmup_steps=int(getattr(m, "scale_warmup_steps", 1000)),
        event_min_pixel_mass=float(getattr(m, "event_min_pixel_mass", .10)),
        force_full_contribution=bool(getattr(m, "force_full_contribution", False)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32,
    )
    model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(raw)), strict=True)
    return model.to(device).eval(), cfg


def main():
    import paired_token_reliability.evaluate_unified_all_exposures as driver
    protocol.save_full_event_visuals = base.save_weight_visuals
    driver.build_model = build_model
    driver.main()


if __name__ == "__main__":
    main()
