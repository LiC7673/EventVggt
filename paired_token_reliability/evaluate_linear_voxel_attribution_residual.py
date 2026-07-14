"""Evaluate RGB + full-event attribution/residual checkpoints without teachers."""
from pathlib import Path

import finetune_event as fe

from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_attribution_residual_model import (
    AttributionResidualLinearVoxelModel,
)
from paired_token_reliability import evaluate_linear_voxel_multiscale as visual_base
import real_reliability_stage.evaluate_stage2_heldout as protocol


def build_model(checkpoint: Path, _override, device):
    raw = torch_load(checkpoint)
    expected = AttributionResidualLinearVoxelModel.checkpoint_schema
    if raw.get("schema") != expected:
        raise ValueError(f"checkpoint schema={raw.get('schema')!r}, expected {expected!r}")
    cfg = cfg_from_checkpoint(raw, None); m = cfg.model
    model = AttributionResidualLinearVoxelModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        hdr_token_bottleneck=int(getattr(m, "hdr_token_bottleneck", 256)),
        hdr_warmup_steps=int(getattr(m, "hdr_warmup_steps", 1000)),
        normal_refine_iterations=int(getattr(m, "normal_refine_iterations", 3)),
        normal_refine_step_limit=float(getattr(m, "normal_refine_step_limit", .05)),
        point_update_scale=float(getattr(m, "point_update_scale", .10)),
        geometry_projection_dim=int(getattr(m, "geometry_projection_dim", 256)),
        saturation_threshold=float(getattr(m, "saturation_threshold", .98)),
        ablate_event_attribution=bool(getattr(m, "ablate_event_attribution", False)),
        ablate_missing_residual=bool(getattr(m, "ablate_missing_residual", False)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32,
    )
    model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(raw)), strict=True)
    return model.to(device).eval(), cfg


def main():
    import paired_token_reliability.evaluate_unified_all_exposures as driver
    protocol.save_full_event_visuals = visual_base.save_weight_visuals
    driver.build_model = build_model
    driver.main()


if __name__ == "__main__":
    main()
