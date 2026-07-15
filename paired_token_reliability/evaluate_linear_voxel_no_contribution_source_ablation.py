"""Evaluate fixed-C oracle-geo or raw-full event-source ablations."""
from pathlib import Path

import finetune_event as fe

from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_no_contribution_source_ablation_model import (
    NoContributionSourceAblationModel,
)
from paired_token_reliability import evaluate_linear_voxel_multiscale as visual_base
import real_reliability_stage.evaluate_stage2_heldout as protocol


_EVAL_SOURCE = "full"
_BASE_CONDITION_VIEWS = protocol.condition_views


def _source_views(views):
    if _EVAL_SOURCE != "geo":
        return views
    selected = []
    for index, view in enumerate(views):
        geo = view.get("geometry_event_voxel")
        if geo is None:
            raise RuntimeError(
                f"E_geo evaluation requires geometry_event_voxel (missing view {index})"
            )
        current = dict(view)
        current["event_voxel"] = geo
        current["event_source_preselected"] = True
        selected.append(current)
    return selected


def condition_source_views(views, condition):
    # Select the ablation source first, then apply zero/flip/time conditions.
    return _BASE_CONDITION_VIEWS(_source_views(views), condition)


def save_source_visuals(args, views, output, depth, depth_gt, valid, batch_idx):
    return visual_base.save_weight_visuals(
        args, _source_views(views), output, depth, depth_gt, valid, batch_idx
    )


def build_model(checkpoint: Path, _override, device):
    global _EVAL_SOURCE
    raw = torch_load(checkpoint)
    expected = NoContributionSourceAblationModel.checkpoint_schema
    if raw.get("schema") != expected:
        raise ValueError(f"checkpoint schema={raw.get('schema')!r}, expected {expected!r}")
    cfg = cfg_from_checkpoint(raw, None); m = cfg.model
    source = str(getattr(m, "event_source_ablation", "full")).strip().lower()
    _EVAL_SOURCE = source
    model = NoContributionSourceAblationModel(
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
        event_source=source,
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32,
    )
    model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(raw)), strict=True)
    return model.to(device).eval(), cfg


def main():
    import paired_token_reliability.evaluate_unified_all_exposures as driver
    protocol.condition_views = condition_source_views
    protocol.save_full_event_visuals = save_source_visuals
    driver.build_model = build_model
    driver.main()


if __name__ == "__main__":
    main()
