"""Corrected alternating detail-first training entry."""
from __future__ import annotations

import finetune_event as fe
import torch

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as old
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_alternating_detail_first_fixed_model import (
    AlternatingDetailFirstFixedModel,
)


def build_model(cfg, args, device):
    m = cfg.model
    model = AlternatingDetailFirstFixedModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        event_count_cmax=float(getattr(m, "event_count_cmax", 3.0)),
        pixel_refiner_hidden=int(getattr(m, "pixel_refiner_hidden", 64)),
        pixel_refine_log_limit=float(getattr(m, "pixel_refine_log_limit", .30)),
        pixel_refiner_delay=int(getattr(m, "pixel_refiner_delay", 500)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .0015)),
        alignment_confidence_tau=.10, hdr_token_bottleneck=256,
        hdr_warmup_steps=0, normal_refine_iterations=1, normal_refine_step_limit=.05,
        c_delay_steps=int(getattr(m, "c_delay_steps", 1000)),
        c_transition_steps=int(getattr(m, "c_transition_steps", 1000)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    loaded = model.load_state_dict(compatible, strict=False)
    required = [k for k in loaded.missing_keys if k.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"missing frozen VGGT weights: {required[:10]}")
    print("[FIXED] Geo derivative/refiner bypass FullAligner; real C_refine; pixel delay=500", flush=True)
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.set_confidence_stage("geo")
        for module in (model.event_encoder, model.event_normal_decoder,
                       model.event_token_projection, model.ldr_event_hdr_aligner,
                       model.pixel_depth_refiner):
            module.requires_grad_(True)
        label = "direct E_geo -> dN/pixel-refiner; both C=1"
    elif phase == "contribution":
        model.set_confidence_stage("full")
        # Full-to-Geo alignment must learn only after the direct Geo teacher is stable.
        model.full_geo_aligner.learned.requires_grad_(True)
        model.contribution_net.learned.requires_grad_(True)
        model.normal_fusion_gate.learned.requires_grad_(True)
        label = "E_full; train FullAligner+C_fusion+C_refine; geometry heads frozen"
    else:
        raise ValueError(phase)
    model.train(); model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(f"[fixed/{phase}] {label}", flush=True)


def optimizer_for(model, phase, args):
    if phase == "adapter":
        modules = (model.event_encoder, model.event_normal_decoder,
                   model.event_token_projection, model.ldr_event_hdr_aligner,
                   model.pixel_depth_refiner)
        multiplier = 3.0
    else:
        modules = (model.full_geo_aligner.learned, model.contribution_net.learned,
                   model.normal_fusion_gate.learned)
        multiplier = 2.0
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"phase {phase} has no trainable parameters")
    return torch.optim.AdamW(params, lr=multiplier * args.lr,
                             weight_decay=args.weight_decay, betas=(.9, .95))


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = old.prepare_pair
    pipeline.build_alternating_phase_schedule = old.schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = old.criterion_for
    pipeline.save_visual = old.final_base.save_visual
    pipeline.capture_runtime_state = old.capture_runtime_state
    pipeline.restore_runtime_state = old.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = AlternatingDetailFirstFixedModel
    pipeline.main(argv)


if __name__ == "__main__": main()
