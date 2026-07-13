"""Train the additive linear-voxel route with metric depth calibration."""
import torch
import finetune_event as fe

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_calibrated_model import (
    CalibratedLinearVoxelMultiscalePixelModel,
)
from paired_token_reliability import train_linear_voxel_multiscale as base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def build_model(cfg, args, device):
    model = CalibratedLinearVoxelMultiscalePixelModel(
        img_size=int(cfg.model.img_size), patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(cfg.model, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(cfg.model, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(cfg.model, "depth_update_scale", .03)),
        event_decay_tau=float(getattr(cfg.model, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(cfg.model, "depth_log_scale_limit", 2.0)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    if bool(getattr(cfg.model, "force_full_contribution", False)) and not model.force_full_contribution:
        from paired_token_reliability.linear_voxel_multiscale_model import _ForcedFullContribution
        model.contribution_net = _ForcedFullContribution(model.contribution_net)
        model.force_full_contribution = True
    message = model.load_state_dict(
        strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))), strict=False
    )
    required = [k for k in message.missing_keys if k.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(
        f"[calibrated pixel base] scale={float(model.metric_depth_scale):.4f} "
        f"new={len(message.missing_keys)} unused={len(message.unexpected_keys)}",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    base.configure_phase(model, phase, _train_heads_a)
    if phase in {"adapter", "joint"}:
        model.depth_log_scale.requires_grad_(True)
        assert model.depth_log_scale.requires_grad
    else:
        model.depth_log_scale.requires_grad_(False)


def optimizer_for(model, phase, args):
    # A scalar starting at 1 needs a larger LR than convolution weights to
    # remove a gross global scale mismatch within the A warm-up.
    scale = [model.depth_log_scale] if model.depth_log_scale.requires_grad else []
    scale_id = id(model.depth_log_scale)
    regular = [p for p in model.parameters() if p.requires_grad and id(p) != scale_id]
    groups = []
    if regular:
        groups.append({"params": regular, "lr": args.lr})
    if scale:
        groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = base.criterion_for
    pipeline.save_visual = base.save_visual
    pipeline.UnifiedGeometryContributionModel = CalibratedLinearVoxelMultiscalePixelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
