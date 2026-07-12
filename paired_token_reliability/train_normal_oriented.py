"""Train the additive normal-oriented variant without changing legacy scripts.

All existing CLI flags are retained.  New architecture knobs are supplied as
OmegaConf overrides, for example ``model.event_adapter_levels=[0,1]``.
"""

from __future__ import annotations

import finetune_event as fe

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.normal_oriented_loss import NormalOrientedGeometryLoss
from paired_token_reliability.normal_oriented_model import NormalOrientedGeometryContributionModel
from paired_token_reliability import train_unified_geometry_contribution as legacy


def build_model(cfg, args, device):
    model = NormalOrientedGeometryContributionModel(
        img_size=int(cfg.model.img_size), patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        event_hidden_dim=int(getattr(cfg.model, "adapter_event_hidden_dim", 48)),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        event_num_bins=int(cfg.data.event_resize_bins),
        event_count_cmax=float(getattr(cfg.model, "event_count_cmax", 3.0)),
        event_pyramid_channels=int(getattr(cfg.model, "adapter_event_pyramid_channels", 64)),
        adapter_hidden_channels=int(getattr(cfg.model, "adapter_hidden_channels", 128)),
        contribution_channels=int(getattr(cfg.model, "contribution_channels", 32)),
        contribution_initial_value=0.95,
        contribution_use_geometry_prior=not args.no_geometry_prior,
        event_adapter_levels=tuple(getattr(cfg.model, "event_adapter_levels", [0, 1])),
        support_dilation_kernel=int(getattr(cfg.model, "support_dilation_kernel", 5)),
        enable_event_depth_residual=bool(getattr(cfg.model, "enable_event_depth_residual", False)),
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    message = model.load_state_dict(state, strict=False)
    required = [key for key in message.missing_keys if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"Base checkpoint misses required VGGT weights: {required[:10]}")
    print(f"[normal-oriented base] missing(new)={len(message.missing_keys)} unused={len(message.unexpected_keys)}")
    return model.to(device)


def criterion_for(args, phase):
    return NormalOrientedGeometryLoss(
        depth_weight=1.0, normal_weight=args.normal_weight,
        point_weight=args.point_weight, bridge_beta=args.bridge_beta,
        budget_weight=0.0 if args.no_budget else args.budget_weight,
        pair_weight=0.0 if args.no_pair_consistency else args.pair_weight,
        update_weight=args.update_weight,
        decomposition_weight=args.decomposition_weight,
        geometry_rank_weight=args.geometry_rank_weight,
        geometry_rank_margin=args.geometry_rank_margin,
        geometry_rank_threshold=args.geometry_rank_threshold,
        # The new final normal replaces the legacy independent event-normal
        # objective; these weights are represented by N and DN below.
        event_normal_weight=0.0, depth_event_normal_weight=0.0,
        depth_gradient_weight=args.depth_gradient_weight,
        depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight, grid_patch_size=args.grid_patch_size,
        normal_gradient_weight=0.5,
        depth_normal_consistency_weight=args.depth_event_normal_weight,
        depth_outside_support_weight=0.2,
        detach_normal_target=True, points_loss_type="l1",
    )


def main(argv=None):
    # Reuse data loading, DDP, visualization, alternating A/B/C, and checkpoint
    # handling, while replacing only the model/loss/schema symbols.
    legacy.build_model = build_model
    legacy.criterion_for = criterion_for
    legacy.UnifiedGeometryContributionModel = NormalOrientedGeometryContributionModel
    legacy.main(argv)


if __name__ == "__main__":
    main()
