"""Train the independent five-bin signed multi-scale pixel model."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import finetune_event as fe
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.signed_multiscale_model import SignedMultiscalePixelModel
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def build_model(cfg, args, device):
    model = SignedMultiscalePixelModel(
        img_size=int(cfg.model.img_size), patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim), head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        signed_event_bins=5, pixel_hidden=int(getattr(cfg.model, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(cfg.model, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(cfg.model, "depth_update_scale", .03)),
        event_decay_tau=float(getattr(cfg.model, "event_decay_tau", .003)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.95,
        contribution_use_geometry_prior=not args.no_geometry_prior,
    )
    message = model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))), strict=False)
    required = [k for k in message.missing_keys if k.startswith(("aggregator.", "camera_head."))]
    if required: raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(f"[signed pixel base] new={len(message.missing_keys)} unused={len(message.unexpected_keys)}")
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.event_encoder.requires_grad_(True)
        model.event_normal_decoder.requires_grad_(True)
        model.depth_local_head.requires_grad_(True)
    elif phase == "contribution":
        model.contribution_net.requires_grad_(True)
    elif phase == "joint":
        model.contribution_net.requires_grad_(True)
        model.event_encoder.requires_grad_(True)
        model.event_normal_decoder.requires_grad_(True)
        model.depth_local_head.requires_grad_(True)
    else: raise ValueError(phase)
    model.train(); model.aggregator.eval(); model.camera_head.eval(); model.depth_head.eval(); model.point_head.eval()


def criterion_for(args, _phase):
    return UnifiedGeometryContributionLoss(
        depth_weight=1., normal_weight=args.normal_weight, point_weight=args.point_weight,
        bridge_beta=args.bridge_beta, budget_weight=0. if args.no_budget else args.budget_weight,
        pair_weight=0. if args.no_pair_consistency else args.pair_weight, update_weight=args.update_weight,
        decomposition_weight=args.decomposition_weight, geometry_rank_weight=args.geometry_rank_weight,
        geometry_rank_margin=args.geometry_rank_margin, geometry_rank_threshold=args.geometry_rank_threshold,
        event_normal_weight=args.event_normal_weight,
        depth_event_normal_weight=args.depth_event_normal_weight,
        depth_gradient_weight=args.depth_gradient_weight, depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight, grid_patch_size=args.grid_patch_size, points_loss_type="l1")


_pipeline_save_visual = pipeline.save_visual


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    _pipeline_save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                          event, bridge, output, aux)
    decayed = output.ress[0]["signed_event"][0].detach().float().cpu()
    weights = output.ress[0]["temporal_decay_weights"][0].detach().float().cpu()
    if decayed.shape[0] % 2 == 0:
        bins = decayed.shape[0] // 2
        signed_projection = decayed[:bins].sum(0) - decayed[bins:].sum(0)
        mass_projection = decayed.sum(0)
    else:
        signed_projection = decayed.sum(0)
        mass_projection = decayed.abs().sum(0)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    limit = max(float(signed_projection.abs().max()), 1e-6)
    axes[0].imshow(signed_projection, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0].set_title("time-decayed signed voxel sum")
    axes[1].imshow(mass_projection, cmap="magma")
    axes[1].set_title("time-decayed event mass")
    for axis in axes: axis.axis("off")
    figure.suptitle("bin weights=" + ", ".join(f"{float(x):.3f}" for x in weights))
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}_temporal_decay.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.UnifiedGeometryContributionModel = SignedMultiscalePixelModel
    pipeline.main(argv)


if __name__ == "__main__": main()
