"""Train event attribution + missing exposure-geometry residual recovery."""
from __future__ import annotations

import math
import os
from pathlib import Path

import finetune_event as fe
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_attribution_residual_model import (
    AttributionResidualLinearVoxelModel,
)
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as visual_base
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as dual_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def _weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class AttributionResidualObjective:
    def __init__(self, base, utility_margin=.02, gain_margin=.005):
        self.base = base
        self.utility_margin = float(utility_margin)
        self.gain_margin = float(gain_margin)
        self.calls = 0

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        pred_c = torch.stack(
            [item["event_contribution"] for item in output.ress], dim=1
        ).float()
        target_c = torch.stack(
            [item["event_contribution_target"] for item in output.ress], dim=1
        ).float().detach()
        selected_n = torch.stack(
            [item["event_normal_selected"] for item in output.ress], dim=1
        ).float()
        rejected_n = torch.stack(
            [item["event_normal_rejected"] for item in output.ress], dim=1
        ).float()
        pred_residual = torch.stack(
            [item["predicted_missing_geometry_residual"] for item in output.ress], dim=1
        ).float()
        target_residual = torch.stack(
            [item["target_missing_geometry_residual"] for item in output.ress], dim=1
        ).float().detach()
        sat = torch.stack(
            [item["saturation_patch_mask"] for item in output.ress], dim=1
        ).float().clamp(0, 1)
        valid = result.aux["valid_live"].bool()
        normal_valid = result.aux["normal_valid_live"].bool()
        normal_gt = F.normalize(
            result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6
        )
        support = torch.stack(
            [item["event_normal_support"] for item in output.ress], dim=1
        ).bool()
        ablate_attr = bool(float(output.ress[0]["ablate_event_attribution"].detach()))
        ablate_residual = bool(float(output.ress[0]["ablate_missing_residual"].detach()))

        if ablate_attr:
            attribution = pred_c.sum() * 0.0
            utility = attribution
            selected_error = attribution
            rejected_error = attribution
        else:
            attribution = _weighted_mean(
                F.smooth_l1_loss(pred_c, target_c, beta=.05, reduction="none"),
                valid.float(),
            )
            selected_u = F.normalize(selected_n, dim=-1, eps=1e-6)
            rejected_u = F.normalize(rejected_n, dim=-1, eps=1e-6)
            utility_weight = (normal_valid & support).float()
            selected_error = _weighted_mean(
                1.0 - (selected_u * normal_gt).sum(-1).clamp(-1, 1),
                utility_weight,
            )
            rejected_error = _weighted_mean(
                1.0 - (rejected_u * normal_gt).sum(-1).clamp(-1, 1),
                utility_weight,
            )
            utility = F.relu(
                selected_error - rejected_error + self.utility_margin
            )

        residual_error = (pred_residual - target_residual).abs().mean(-1)
        residual_magnitude = pred_residual.abs().mean(-1)
        if ablate_residual:
            residual = pred_residual.sum() * 0.0
            keep = residual
        else:
            residual = _weighted_mean(residual_error, sat)
            keep = _weighted_mean(residual_magnitude, 1.0 - sat)

        # Geometry gain is evaluated in output space, not feature space.
        depth_final = result.aux["depth_pred_live"].float()
        depth_gt = result.aux["depth_gt"].to(depth_final).float()
        depth_coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], dim=1
        ).float()
        depth_final_error = _weighted_mean((depth_final - depth_gt).abs(), valid.float())
        depth_coarse_error = _weighted_mean(
            (depth_coarse - depth_gt).abs(), valid.float()
        ).detach()

        normal_final = F.normalize(
            result.aux["normal_pred_live"].float(), dim=-1, eps=1e-6
        )
        normal_coarse = F.normalize(torch.stack(
            [item["coarse_normal"] for item in output.ress], dim=1
        ).float(), dim=-1, eps=1e-6)
        nweight = normal_valid.float()
        normal_final_error = _weighted_mean(
            1.0 - (normal_final * normal_gt).sum(-1).clamp(-1, 1), nweight
        )
        normal_coarse_error = _weighted_mean(
            1.0 - (normal_coarse * normal_gt).sum(-1).clamp(-1, 1), nweight
        ).detach()

        point_final = torch.stack(
            [item["pts3d_in_other_view"] for item in output.ress], dim=1
        ).float()
        point_coarse = torch.stack(
            [item["pts3d_coarse"] for item in output.ress], dim=1
        ).float()
        point_gt = result.aux["points_gt"].to(point_final).float()
        pose_pred = torch.stack(
            [item["camera_pose"] for item in output.ress], dim=1
        ).float()
        pose_gt = fe.stack_view_field(views, "camera_pose").to(point_final).float()
        image_hw = depth_gt.shape[-2:]
        pred_c2w, _ = fe.pose_encoding_to_c2w(pose_pred, image_size_hw=image_hw)
        _, first_frame_alignment = fe.align_c2w_by_first_frame(pred_c2w, pose_gt)
        point_final = fe.transform_world_points(
            point_final, first_frame_alignment.detach()
        )
        point_coarse = fe.transform_world_points(
            point_coarse, first_frame_alignment.detach()
        )
        point_scale = point_gt.norm(dim=-1).clamp_min(1e-3)
        point_final_error = _weighted_mean(
            (point_final - point_gt).norm(dim=-1) / point_scale, valid.float()
        )
        point_coarse_error = _weighted_mean(
            (point_coarse - point_gt).norm(dim=-1) / point_scale, valid.float()
        ).detach()

        final_geometry = (
            depth_final_error + .25 * normal_final_error + .10 * point_final_error
        )
        coarse_geometry = (
            depth_coarse_error + .25 * normal_coarse_error + .10 * point_coarse_error
        )
        gain = F.relu(final_geometry - coarse_geometry + self.gain_margin)

        result.loss = result.loss + (
            .50 * attribution + .25 * utility
            + 1.00 * residual + .25 * keep + .25 * gain
        )
        result.details.update({
            "source_attribution": attribution,
            "event_utility": utility,
            "selected_geometry_error": selected_error,
            "rejected_geometry_error": rejected_error,
            "missing_geometry_residual": residual,
            "non_saturated_keep": keep,
            "geometry_gain": gain,
            "final_geometry_score": final_geometry,
            "coarse_geometry_score": coarse_geometry,
        })
        self.calls += int(torch.is_grad_enabled())
        if torch.is_grad_enabled() and self.calls % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[attr-res@{self.calls:05d}] abC={int(ablate_attr)} abR={int(ablate_residual)} "
                f"C={float(pred_c.mean().detach()):.4f} Cgt={float(target_c.mean()):.4f} "
                f"attr={float(attribution.detach()):.5f} util={float(utility.detach()):.5f} "
                f"res={float(residual.detach()):.5f} keep={float(keep.detach()):.5f} "
                f"gain={float(gain.detach()):.5f}",
                flush=True,
            )
        return result


def build_model(cfg, args, device):
    m = cfg.model
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
        contribution_channels=32, contribution_initial_value=.70,
    )
    message = model.load_state_dict(
        strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))), strict=False
    )
    required = [key for key in message.missing_keys if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(
        "[attribution residual] no full/geo token L1, no complete ev0 token L1; "
        f"ablate_C={model.ablate_event_attribution} "
        f"ablate_residual={model.ablate_missing_residual}",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    if phase != "adapter":
        raise ValueError(f"one-stage attribution training expected adapter, got {phase}")
    model.requires_grad_(False)
    model.depth_log_scale.requires_grad_(True)
    for module in (
        model.event_encoder, model.event_normal_decoder, model.contribution_net,
        model.event_token_projection, model.geometry_residual_adapter,
        model.normal_depth_refiner, model.point_refiner,
    ):
        module.requires_grad_(True)
    model.train()
    model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(
        "[trainable] C attribution + missing-geometry residual + normal/depth/point refiners",
        flush=True,
    )


def optimizer_for(model, _phase, args):
    scale_id = id(model.depth_log_scale)
    encoder_ids = {id(parameter) for parameter in model.event_encoder.parameters()}
    fast_modules = (
        model.event_normal_decoder, model.contribution_net,
        model.event_token_projection, model.geometry_residual_adapter,
        model.normal_depth_refiner, model.point_refiner,
    )
    fast_ids = {id(parameter) for module in fast_modules for parameter in module.parameters()}
    encoder, fast, scale, regular = [], [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) == scale_id:
            scale.append(parameter)
        elif id(parameter) in encoder_ids:
            encoder.append(parameter)
        elif id(parameter) in fast_ids:
            fast.append(parameter)
        else:
            regular.append(parameter)
    groups = []
    if regular: groups.append({"params": regular, "lr": args.lr})
    if encoder: groups.append({"params": encoder, "lr": 2.0 * args.lr})
    if fast: groups.append({"params": fast, "lr": 5.0 * args.lr})
    if scale: groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def criterion_for(args, _phase):
    geometry = UnifiedGeometryContributionLoss(
        depth_weight=1.0, normal_weight=args.normal_weight,
        point_weight=args.point_weight, bridge_beta=args.bridge_beta,
        budget_weight=0.0, pair_weight=0.0, update_weight=args.update_weight,
        decomposition_weight=0.0, geometry_rank_weight=0.0,
        event_normal_weight=args.event_normal_weight,
        depth_event_normal_weight=args.depth_event_normal_weight,
        depth_gradient_weight=args.depth_gradient_weight,
        depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight,
        grid_patch_size=args.grid_patch_size, points_loss_type="l1",
    )
    return AttributionResidualObjective(geometry)


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    visual_base.save_visual(
        output_root, phase, epoch, batch_index, views, reference_views,
        event, bridge, output, aux,
    )
    item = output.ress[0]
    pred_c = item["event_contribution"][0].detach().float().cpu()
    target_c = item["event_contribution_target"][0].detach().float().cpu()
    sat = item["saturation_patch_mask"][0].detach().float().cpu()
    pred_res = item["predicted_missing_geometry_residual"][0].detach().float().abs().mean(-1).cpu()
    target_res = item["target_missing_geometry_residual"][0].detach().float().abs().mean(-1).cpu()
    patch_count = sat.numel()
    grid_h = max(1, round(math.sqrt(patch_count)))
    while grid_h > 1 and patch_count % grid_h:
        grid_h -= 1
    grid_w = patch_count // grid_h
    sat = sat.reshape(grid_h, grid_w)
    pred_res = pred_res.reshape(grid_h, grid_w)
    target_res = target_res.reshape(grid_h, grid_w)
    panels = (
        (pred_c, "predicted C"), (target_c, "C_gt=geo/full mass"),
        ((pred_c-target_c).abs(), "|C-C_gt|"),
        (sat, "saturation patch mask"),
        (pred_res, "pred missing geometry residual"),
        (target_res, "target missing geometry residual"),
    )
    figure, axes = plt.subplots(1, 6, figsize=(30, 5))
    for axis, (image, title) in zip(axes, panels):
        shown = axis.imshow(image.numpy(), cmap="magma", vmin=0)
        axis.set_title(title); axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=.046, pad=.04)
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}_attribution_residual.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def main(argv=None):
    pipeline.prepare_pair = dual_base.prepare_dual_alignment_pair
    pipeline.build_alternating_phase_schedule = dual_base.one_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.capture_runtime_state = dual_base.capture_runtime_state
    pipeline.restore_runtime_state = dual_base.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = AttributionResidualLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
