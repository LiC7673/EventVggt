"""Pixel-balanced, multi-scale event normal-derivative training for V11."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr_derivative as derivative
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_dual_alignment_hdr_pixel_hf_model import (
    PixelHighFrequencyDerivativeV10Model,
)
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss


def build_model(cfg, args, device):
    m = cfg.model
    model = PixelHighFrequencyDerivativeV10Model(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        alignment_confidence_tau=float(getattr(m, "alignment_confidence_tau", .10)),
        hdr_token_bottleneck=int(getattr(m, "hdr_token_bottleneck", 256)),
        hdr_warmup_steps=int(getattr(m, "hdr_warmup_steps", 1000)),
        normal_refine_iterations=1,
        normal_refine_step_limit=float(getattr(m, "normal_refine_step_limit", .05)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    compatible = {k: value for k, value in state.items()
                  if k in own and own[k].shape == value.shape
                  and not k.startswith("event_normal_decoder.")}
    loaded = model.load_state_dict(compatible, strict=False)
    required = [k for k in loaded.missing_keys if k.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print("[pixel-HF V11] new full-resolution 6-channel derivative head", flush=True)
    return model.to(device)


def normal_derivative_at(normal, step):
    dx, dy = torch.zeros_like(normal), torch.zeros_like(normal)
    dx[:, :, :, :-step] = (normal[:, :, :, step:] - normal[:, :, :, :-step]) / step
    dy[:, :, :-step] = (normal[:, :, step:] - normal[:, :, :-step]) / step
    return torch.stack((dx, dy), dim=-2)


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    # Per-pixel tensor layout is [d/dx,d/dy] x [nx,ny,nz].
    components = value.shape[-2] * value.shape[-1]
    return (value * mask).sum() / (mask.sum().clamp_min(1) * components)


def balanced_hf_loss(pred, target, valid, support, edge_threshold=.01):
    # Only one-pixel dilation: do not turn sparse edges into broad flat regions.
    local = valid & support.bool()
    target_mag = target.square().sum((-1, -2)).sqrt()
    edge = local & (target_mag > edge_threshold)
    flat = local & ~edge

    per_component = F.smooth_l1_loss(pred, target, beta=.01, reduction="none")
    edge_loss = masked_mean(per_component, edge)
    flat_loss = masked_mean(per_component, flat)

    pred_vec = pred.flatten(-2)
    target_vec = target.flatten(-2)
    direction = 1.0 - F.cosine_similarity(pred_vec, target_vec, dim=-1, eps=1e-6)
    direction_loss = (direction * edge.float()).sum() / edge.float().sum().clamp_min(1)

    pred_mag = pred.square().sum((-1, -2)).sqrt()
    magnitude_loss = ((pred_mag - target_mag).abs() * edge.float()).sum() / edge.float().sum().clamp_min(1)
    # Edge and flat terms are normalized independently. Thin edges therefore
    # cannot disappear merely because flat pixels are more numerous.
    total = edge_loss + .10 * flat_loss + .25 * direction_loss + .25 * magnitude_loss
    stats = {
        "hf_edge_loss": edge_loss.detach(),
        "hf_flat_loss": flat_loss.detach(),
        "hf_direction_loss": direction_loss.detach(),
        "hf_edge_ratio": edge.float().sum().detach() / local.float().sum().clamp_min(1),
        "hf_pred_magnitude": (pred_mag * edge.float()).sum().detach() / edge.float().sum().clamp_min(1),
        "hf_target_magnitude": (target_mag * edge.float()).sum().detach() / edge.float().sum().clamp_min(1),
    }
    return total, local, stats


class PixelHFObjective(derivative.DerivativeObjective):
    def __call__(self, output, views, *args, **kwargs):
        # Run only the common geometry objective here; replace V10 derivative
        # averaging with edge-balanced pixel and two-scale supervision.
        result = self.base(output, views, *args, **kwargs)
        full = torch.stack([x["event_normal_derivative_full"] for x in output.ress], 1).float()
        geo = torch.stack([x["event_normal_derivative_geo"] for x in output.ress], 1).float()
        gt_n = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6)
        final_n = F.normalize(result.aux["normal_pred_live"].float(), dim=-1, eps=1e-6)
        valid = result.aux["normal_valid_live"].bool()
        full_support = torch.stack([x["event_normal_support"] for x in output.ress], 1).bool()
        geo_support = torch.stack([x["geo_event_support"] for x in output.ress], 1).bool()

        target1 = normal_derivative_at(gt_n, 1)
        full_loss, local, stats = balanced_hf_loss(full, target1, valid, full_support)
        geo_loss, _, _ = balanced_hf_loss(geo, target1, valid, geo_support)
        depth1 = normal_derivative_at(final_n, 1)
        depth_loss, _, _ = balanced_hf_loss(depth1, full.detach(), valid, full_support)

        # Two-pixel derivatives stabilize wider structures without pooling the prediction.
        target2 = normal_derivative_at(gt_n, 2)
        pred2 = .5 * (full + torch.roll(full, shifts=-1, dims=3))
        scale2_loss, _, _ = balanced_hf_loss(pred2, target2, valid, full_support, edge_threshold=.0075)
        distill, _, _ = balanced_hf_loss(full, geo.detach(), valid, full_support | geo_support)

        feature_error = torch.stack([x["alignment_feature_error"] for x in output.ress], 1).float()
        align_weight = (valid & (full_support | geo_support)).float()
        event_align = (feature_error * align_weight).sum() / align_weight.sum().clamp_min(1)
        hdr_align = torch.stack([x["hdr_token_alignment_error"] for x in output.ress], 1).float().mean()
        reliability = torch.stack([x["event_contribution"] for x in output.ress], 1).float()
        reliability_target = torch.stack([x["alignment_reliability_target"] for x in output.ress], 1).float()
        full_mass = torch.stack([x["full_event_mass"] for x in output.ress], 1).float().detach()
        c_weight = (full_support | geo_support).float() + .05 * (~(full_support | geo_support)).float()
        confidence = (F.smooth_l1_loss(reliability, reliability_target, beta=.05, reduction="none") * c_weight).sum() / c_weight.sum().clamp_min(1)

        mass_denominator = full_mass.flatten(1).sum(1).clamp_min(1.0e-6)
        predicted_mass_ratio = (full_mass * reliability).flatten(1).sum(1) / mass_denominator
        target_mass_ratio = (full_mass * reliability_target).flatten(1).sum(1) / mass_denominator
        mass_loss = F.mse_loss(predicted_mass_ratio, target_mass_ratio)
        contribution_warmup = bool(float(
            output.ress[0]["contribution_warmup_active"].detach()
        ))
        source_weight = 2.0 if contribution_warmup else 1.0

        # Stage 1 (0-1k): learn event derivatives only. Stage 2 (1k-2k):
        # progressively transfer them to depth. Stage 3: full coupling.
        train_step = int(float(output.ress[0]["pixel_hf_train_step"].detach()))
        depth_coupling = max(0.0, min(1.0, (train_step - 1000) / 1000.0))
        effective_depth_weight = self.args.depth_event_normal_weight * depth_coupling

        result.loss = result.loss + event_align + hdr_align + source_weight * confidence + 2.0 * mass_loss + self.args.event_normal_weight * (
            full_loss + geo_loss + .50 * scale2_loss + .10 * distill
        ) + effective_depth_weight * depth_loss
        result.details.update({
            "event_normal": full_loss, "depth_event_normal": depth_loss,
            "geo_event_normal_derivative": geo_loss, "event_derivative_multiscale": scale2_loss,
            "event_derivative_distill": distill, "event_feature_alignment": event_align,
            "event_alignment_confidence": confidence, "hdr_token_alignment": hdr_align,
            "event_mass_attribution": mass_loss,
            "predicted_event_mass_ratio": predicted_mass_ratio.mean(),
            "target_event_mass_ratio": target_mass_ratio.mean(),
            "depth_event_coupling": result.loss.new_tensor(depth_coupling),
            "effective_depth_event_weight": result.loss.new_tensor(effective_depth_weight),
            **stats, "loss": result.loss,
        })
        result.details["legacy_budget_disabled"] = result.details["budget"]
        result.details["budget"] = mass_loss
        result.aux["event_normal_local_support"] = local
        return result


def criterion_for(args, _phase):
    base = UnifiedGeometryContributionLoss(
        depth_weight=1.0, normal_weight=args.normal_weight, point_weight=args.point_weight,
        bridge_beta=args.bridge_beta, budget_weight=0.0, pair_weight=0.0,
        update_weight=args.update_weight, decomposition_weight=0.0,
        geometry_rank_weight=0.0, event_normal_weight=0.0,
        depth_event_normal_weight=0.0, depth_gradient_weight=args.depth_gradient_weight,
        depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight, grid_patch_size=args.grid_patch_size,
        points_loss_type="l1",
    )
    return PixelHFObjective(base, args)


def main(argv=None):
    pipeline.prepare_pair = v10.prepare_dual_alignment_pair
    pipeline.build_alternating_phase_schedule = v10.one_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = derivative.configure_phase
    pipeline.optimizer_for = v10.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = derivative.save_visual
    pipeline.capture_runtime_state = v10.capture_runtime_state
    pipeline.restore_runtime_state = v10.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = PixelHighFrequencyDerivativeV10Model
    pipeline.main(argv)


if __name__ == "__main__":
    main()
