"""Train V10 with direct event normal-derivative prediction."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10
from paired_token_reliability.linear_voxel_dual_alignment_hdr_derivative_model import (
    EventNormalDerivativeV10Model,
)
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss
from paired_token_reliability.common import strip_module_prefix, torch_load


def build_model(cfg, args, device):
    m = cfg.model
    model = EventNormalDerivativeV10Model(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 5)),
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
    compatible = {
        key: value for key, value in state.items()
        if key in own and own[key].shape == value.shape
        and not key.startswith("event_normal_decoder.")
    }
    loaded = model.load_state_dict(compatible, strict=False)
    required = [key for key in loaded.missing_keys if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print("[direct derivative V10] initialized new 6-channel dN head; absolute-normal head skipped", flush=True)
    return model.to(device)


def normal_derivative(normal):
    dx, dy = torch.zeros_like(normal), torch.zeros_like(normal)
    dx[:, :, :, :-1, :] = normal[:, :, :, 1:, :] - normal[:, :, :, :-1, :]
    dy[:, :, :-1, :, :] = normal[:, :, 1:, :, :] - normal[:, :, :-1, :, :]
    return torch.stack((dx, dy), dim=-2)


def derivative_loss(pred, target, valid, support):
    b, v, h, w = support.shape
    local = F.max_pool2d(
        support.float().reshape(b * v, 1, h, w), 3, 1, 1
    ).reshape(b, v, h, w) > 0
    mask = (valid & local).unsqueeze(-1).float()
    error = F.smooth_l1_loss(pred, target, beta=.02, reduction="none").mean(-1)
    vector = (error * mask).sum() / (mask.sum().clamp_min(1) * 2.0)
    pred_mag, target_mag = pred.norm(dim=-1), target.norm(dim=-1)
    magnitude = ((pred_mag - target_mag).abs() * mask).sum() / (mask.sum().clamp_min(1) * 2.0)
    return vector + .25 * magnitude, local


class DerivativeObjective:
    def __init__(self, base, args):
        self.base, self.args = base, args

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        full = torch.stack([x["event_normal_derivative_full"] for x in output.ress], 1).float()
        geo = torch.stack([x["event_normal_derivative_geo"] for x in output.ress], 1).float()
        gt_n = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6)
        final_n = F.normalize(result.aux["normal_pred_live"].float(), dim=-1, eps=1e-6)
        target = normal_derivative(gt_n)
        final_d = normal_derivative(final_n)
        valid = result.aux["normal_valid_live"].bool()
        full_support = torch.stack([x["event_normal_support"] for x in output.ress], 1).bool()
        geo_support = torch.stack([x["geo_event_support"] for x in output.ress], 1).bool()
        full_loss, local = derivative_loss(full, target, valid, full_support)
        geo_loss, _ = derivative_loss(geo, target, valid, geo_support)
        depth_loss, _ = derivative_loss(final_d, full.detach(), valid, full_support)
        distill, _ = derivative_loss(full, geo.detach(), valid, full_support | geo_support)

        feature_error = torch.stack([x["alignment_feature_error"] for x in output.ress], 1).float()
        align_weight = (valid & (full_support | geo_support)).float()
        event_align = (feature_error * align_weight).sum() / align_weight.sum().clamp_min(1)
        hdr_align = torch.stack([x["hdr_token_alignment_error"] for x in output.ress], 1).float().mean()
        reliability = torch.stack([x["event_contribution"] for x in output.ress], 1).float()
        reliability_target = torch.stack([x["alignment_reliability_target"] for x in output.ress], 1).float()
        full_mass = torch.stack([x["full_event_mass"] for x in output.ress], 1).float().detach()
        c_weight = (full_support | geo_support).float() + .05 * (~(full_support | geo_support)).float()
        c_error = F.smooth_l1_loss(reliability, reliability_target, beta=.05, reduction="none")
        confidence = (c_error * c_weight).sum() / c_weight.sum().clamp_min(1)

        # Preserve the V10 anti-collapse constraint.  Pixel-wise confidence
        # alone is dominated by empty/low-ratio locations and allowed C -> 0.
        mass_denominator = full_mass.flatten(1).sum(1).clamp_min(1.0e-6)
        predicted_mass_ratio = (full_mass * reliability).flatten(1).sum(1) / mass_denominator
        target_mass_ratio = (full_mass * reliability_target).flatten(1).sum(1) / mass_denominator
        mass_loss = F.mse_loss(predicted_mass_ratio, target_mass_ratio)
        contribution_warmup = bool(float(
            output.ress[0]["contribution_warmup_active"].detach()
        ))
        source_weight = 2.0 if contribution_warmup else 1.0

        result.loss = result.loss + event_align + hdr_align + source_weight * confidence + 2.0 * mass_loss + (
            self.args.event_normal_weight * (full_loss + geo_loss)
            + .10 * distill
            + self.args.depth_event_normal_weight * depth_loss
        )
        result.details["event_normal"] = full_loss
        result.details["depth_event_normal"] = depth_loss
        result.details["geo_event_normal_derivative"] = geo_loss
        result.details["event_derivative_distill"] = distill
        result.details["event_feature_alignment"] = event_align
        result.details["event_alignment_confidence"] = confidence
        result.details["event_mass_attribution"] = mass_loss
        result.details["predicted_event_mass_ratio"] = predicted_mass_ratio.mean()
        result.details["target_event_mass_ratio"] = target_mass_ratio.mean()
        result.details["legacy_budget_disabled"] = result.details["budget"]
        result.details["budget"] = mass_loss
        result.details["hdr_token_alignment"] = hdr_align
        result.details["loss"] = result.loss
        result.aux["event_normal_local_support"] = local
        return result


def criterion_for(args, _phase):
    base = UnifiedGeometryContributionLoss(
        depth_weight=1.0, normal_weight=args.normal_weight,
        point_weight=args.point_weight, bridge_beta=args.bridge_beta,
        budget_weight=0.0, pair_weight=0.0, update_weight=args.update_weight,
        decomposition_weight=0.0, geometry_rank_weight=0.0,
        event_normal_weight=0.0, depth_event_normal_weight=0.0,
        depth_gradient_weight=args.depth_gradient_weight,
        depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight,
        grid_patch_size=args.grid_patch_size, points_loss_type="l1",
    )
    return DerivativeObjective(base, args)


def configure_phase(model, phase, train_heads_a=False):
    v10.configure_phase(model, phase, train_heads_a)
    model.normal_depth_refiner.requires_grad_(False)
    print("[direct derivative V10] absolute-normal depth refiner disabled", flush=True)


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    v10.save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                    event, bridge, output, aux)
    pred = output.ress[0]["event_normal_derivative_full"][0].detach().float().cpu()
    target = normal_derivative(F.normalize(aux["normal_gt_live"].float(), dim=-1))[0, 0].detach().cpu()
    pred_mag, target_mag = pred.square().sum((-1, -2)).sqrt(), target.square().sum((-1, -2)).sqrt()
    panels = ((pred_mag, "pred |normal derivative|"), (target_mag, "GT |normal derivative|"),
              ((pred_mag-target_mag).abs(), "derivative magnitude error"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image.numpy(), cmap="magma"); axis.set_title(title); axis.axis("off")
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}_direct_derivative.png"
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None):
    pipeline.prepare_pair = v10.prepare_dual_alignment_pair
    pipeline.build_alternating_phase_schedule = v10.one_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = v10.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.capture_runtime_state = v10.capture_runtime_state
    pipeline.restore_runtime_state = v10.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = EventNormalDerivativeV10Model
    pipeline.main(argv)


if __name__ == "__main__":
    main()
