"""One-stage dual alignment: full->geo event and LDR+event->HDR tokens."""
from __future__ import annotations

import math
import os
from pathlib import Path
import finetune_event as fe
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_dual_alignment_hdr_model import (
    DualAlignmentHDRLinearVoxelModel,
)
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as visual_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


_BASE_PREPARE_PAIR = pipeline.prepare_pair


def _exposure_ids(value):
    """Flatten collated string metadata without assuming a batch size."""
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_exposure_ids(item))
        return result
    return [str(value)]


def prepare_dual_alignment_pair(batch, device, args, _phase):
    target, reference, event, bridge = _BASE_PREPARE_PAIR(
        batch, device, args, "contribution"
    )
    if len(target) != len(reference):
        raise RuntimeError("LDR/HDR teacher view counts differ")
    for student_view, teacher_view in zip(target, reference):
        teacher_ids = {
            value.strip().lower() for value in _exposure_ids(
                teacher_view.get("ldr_event_id", "missing")
            )
        }
        if teacher_ids != {"ev_0"}:
            raise RuntimeError(
                "HDR-token teacher must contain only ev_0, but got "
                f"{sorted(teacher_ids)}. Run this route with --pair-mode anchor; "
                "the check also prevents saturation orientation from silently "
                "selecting a different exposure as teacher."
            )
        student_view["hdr_img"] = teacher_view["img"]
        student_view["hdr_ldr_event_id"] = "ev_0"
        # This route consumes E_full; E_geo is a training-only reference.
        student_view["event_source_label"] = "E_full"
    return target, reference, event, bridge


def one_stage_schedule(epochs, contribution_epochs, joint_epochs=0):
    if int(contribution_epochs) or int(joint_epochs):
        raise ValueError("dual alignment requires --epochs-b=0 --epochs-c=0")
    if int(epochs) <= 0:
        raise ValueError("dual alignment requires --epochs-a > 0")
    return ["adapter"] * int(epochs)


def _weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _hf(normal, kernel):
    channels = normal.movedim(-1, -3)
    flat = channels.flatten(0, 1)
    blur = F.avg_pool2d(flat, kernel, 1, kernel // 2)
    return (flat - blur).reshape_as(channels).movedim(-3, -1)


class DualAlignmentObjective:
    def __init__(self, base):
        self.base = base
        self.calls = 0

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        full_n = torch.stack([x["event_normal_full"] for x in output.ress], 1).float()
        geo_n = torch.stack([x["event_normal_geo"] for x in output.ress], 1).float()
        full_support = torch.stack([x["event_normal_support"] for x in output.ress], 1).bool()
        geo_support = torch.stack([x["geo_event_support"] for x in output.ress], 1).bool()
        feature_error = torch.stack([x["alignment_feature_error"] for x in output.ress], 1).float()
        reliability = torch.stack([x["event_contribution"] for x in output.ress], 1).float()
        reliability_target = torch.stack(
            [x["alignment_reliability_target"] for x in output.ress], 1
        ).float()
        full_mass = torch.stack(
            [x["full_event_mass"] for x in output.ress], 1
        ).float().detach()
        hdr_error = torch.stack(
            [x["hdr_token_alignment_error"] for x in output.ress], 1
        ).float()
        gt = result.aux["normal_gt_live"].float().detach()
        valid = result.aux["normal_valid_live"].bool()
        full_weight = (valid & full_support).float()
        geo_weight = (valid & geo_support).float()
        pair_weight = (valid & (full_support | geo_support)).float()
        full_u = F.normalize(full_n, dim=-1, eps=1e-6)
        geo_u = F.normalize(geo_n, dim=-1, eps=1e-6)
        gt_u = F.normalize(gt, dim=-1, eps=1e-6)

        geo_normal = _weighted_mean(
            1.0 - (geo_u * gt_u).sum(-1).clamp(-1, 1), geo_weight
        )
        normal_distill = _weighted_mean(
            1.0 - (full_u * geo_u.detach()).sum(-1).clamp(-1, 1), pair_weight
        )
        hf_error = 0.0
        for kernel in (3, 7):
            hf_error = hf_error + (
                _hf(full_u, kernel) - _hf(gt_u, kernel)
            ).abs().mean(-1)
        normal_hf = _weighted_mean(.5 * hf_error, full_weight)
        event_align = _weighted_mean(feature_error, pair_weight)
        # Event pixels carry the main attribution supervision.  A weak global
        # zero-event term suppresses unconstrained bright background without
        # allowing the numerous empty pixels to dominate C.
        event_support = full_support | geo_support
        confidence_weight = event_support.float() + .05 * (~event_support).float()
        confidence = _weighted_mean(
            F.smooth_l1_loss(
                reliability, reliability_target, beta=.05, reduction="none"
            ), confidence_weight,
        )
        # Match the selected event mass to the actual geo/full mass for each
        # sample.  Unlike the legacy rho=1 budget, this never forces all full
        # events to be trusted and gives C a strong anti-collapse gradient.
        mass_denominator = full_mass.flatten(1).sum(1).clamp_min(1.0e-6)
        predicted_mass_ratio = (
            full_mass * reliability
        ).flatten(1).sum(1) / mass_denominator
        target_mass_ratio = (
            full_mass * reliability_target
        ).flatten(1).sum(1) / mass_denominator
        mass_loss = F.mse_loss(predicted_mass_ratio, target_mass_ratio)
        hdr_align = hdr_error.mean()
        contribution_warmup = bool(float(
            output.ress[0]["contribution_warmup_active"].detach()
        ))
        source_weight = 2.0 if contribution_warmup else 1.0
        result.loss = result.loss + (
            1.0 * event_align + source_weight * confidence + 2.0 * mass_loss
            + 2.0 * geo_normal + .10 * normal_distill + 1.0 * normal_hf
            + 1.0 * hdr_align
        )
        result.details["event_feature_alignment"] = event_align
        result.details["event_alignment_confidence"] = confidence
        result.details["event_mass_attribution"] = mass_loss
        result.details["predicted_event_mass_ratio"] = predicted_mass_ratio.mean()
        result.details["target_event_mass_ratio"] = target_mass_ratio.mean()
        result.details["geo_event_normal"] = geo_normal
        result.details["event_normal_distill"] = normal_distill
        result.details["event_normal_hf"] = normal_hf
        result.details["hdr_token_alignment"] = hdr_align
        # Replace the disabled legacy rho=1 diagnostic with the dynamic mass
        # loss that actually participates in this route's objective.
        result.details["legacy_budget_disabled"] = result.details["budget"]
        result.details["budget"] = mass_loss
        self.calls += int(torch.is_grad_enabled())
        if torch.is_grad_enabled() and self.calls % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            warmup = int(float(output.ress[0]["hdr_warmup_active"].detach()))
            print(
                f"[dual-align@{self.calls:05d}] hdr_active={1-warmup} "
                f"Ealign={float(event_align.detach()):.5f} "
                f"HDRalign={float(hdr_align.detach()):.5f} "
                f"Ngeo={float(geo_normal.detach()):.5f} "
                f"Ndist={float(normal_distill.detach()):.5f} "
                f"C={float(_weighted_mean(reliability, pair_weight).detach()):.4f} "
                f"Ct={float(_weighted_mean(reliability_target, pair_weight).detach()):.4f} "
                f"Cmass={float(predicted_mass_ratio.mean().detach()):.4f} "
                f"Ctmass={float(target_mass_ratio.mean().detach()):.4f} "
                f"Cwarm={int(contribution_warmup)}",
                flush=True,
            )
        return result


def build_model(cfg, args, device):
    m = cfg.model
    model = DualAlignmentHDRLinearVoxelModel(
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
        normal_refine_iterations=int(getattr(m, "normal_refine_iterations", 3)),
        normal_refine_step_limit=float(getattr(m, "normal_refine_step_limit", .05)),
        point_update_scale=float(getattr(m, "point_update_scale", .10)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    message = model.load_state_dict(
        strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))), strict=False
    )
    required = [k for k in message.missing_keys if k.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(
        f"[dual alignment] event=full->geo token, exposure=LDR+event->HDR token "
        f"hdr_teacher=ev_0-only hdr_warmup={model.hdr_warmup_steps}",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    if phase != "adapter":
        raise ValueError(f"dual alignment is one-stage, got {phase}")
    model.requires_grad_(False)
    model.depth_log_scale.requires_grad_(True)
    model.event_encoder.requires_grad_(True)
    model.event_normal_decoder.requires_grad_(True)
    model.full_geo_aligner.requires_grad_(True)
    # The legacy reliability sub-head inside FullToGeoAlignment is not the
    # deployed C predictor.  C comes from ContributionNet(E_full, RGB,
    # coarse geometry), so keep the obsolete sub-head frozen.
    model.full_geo_aligner.reliability.requires_grad_(False)
    model.contribution_net.requires_grad_(True)
    model.event_token_projection.requires_grad_(True)
    model.ldr_event_hdr_aligner.requires_grad_(True)
    model.normal_depth_refiner.requires_grad_(True)
    model.point_refiner.requires_grad_(True)
    model.train()
    model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(
        "[dual alignment trainable] event_encoder+event_normal+event_aligner+ContributionNet+"
        "event_token_projection+hdr_aligner+normal_depth_refiner+point_refiner+depth_scale",
        flush=True,
    )


def optimizer_for(model, _phase, args):
    scale_id = id(model.depth_log_scale)
    event_encoder_ids = {id(p) for p in model.event_encoder.parameters()}
    fast_ids = {
        id(p) for module in (
            model.event_normal_decoder, model.full_geo_aligner,
            model.contribution_net,
            model.event_token_projection, model.ldr_event_hdr_aligner,
            model.normal_depth_refiner,
            model.point_refiner,
        ) for p in module.parameters()
    }
    regular, encoder, fast, scale = [], [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) == scale_id:
            scale.append(parameter)
        elif id(parameter) in fast_ids:
            fast.append(parameter)
        elif id(parameter) in event_encoder_ids:
            encoder.append(parameter)
        else:
            regular.append(parameter)
    groups = []
    if regular: groups.append({"params": regular, "lr": args.lr})
    if encoder: groups.append({"params": encoder, "lr": 2.0 * args.lr})
    if fast: groups.append({"params": fast, "lr": 5.0 * args.lr})
    if scale: groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def criterion_for(args, _phase):
    base = UnifiedGeometryContributionLoss(
        depth_weight=1.0, normal_weight=args.normal_weight,
        point_weight=args.point_weight, bridge_beta=args.bridge_beta,
        budget_weight=0.0, pair_weight=0.0,
        update_weight=args.update_weight, decomposition_weight=0.0,
        geometry_rank_weight=0.0,
        event_normal_weight=args.event_normal_weight,
        depth_event_normal_weight=args.depth_event_normal_weight,
        depth_gradient_weight=args.depth_gradient_weight,
        depth_curvature_weight=args.depth_curvature_weight,
        patch_grid_weight=args.patch_grid_weight,
        grid_patch_size=args.grid_patch_size, points_loss_type="l1",
    )
    return DualAlignmentObjective(base)


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    visual_base.save_visual(
        output_root, phase, epoch, batch_index, views, reference_views,
        event, bridge, output, aux,
    )
    item = output.ress[0]
    token_error = item["hdr_token_alignment_error"][0].detach().float().cpu()
    count = token_error.numel()
    grid_h = max(1, round(math.sqrt(count)))
    while grid_h > 1 and count % grid_h:
        grid_h -= 1
    grid_w = count // grid_h
    predicted_c = item["event_contribution"][0].detach().float().cpu()
    target_c = item["alignment_reliability_target"][0].detach().float().cpu()
    target_available = bool(float(
        item["alignment_reliability_target_available"].detach().cpu()
    ))
    target_title = "C target: |E_geo| / |E_full|" if target_available else "C target unavailable (inference)"
    error_title = "|C pred - C target|" if target_available else "C target error unavailable"
    confidence_error = (predicted_c - target_c).abs() if target_available else torch.zeros_like(predicted_c)
    panels = (
        (item["alignment_feature_error"][0].detach().float().cpu(), "full->geo feature error"),
        (predicted_c, "predicted reliability C"),
        (target_c, target_title),
        (confidence_error, error_title),
        (token_error.reshape(grid_h, grid_w), "LDR+event -> HDR token error"),
    )
    figure, axes = plt.subplots(1, 5, figsize=(25, 5))
    for axis, (image, title) in zip(axes, panels):
        shown = axis.imshow(image.numpy(), cmap="magma")
        axis.set_title(title); axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=.046, pad=.04)
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}_dual_alignment.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def capture_runtime_state(model):
    return {"dual_alignment_step": int(model._dual_alignment_step)}


def restore_runtime_state(model, state):
    model._dual_alignment_step = int(state.get("dual_alignment_step", 0))


def main(argv=None):
    pipeline.prepare_pair = prepare_dual_alignment_pair
    pipeline.build_alternating_phase_schedule = one_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.capture_runtime_state = capture_runtime_state
    pipeline.restore_runtime_state = restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = DualAlignmentHDRLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
