"""Strict cur_event training with explicit HF depth-residual supervision."""
from __future__ import annotations

import sys

import finetune_event as fe
import torch
import torch.nn.functional as F

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_alternating_detail_first as alternating
from paired_token_reliability import train_linear_voxel_alternating_detail_first_fixed as fixed
from paired_token_reliability import train_linear_voxel_alternating_detail_first_fixed_cur_event as cur
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_cur_event_hf_residual_model import (
    CurEventHFResidualModel,
)
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss


def build_model(cfg, args, device):
    m = cfg.model
    model = CurEventHFResidualModel(
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
    print("[HF-RESIDUAL] Geo pixel refiner active immediately; explicit target=HF(logGT-logBase)", flush=True)
    return model.to(device)


def _masked_blur(value, valid, kernel=9):
    # Normalized filtering prevents background zeros leaking into object edges.
    pad = kernel // 2
    weight = F.avg_pool2d(valid.float(), kernel, stride=1, padding=pad)
    mean = F.avg_pool2d(value * valid.float(), kernel, stride=1, padding=pad)
    return mean / weight.clamp_min(1.0e-6)


def _normal_derivative(normal):
    dx = torch.zeros_like(normal); dy = torch.zeros_like(normal)
    dx[:, :, :, :-1] = normal[:, :, :, 1:] - normal[:, :, :, :-1]
    dy[:, :, :-1, :] = normal[:, :, 1:, :] - normal[:, :, :-1, :]
    return torch.cat((dx, dy), dim=-1)


def _balanced_loss(error, magnitude, valid, strong_quantile=.70, floor=1.e-4):
    values = magnitude[valid]
    threshold = (
        torch.quantile(values.detach(), strong_quantile)
        if values.numel() else magnitude.new_tensor(float("inf"))
    )
    strong = valid & (magnitude >= threshold) & (magnitude > floor)
    weak = valid & ~strong
    strong_loss = (error * strong).sum() / strong.sum().clamp_min(1)
    weak_loss = (error * weak).sum() / weak.sum().clamp_min(1)
    return strong_loss + .10 * weak_loss, strong, weak


class CleanCurEventGeometryObjective:
    """One non-conflicting objective for each link in the geometry chain."""

    def __init__(self, base, phase, args):
        self.base, self.phase, self.args = base, phase, args

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        zero = result.loss.new_zeros(())

        # These are the only objectives used to learn cur_event->Geo,
        # reliability, and LDR+event->HDR. They do not supervise geometry
        # derivatives and therefore do not duplicate the three losses below.
        feature_error = torch.stack(
            [x["alignment_feature_error"] for x in output.ress], 1
        ).float()
        hdr_align = torch.stack(
            [x["hdr_token_alignment_error"] for x in output.ress], 1
        ).float().mean()
        valid = result.aux["normal_valid_live"].bool()
        full_support = torch.stack(
            [x["event_normal_support"] for x in output.ress], 1
        ).bool()
        geo_support = torch.stack(
            [x["geo_event_support"] for x in output.ress], 1
        ).bool()
        pair = valid & (full_support | geo_support)
        event_align = (feature_error * pair).sum() / pair.sum().clamp_min(1)

        reliability = torch.stack(
            [x["event_contribution"] for x in output.ress], 1
        ).float()
        refine_c = torch.stack(
            [x["normal_fusion_gate"] for x in output.ress], 1
        ).float()
        c_target = torch.stack(
            [x["alignment_reliability_target"] for x in output.ress], 1
        ).float().detach()
        c_weight = (full_support | geo_support).float() + .05 * (~(full_support | geo_support)).float()
        c_den = c_weight.sum().clamp_min(1)
        c_fusion_loss = (
            F.smooth_l1_loss(reliability, c_target, beta=.05, reduction="none") * c_weight
        ).sum() / c_den
        c_refine_loss = (
            F.smooth_l1_loss(refine_c, c_target, beta=.05, reduction="none") * c_weight
        ).sum() / c_den
        full_mass = torch.stack([x["full_event_mass"] for x in output.ress], 1).float().detach()
        mass_den = full_mass.flatten(1).sum(1).clamp_min(1.e-6)
        predicted_mass = (full_mass * reliability).flatten(1).sum(1) / mass_den
        target_mass = (full_mass * c_target).flatten(1).sum(1) / mass_den
        mass_loss = F.mse_loss(predicted_mass, target_mass)

        if self.phase == "contribution":
            result.loss = result.loss + event_align + hdr_align + c_fusion_loss + c_refine_loss + 2.0 * mass_loss
            result.details.update(
                event_feature_alignment=event_align,
                hdr_token_alignment=hdr_align,
                event_alignment_confidence=c_fusion_loss,
                refinement_alignment_confidence=c_refine_loss,
                event_mass_attribution=mass_loss,
                event_normal=zero,
                depth_event_normal=zero,
                explicit_hf_residual=zero,
                final_gt_normal_derivative=zero,
                loss=result.loss,
            )
            return result

        base = torch.stack([x["depth_hdr_base"][..., 0] for x in output.ress], 1).float()
        final = torch.stack([x["depth"][..., 0] for x in output.ress], 1).float()
        pred = torch.stack([x["pixel_refiner_bounded_update"] for x in output.ress], 1).float()
        gt = torch.stack([v["depthmap"] for v in views], 1).to(base).float()
        valid = result.aux["normal_valid_live"].bool()
        support = torch.stack([x["event_normal_support"] for x in output.ress], 1).bool()
        support = F.max_pool2d(
            support.flatten(0, 1).float().unsqueeze(1), 3, stride=1, padding=1
        )[:, 0].reshape_as(support).bool()
        live = valid & support & (base > 1.e-6) & (gt > 1.e-6)

        log_residual = torch.log(gt.clamp_min(1.e-6)) - torch.log(base.clamp_min(1.e-6))
        bv = log_residual.flatten(0, 1).unsqueeze(1)
        vv = valid.flatten(0, 1).unsqueeze(1)
        low = _masked_blur(bv, vv, 9)[:, 0].reshape_as(log_residual)
        target = (log_residual - low).detach()
        # (1) Refiner -> signed GT high-frequency log-depth residual. Strong
        # target pixels are isolated before averaging, so flat pixels cannot
        # win by driving the whole prediction to zero.
        residual_error = F.smooth_l1_loss(pred, target, beta=.01, reduction="none")
        hf_loss, hf_strong, _ = _balanced_loss(
            residual_error, target.abs(), live, strong_quantile=.70, floor=1.e-4
        )
        outside = valid & ~live
        keep_loss = (pred.abs() * outside).sum() / outside.sum().clamp_min(1)
        hf_loss = hf_loss + .25 * keep_loss

        intrinsics = torch.stack([v["camera_intrinsics"] for v in views], 1).to(final).float()
        final_n = F.normalize(fe.depth_to_normals(final, intrinsics), dim=-1, eps=1.e-6)
        gt_n = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1.e-6)
        # (2) Event derivative -> GT derivative. Only this one loss supervises
        # the event derivative head; no RecallFirst/PixelHF wrapper remains.
        event_dn = torch.stack(
            [x["event_normal_derivative_full"] for x in output.ress], 1
        ).float()
        event_dn = event_dn.reshape(*event_dn.shape[:4], 6)
        target_dn = _normal_derivative(gt_n).detach()
        target_dn_mag = target_dn.norm(dim=-1)
        event_error = F.smooth_l1_loss(
            event_dn, target_dn, beta=.01, reduction="none"
        ).mean(-1)
        event_loss, _, _ = _balanced_loss(
            event_error, target_dn_mag, live, strong_quantile=.70, floor=1.e-4
        )

        # (3) Final-depth derivative -> GT derivative. It never follows the
        # event prediction, avoiding a moving/noisy intermediate teacher.
        pred_dn = _normal_derivative(final_n)
        dn_error = F.smooth_l1_loss(pred_dn, target_dn, beta=.01, reduction="none").mean(-1)
        dn_loss, dn_strong, _ = _balanced_loss(
            dn_error, target_dn_mag, live, strong_quantile=.70, floor=1.e-4
        )

        result.loss = result.loss + hdr_align + self.args.event_normal_weight * event_loss + 2.0 * hf_loss + dn_loss
        result.details.update(
            event_normal=event_loss,
            depth_event_normal=dn_loss,
            explicit_hf_residual=hf_loss,
            explicit_hf_keep=keep_loss,
            final_gt_normal_derivative=dn_loss,
            hdr_token_alignment=hdr_align,
            hf_target_abs=(target.abs() * hf_strong).sum() / hf_strong.sum().clamp_min(1),
            hf_prediction_abs=(pred.abs() * hf_strong).sum() / hf_strong.sum().clamp_min(1),
            hf_strong_fraction=hf_strong.float().sum() / live.float().sum().clamp_min(1),
            dn_strong_fraction=dn_strong.float().sum() / live.float().sum().clamp_min(1),
            loss=result.loss,
        )
        return result


def criterion_for(args, phase):
    base = UnifiedGeometryContributionLoss(
        depth_weight=1.0,
        normal_weight=args.normal_weight,
        point_weight=args.point_weight,
        bridge_beta=args.bridge_beta,
        budget_weight=0.0,
        pair_weight=0.0,
        update_weight=0.0,
        decomposition_weight=0.0,
        geometry_rank_weight=0.0,
        event_normal_weight=0.0,
        depth_event_normal_weight=0.0,
        # Do not stack legacy derivative/grid losses under the explicit HF and
        # final-normal-derivative objectives.
        depth_gradient_weight=0.0,
        depth_curvature_weight=0.0,
        patch_grid_weight=0.0,
        grid_patch_size=args.grid_patch_size,
        points_loss_type="l1",
    )
    return CleanCurEventGeometryObjective(base, phase, args)


def _force(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    prefixes = ("data.event_source_mode=", "data.decomposition_supervision=",
                "data.decomposition_geo_branch=")
    values = [x for x in values if not x.startswith(prefixes)]
    values += ["data.event_source_mode=cur_event", "data.decomposition_supervision=true",
               "data.decomposition_geo_branch=geometry_motion"]
    return values


def main(argv=None):
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = cur.prepare_pair
    pipeline.build_alternating_phase_schedule = alternating.schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = cur.configure_phase
    pipeline.optimizer_for = fixed.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = alternating.final_base.save_visual
    pipeline.capture_runtime_state = alternating.capture_runtime_state
    pipeline.restore_runtime_state = alternating.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = CurEventHFResidualModel
    pipeline.main(_force(argv))


if __name__ == "__main__":
    main()
