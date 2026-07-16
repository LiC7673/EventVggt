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


class ExplicitHFResidualObjective:
    def __init__(self, base, phase):
        self.base, self.phase = base, phase

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        if self.phase != "adapter":
            result.details["explicit_hf_residual"] = result.loss.new_zeros(())
            result.details["final_gt_normal_derivative"] = result.loss.new_zeros(())
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
        # Exact signed residual on event-supported interior pixels. Everywhere
        # else is a zero-update target, so event texture has no free shortcut.
        inside = F.smooth_l1_loss(pred, target, beta=.01, reduction="none")
        inside_loss = (inside * live).sum() / live.sum().clamp_min(1)
        outside = valid & ~live
        keep_loss = (pred.abs() * outside).sum() / outside.sum().clamp_min(1)
        hf_loss = inside_loss + .25 * keep_loss

        intrinsics = torch.stack([v["camera_intrinsics"] for v in views], 1).to(final).float()
        final_n = F.normalize(fe.depth_to_normals(final, intrinsics), dim=-1, eps=1.e-6)
        gt_n = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1.e-6)
        pred_dn = _normal_derivative(final_n)
        target_dn = _normal_derivative(gt_n).detach()
        dn_error = F.smooth_l1_loss(pred_dn, target_dn, beta=.01, reduction="none").mean(-1)
        dn_live = live
        dn_loss = (dn_error * dn_live).sum() / dn_live.sum().clamp_min(1)

        # HF residual is the primary refiner target; final-depth differential
        # geometry prevents a numerically good but non-integrable/noisy update.
        result.loss = result.loss + 2.0 * hf_loss + 1.0 * dn_loss
        result.details.update(
            explicit_hf_residual=hf_loss,
            explicit_hf_keep=keep_loss,
            final_gt_normal_derivative=dn_loss,
            hf_target_abs=(target.abs() * live).sum() / live.sum().clamp_min(1),
            hf_prediction_abs=(pred.abs() * live).sum() / live.sum().clamp_min(1),
            loss=result.loss,
        )
        return result


def criterion_for(args, phase):
    return ExplicitHFResidualObjective(alternating.criterion_for(args, phase), phase)


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
