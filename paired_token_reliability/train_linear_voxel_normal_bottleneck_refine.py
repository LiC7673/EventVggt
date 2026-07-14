"""Train the event-normal bottleneck without a direct event-to-depth path."""
from __future__ import annotations

import os
import finetune_event as fe
import torch
import torch.nn.functional as F

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_normal_bottleneck_refine_model import (
    NormalBottleneckRefineLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_conditioned_dense_scale_warmup as dense
from paired_token_reliability import train_linear_voxel_conditioned_soft_dc as softdc
from paired_token_reliability import train_linear_voxel_conditioned_confidence_refine as visual_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def _weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _high_frequency(normal, kernel):
    channels = normal.movedim(-1, -3)
    flat = channels.flatten(0, 1)
    blur = F.avg_pool2d(flat, kernel, stride=1, padding=kernel // 2)
    return (flat - blur).reshape_as(channels).movedim(-3, -1)


class NormalBottleneckObjective:
    def __init__(self, base):
        self.base = base

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        step = dense._GLOBAL_DENSE_STEP
        bottleneck_active = bool(
            float(output.ress[0]["normal_bottleneck_active"].detach())
        )
        # DenseScheduledObjective is inherited for the final stage. During the
        # normal-only interval its depth residual has deliberately been held at
        # zero, so remove that diagnostic term from the scalar loss instead of
        # reporting an unavoidable loss with no trainable prediction path.
        if not bottleneck_active and "dense_depth_residual" in result.details:
            result.loss = result.loss - (
                result.details["detail_schedule_weight"]
                * result.details["dense_depth_residual"]
            )
        # Scale is aligned for the first 1k; normal supervision then reaches
        # full strength over the following 1k normal-only steps.
        normal_w = dense._cosine_ramp(step, 1000, 1000, 1.0)
        residual_w = dense._cosine_ramp(step, 1000, 1000, 1.0)
        hf_w = dense._cosine_ramp(step, 1000, 1000, 2.0)
        confidence_w = dense._cosine_ramp(step, 1000, 1000, .10)

        pred = torch.stack(
            [item["normal_refine_target"] for item in output.ress], dim=1
        ).float()
        learned_confidence = torch.stack(
            [item["learned_normal_confidence"] for item in output.ress], dim=1
        ).float()
        coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], dim=1
        ).float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(coarse).float()
        coarse_normal = fe.depth_to_normals(coarse, intrinsics).float()
        gt = result.aux["normal_gt_live"].float().detach()
        valid = result.aux["normal_valid_live"].bool()
        support = torch.stack(
            [item["event_normal_support"] for item in output.ress], dim=1
        ).bool()
        weight = (valid & support).float()

        pred_u = F.normalize(pred, dim=-1, eps=1e-6)
        gt_u = F.normalize(gt, dim=-1, eps=1e-6)
        coarse_u = F.normalize(coarse_normal.detach(), dim=-1, eps=1e-6)
        cosine = 1.0 - (pred_u * gt_u).sum(dim=-1).clamp(-1, 1)
        normal_loss = _weighted_mean(cosine, weight)

        pred_residual = pred_u - coarse_u
        target_residual = gt_u - coarse_u
        residual_error = F.smooth_l1_loss(
            pred_residual, target_residual, beta=.02, reduction="none"
        ).mean(dim=-1)
        residual_loss = _weighted_mean(residual_error, weight)

        hf_error = 0.0
        for kernel in (3, 7):
            hf_error = hf_error + (
                _high_frequency(pred_u, kernel) - _high_frequency(gt_u, kernel)
            ).abs().mean(dim=-1)
        hf_loss = _weighted_mean(hf_error * .5, weight)

        coarse_error = 1.0 - (coarse_u * gt_u).sum(dim=-1).clamp(-1, 1)
        confidence_target = (coarse_error / .10).clamp(0.0, 1.0)
        confidence_error = F.binary_cross_entropy(
            learned_confidence.clamp(1e-5, 1.0 - 1e-5),
            confidence_target, reduction="none",
        )
        confidence_loss = _weighted_mean(confidence_error, weight)
        result.loss = result.loss + (
            normal_w * normal_loss
            + residual_w * residual_loss
            + hf_w * hf_loss
            + confidence_w * confidence_loss
        )
        result.details["normal_residual"] = normal_loss
        result.details["normal_delta_direct"] = residual_loss
        result.details["normal_hf"] = hf_loss
        result.details["normal_confidence"] = confidence_loss
        if torch.is_grad_enabled() and step % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[normal-bottleneck@{step:05d}] active={float(bottleneck_active):.0f} "
                f"N={normal_w:.3f} R={residual_w:.3f} HF={hf_w:.3f} "
                f"Conf={confidence_w:.3f}", flush=True,
            )
        return result


def build_model(cfg, args, device):
    m = cfg.model
    model = NormalBottleneckRefineLinearVoxelModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        event_dc_limit=float(getattr(m, "event_dc_limit", .50)),
        event_residual_target_limit=float(getattr(m, "event_residual_target_limit", .50)),
        scale_warmup_steps=int(getattr(m, "scale_warmup_steps", 1000)),
        normal_bottleneck_warmup_steps=int(
            getattr(m, "normal_bottleneck_warmup_steps", 2000)
        ),
        event_min_pixel_mass=float(getattr(m, "event_min_pixel_mass", .10)),
        normal_refine_iterations=int(getattr(m, "normal_refine_iterations", 3)),
        normal_refine_step_limit=float(getattr(m, "normal_refine_step_limit", .05)),
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
        f"[normal bottleneck] scale_warmup={model.scale_warmup_steps} "
        f"normal_only_until={model.normal_bottleneck_warmup_steps} "
        f"iterations={model.normal_refine_iterations} raw_event_to_depth=OFF",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, train_heads_a=False):
    softdc.conditioned_base.configure_phase(model, phase, train_heads_a)
    # Disable both old shortcuts. Event features must first become normals.
    model.event_normal_decoder.requires_grad_(False)
    model.conditioned_depth_head.requires_grad_(False)
    enabled = phase in {"adapter", "joint"}
    model.event_normal_delta_head.requires_grad_(enabled)
    model.normal_confidence_head.requires_grad_(enabled)
    model.normal_bottleneck_refiner.requires_grad_(enabled)


def optimizer_for(model, phase, args):
    scale_id = id(model.depth_log_scale)
    fast_ids = {
        id(parameter)
        for module in (
            model.event_normal_delta_head,
            model.normal_confidence_head,
            model.normal_bottleneck_refiner,
        )
        for parameter in module.parameters()
    }
    encoder_ids = {id(parameter) for parameter in model.event_encoder.parameters()}
    scale, fast, encoder, regular = [], [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        pid = id(parameter)
        if pid == scale_id:
            scale.append(parameter)
        elif pid in fast_ids:
            fast.append(parameter)
        elif pid in encoder_ids:
            encoder.append(parameter)
        else:
            regular.append(parameter)
    groups = []
    if regular:
        groups.append({"params": regular, "lr": args.lr})
    if encoder:
        groups.append({"params": encoder, "lr": 2.0 * args.lr})
    if fast:
        groups.append({"params": fast, "lr": 5.0 * args.lr})
    if scale:
        groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def criterion_for(args, phase):
    base = dense.criterion_for(args, phase)
    return NormalBottleneckObjective(base) if phase in {"adapter", "joint"} else base


def capture_runtime_state(model):
    return {
        "dense_global_step": int(dense._GLOBAL_DENSE_STEP),
        "soft_dc_global_step": int(softdc._GLOBAL_SOFT_DC_STEP),
        "scale_warmup_forward_step": int(model._scale_warmup_forward_step),
    }


def restore_runtime_state(model, state):
    dense._GLOBAL_DENSE_STEP = int(state.get("dense_global_step", 0))
    softdc._GLOBAL_SOFT_DC_STEP = int(state.get("soft_dc_global_step", 0))
    model._scale_warmup_forward_step = int(
        state.get("scale_warmup_forward_step", dense._GLOBAL_DENSE_STEP)
    )


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = visual_base.save_visual
    pipeline.capture_runtime_state = capture_runtime_state
    pipeline.restore_runtime_state = restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = NormalBottleneckRefineLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
