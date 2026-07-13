"""Train C-connected normal confidence and iterative dense depth refinement."""
from __future__ import annotations

import os
from pathlib import Path
import finetune_event as fe
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_conditioned_confidence_refine_model import (
    ConditionedConfidenceNormalRefineLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_conditioned_dense_scale_warmup as dense
from paired_token_reliability import train_linear_voxel_conditioned_soft_dc as softdc
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as normal_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def _weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _high_frequency(normal, kernel=7):
    channels = normal.movedim(-1, -3)
    flat = channels.flatten(0, 1)
    blur = F.avg_pool2d(flat, kernel, stride=1, padding=kernel // 2)
    return (flat - blur).reshape_as(channels).movedim(-3, -1)


class ConfidenceNormalObjective:
    def __init__(self, base, args):
        self.base = base
        self.args = args

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        # The dense wrapper owns the global training step and increments it
        # before reaching this outer objective.
        step = dense._GLOBAL_DENSE_STEP
        normal_weight = dense._cosine_ramp(
            step, dense.SCALE_WARMUP_STEPS, 2000, .50
        )
        confidence_weight = dense._cosine_ramp(
            step, dense.SCALE_WARMUP_STEPS, 2000, .10
        )
        hf_weight = dense._cosine_ramp(
            step, dense.SCALE_WARMUP_STEPS, 2000, .50
        )

        target_normal = torch.stack(
            [item["normal_refine_target"] for item in output.ress], dim=1
        ).float()
        confidence = torch.stack(
            [item["normal_confidence"] for item in output.ress], dim=1
        ).float()
        learned_confidence = torch.stack(
            [item["learned_normal_confidence"] for item in output.ress], dim=1
        ).float()
        coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], dim=1
        ).float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(coarse).float()
        coarse_normal = fe.depth_to_normals(coarse, intrinsics).float()
        gt_normal = result.aux["normal_gt_live"].float()
        valid = result.aux["normal_valid_live"].bool()
        support = torch.stack(
            [item["event_normal_support"] for item in output.ress], dim=1
        ).bool()
        mask = valid & support
        weight = mask.float()

        pred_unit = F.normalize(target_normal, dim=-1, eps=1e-6)
        gt_unit = F.normalize(gt_normal.detach(), dim=-1, eps=1e-6)
        coarse_unit = F.normalize(coarse_normal.detach(), dim=-1, eps=1e-6)
        cosine = 1.0 - (pred_unit * gt_unit).sum(dim=-1).clamp(-1, 1)
        normal_residual_loss = _weighted_mean(cosine, weight)

        coarse_error = 1.0 - (coarse_unit * gt_unit).sum(dim=-1).clamp(-1, 1)
        confidence_target = (coarse_error.detach() / .20).clamp(0.0, 1.0)
        confidence_error = F.binary_cross_entropy(
            learned_confidence.clamp(1e-5, 1-1e-5),
            confidence_target,
            reduction="none",
        )
        confidence_loss = _weighted_mean(confidence_error, weight)

        pred_hf = _high_frequency(pred_unit, kernel=7)
        gt_hf = _high_frequency(gt_unit, kernel=7)
        hf_error = (pred_hf - gt_hf).abs().mean(dim=-1)
        hf_loss = _weighted_mean(hf_error, weight)

        result.loss = (
            result.loss
            + normal_weight * normal_residual_loss
            + confidence_weight * confidence_loss
            + hf_weight * hf_loss
        )
        result.details["normal_residual"] = normal_residual_loss
        result.details["normal_confidence"] = confidence_loss
        result.details["normal_hf"] = hf_loss
        result.details["normal_confidence_mean"] = _weighted_mean(confidence, weight)
        if (
            torch.is_grad_enabled() and step % 100 == 0
            and int(os.environ.get("RANK", "0")) == 0
        ):
            print(
                f"[c-normal@{step:05d}] NresW={normal_weight:.4f} "
                f"ConfW={confidence_weight:.4f} HFW={hf_weight:.4f} "
                f"CConf={float(result.details['normal_confidence_mean'].detach()):.4f}",
                flush=True,
            )
        return result


def build_model(cfg, args, device):
    m = cfg.model
    model = ConditionedConfidenceNormalRefineLinearVoxelModel(
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
        f"[C-confidence normal-refine base] scale={float(model.metric_depth_scale):.4f} "
        f"warmup={model.scale_warmup_steps} iterations={model.normal_refine_iterations} "
        f"step_limit={model.normal_refine_step_limit:.3f} CxConf=ON",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, train_heads_a=False):
    softdc.conditioned_base.configure_phase(model, phase, train_heads_a)
    # The obsolete absolute-normal decoder is replaced by coarse+delta normal.
    model.event_normal_decoder.requires_grad_(False)
    enabled = phase in {"adapter", "joint"}
    model.event_normal_delta_head.requires_grad_(enabled)
    model.normal_confidence_head.requires_grad_(enabled)
    model.normal_depth_refiner.requires_grad_(enabled)


def optimizer_for(model, phase, args):
    """Give the newly initialized geometry heads enough learning speed."""
    scale_id = id(model.depth_log_scale)
    fast_ids = {
        id(parameter)
        for module in (
            model.conditioned_depth_head,
            model.event_normal_delta_head,
            model.normal_confidence_head,
            model.normal_depth_refiner,
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
    return torch.optim.AdamW(
        groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95)
    )


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    normal_base.save_visual(
        output_root, phase, epoch, batch_index, views, reference_views,
        event, bridge, output, aux,
    )
    item = output.ress[0]
    contribution = item["event_contribution"][0].detach().float().cpu()
    learned = item["learned_normal_confidence"][0].detach().float().cpu()
    connected = item["normal_confidence"][0].detach().float().cpu()
    delta = item["event_normal_delta"][0].detach().float().cpu().norm(dim=-1)
    panels = (
        (contribution, "C: contribution"),
        (learned, "learned normal confidence"),
        (connected, "C x normal confidence"),
        (delta, "event normal residual magnitude"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for axis, (image, title) in zip(axes, panels):
        rendered = axis.imshow(image.numpy(), cmap="magma", vmin=0.0)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(rendered, ax=axis, fraction=.046, pad=.04)
    path = (
        Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}"
        / f"batch_{batch_index+1:06d}_c_normal_confidence.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def criterion_for(args, phase):
    base = dense.criterion_for(args, phase)
    return ConfidenceNormalObjective(base, args) if phase in {"adapter", "joint"} else base


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.UnifiedGeometryContributionModel = ConditionedConfidenceNormalRefineLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
