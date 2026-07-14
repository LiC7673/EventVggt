"""One-stage full/geo alignment, reliability distillation, and geometry training."""
from __future__ import annotations

import os
from pathlib import Path
import finetune_event as fe
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_full_geo_alignment_model import (
    FullGeoAlignmentLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_conditioned_dense_scale_warmup as dense
from paired_token_reliability import train_linear_voxel_conditioned_soft_dc as softdc
from paired_token_reliability import train_linear_voxel_detail_normal_derivative as normal_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


_BASE_PREPARE_PAIR = pipeline.prepare_pair


def prepare_full_geo_pair(batch, device, args, _phase):
    # contribution mode retains E_full in event_voxel while decomposition
    # supervision keeps the paired E_geo in geometry_event_voxel.
    return _BASE_PREPARE_PAIR(batch, device, args, "contribution")


def one_stage_schedule(epochs, contribution_epochs, joint_epochs=0):
    if int(contribution_epochs) != 0 or int(joint_epochs) != 0:
        raise ValueError("one-stage alignment requires --epochs-b=0 --epochs-c=0")
    if int(epochs) <= 0:
        raise ValueError("one-stage alignment requires --epochs-a > 0")
    return ["adapter"] * int(epochs)


def _weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _hf(normal, kernel):
    channels = normal.movedim(-1, -3)
    flat = channels.flatten(0, 1)
    blur = F.avg_pool2d(flat, kernel, 1, kernel // 2)
    return (flat - blur).reshape_as(channels).movedim(-3, -1)


class FullGeoAlignmentObjective:
    def __init__(self, base):
        self.base = base

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        step = dense._GLOBAL_DENSE_STEP
        active = bool(float(output.ress[0]["normal_bottleneck_active"].detach()))
        if not active and "dense_depth_residual" in result.details:
            result.loss = result.loss - (
                result.details["detail_schedule_weight"]
                * result.details["dense_depth_residual"]
            )

        full_n = torch.stack([x["event_normal_full"] for x in output.ress], 1).float()
        geo_n = torch.stack([x["event_normal_geo"] for x in output.ress], 1).float()
        full_support = torch.stack([x["event_normal_support"] for x in output.ress], 1).bool()
        geo_support = torch.stack([x["geo_event_support"] for x in output.ress], 1).bool()
        feature_error = torch.stack([x["alignment_feature_error"] for x in output.ress], 1).float()
        reliability = torch.stack([x["alignment_reliability"] for x in output.ress], 1).float()
        reliability_target = torch.stack(
            [x["alignment_reliability_target"] for x in output.ress], 1
        ).float()
        gt = result.aux["normal_gt_live"].float().detach()
        valid = result.aux["normal_valid_live"].bool()
        full_weight = (valid & full_support).float()
        geo_weight = (valid & geo_support).float()
        pair_weight = (valid & (full_support | geo_support)).float()

        full_u = F.normalize(full_n, dim=-1, eps=1e-6)
        geo_u = F.normalize(geo_n, dim=-1, eps=1e-6)
        gt_u = F.normalize(gt, dim=-1, eps=1e-6)
        full_normal = _weighted_mean(
            1.0 - (full_u * gt_u).sum(-1).clamp(-1, 1), full_weight
        )
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
        full_hf = _weighted_mean(.5 * hf_error, full_weight)
        align_loss = _weighted_mean(feature_error, pair_weight)
        reliability_loss = _weighted_mean(
            F.smooth_l1_loss(
                reliability, reliability_target, beta=.05, reduction="none"
            ), pair_weight,
        )

        result.loss = result.loss + (
            1.0 * full_normal + 1.0 * geo_normal + .5 * normal_distill
            + 1.0 * full_hf + 1.0 * align_loss + .25 * reliability_loss
        )
        result.details["full_event_normal"] = full_normal
        result.details["geo_event_normal"] = geo_normal
        result.details["normal_distill"] = normal_distill
        result.details["normal_hf"] = full_hf
        result.details["feature_alignment"] = align_loss
        result.details["alignment_reliability"] = reliability_loss
        if torch.is_grad_enabled() and step % 100 == 0 and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[full-geo-align@{step:05d}] depth_active={int(active)} "
                f"align={float(align_loss.detach()):.5f} "
                f"Nfull={float(full_normal.detach()):.5f} "
                f"Ngeo={float(geo_normal.detach()):.5f} "
                f"C={float(_weighted_mean(reliability, pair_weight).detach()):.4f} "
                f"Ct={float(_weighted_mean(reliability_target, pair_weight).detach()):.4f}",
                flush=True,
            )
        return result


def build_model(cfg, args, device):
    m = cfg.model
    model = FullGeoAlignmentLinearVoxelModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        event_dc_limit=float(getattr(m, "event_dc_limit", .50)),
        event_residual_target_limit=float(getattr(m, "event_residual_target_limit", .50)),
        scale_warmup_steps=int(getattr(m, "scale_warmup_steps", 1000)),
        normal_bottleneck_warmup_steps=int(getattr(m, "normal_bottleneck_warmup_steps", 1000)),
        alignment_confidence_tau=float(getattr(m, "alignment_confidence_tau", .10)),
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
        f"[one-stage full/geo alignment] epochs=A-only "
        f"depth_warmup={model.normal_bottleneck_warmup_steps} "
        f"separate_contribution_net=OFF raw_event_to_depth=OFF",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, train_heads_a=False):
    if phase != "adapter":
        raise ValueError(f"one-stage route only supports adapter, got {phase}")
    softdc.conditioned_base.configure_phase(model, phase, train_heads_a)
    model.depth_local_head.requires_grad_(False)
    model.conditioned_depth_head.requires_grad_(False)
    model.contribution_net.requires_grad_(False)
    model.event_encoder.requires_grad_(True)
    model.event_normal_decoder.requires_grad_(True)
    model.full_geo_aligner.requires_grad_(True)
    model.normal_bottleneck_refiner.requires_grad_(True)


def optimizer_for(model, _phase, args):
    scale_id = id(model.depth_log_scale)
    fast_ids = {
        id(p) for module in (
            model.event_normal_decoder, model.full_geo_aligner,
            model.normal_bottleneck_refiner,
        ) for p in module.parameters()
    }
    encoder_ids = {id(p) for p in model.event_encoder.parameters()}
    groups = {"regular": [], "encoder": [], "fast": [], "scale": []}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        pid = id(parameter)
        if pid == scale_id:
            groups["scale"].append(parameter)
        elif pid in fast_ids:
            groups["fast"].append(parameter)
        elif pid in encoder_ids:
            groups["encoder"].append(parameter)
        else:
            groups["regular"].append(parameter)
    params = []
    if groups["regular"]:
        params.append({"params": groups["regular"], "lr": args.lr})
    if groups["encoder"]:
        params.append({"params": groups["encoder"], "lr": 2.0 * args.lr})
    if groups["fast"]:
        params.append({"params": groups["fast"], "lr": 5.0 * args.lr})
    if groups["scale"]:
        params.append({"params": groups["scale"], "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    normal_base.save_visual(
        output_root, phase, epoch, batch_index, views, reference_views,
        event, bridge, output, aux,
    )
    item = output.ress[0]
    full = item["event_normal_full"][0].detach().float().cpu()
    geo = item["event_normal_geo"][0].detach().float().cpu()
    panels = (
        (item["alignment_reliability"][0].detach().float().cpu(), "alignment C", "magma"),
        (item["alignment_reliability_target"][0].detach().float().cpu(), "alignment target", "magma"),
        (item["alignment_feature_error"][0].detach().float().cpu(), "feature error", "magma"),
        ((full - geo).norm(dim=-1), "full/geo normal difference", "magma"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for axis, (image, title, cmap) in zip(axes, panels):
        shown = axis.imshow(image.numpy(), cmap=cmap)
        axis.set_title(title); axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=.046, pad=.04)
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}_full_geo_alignment.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def criterion_for(args, phase):
    base = dense.criterion_for(args, phase)
    return FullGeoAlignmentObjective(base)


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
    pipeline.prepare_pair = prepare_full_geo_pair
    pipeline.build_alternating_phase_schedule = one_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.capture_runtime_state = capture_runtime_state
    pipeline.restore_runtime_state = restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = FullGeoAlignmentLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
