"""Train event depth detail with a direct, scale-free residual target."""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_detail_residual_model import (
    DetailResidualLinearVoxelModel,
    support_center,
)
from paired_token_reliability import train_linear_voxel_multiscale as linear_base
from paired_token_reliability import train_signed_multiscale as visual_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


DETAIL_WEIGHT = 5.0
DETAIL_LIMIT = 0.25


def build_model(cfg, args, device):
    model = DetailResidualLinearVoxelModel(
        img_size=int(cfg.model.img_size), patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(cfg.model, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(cfg.model, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(cfg.model, "depth_update_scale", 1.0)),
        event_decay_tau=float(getattr(cfg.model, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(cfg.model, "depth_log_scale_limit", 2.0)),
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
        f"[detail residual base] scale={float(model.metric_depth_scale):.4f} "
        f"new={len(message.missing_keys)} unused={len(message.unexpected_keys)}",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    linear_base.configure_phase(model, phase, _train_heads_a)
    model.depth_log_scale.requires_grad_(phase in {"adapter", "joint"})


def optimizer_for(model, phase, args):
    scale_id = id(model.depth_log_scale)
    local_ids = {id(p) for p in model.depth_local_head.parameters()}
    encoder_ids = {id(p) for p in model.event_encoder.parameters()}
    scale, local, encoder, regular = [], [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        pid = id(parameter)
        if pid == scale_id:
            scale.append(parameter)
        elif pid in local_ids:
            local.append(parameter)
        elif pid in encoder_ids:
            encoder.append(parameter)
        else:
            regular.append(parameter)
    groups = []
    if regular:
        groups.append({"params": regular, "lr": args.lr})
    if encoder:
        groups.append({"params": encoder, "lr": 2.0 * args.lr})
    if local:
        groups.append({"params": local, "lr": 5.0 * args.lr})
    if scale:
        groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


class DetailResidualObjective:
    def __init__(self, base, weight=DETAIL_WEIGHT, limit=DETAIL_LIMIT):
        self.base = base
        self.weight = float(weight)
        self.limit = float(limit)
        self.calls = 0

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        pred = torch.stack([x["depth_delta_ratio"] for x in output.ress], dim=1).float()
        coarse = torch.stack([x["depth_coarse"] for x in output.ress], dim=1)[..., 0].float()
        gt = fe.stack_view_field(views, "depthmap").to(pred).float()
        valid = result.aux["valid_live"].bool()
        support = torch.stack(
            [x["event_normal_reliability"] > 0 for x in output.ress], dim=1
        )
        mask = valid & support & torch.isfinite(gt) & torch.isfinite(coarse) & (coarse > 1e-6)

        # Detaching coarse prevents this auxiliary term from changing RGB
        # scale.  Removing the target DC component leaves event-only detail.
        target = gt / coarse.detach().clamp_min(1e-6) - 1.0
        target = support_center(target, mask).clamp(-self.limit, self.limit)
        error = F.smooth_l1_loss(pred, target, beta=.01, reduction="none")
        detail = (error * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        result.loss = result.loss + self.weight * detail
        result.details["detail_residual"] = detail
        result.aux["detail_residual_target"] = target

        self.calls += 1
        if self.calls % 500 == 0 and int(os.environ.get("RANK", "0")) == 0:
            active_pred = pred[mask].abs()
            active_target = target[mask].abs()
            if active_pred.numel():
                print(
                    f"[detail-target@{self.calls:05d}] loss={float(detail.detach()):.6f} "
                    f"pred_abs={float(active_pred.mean()):.6f} "
                    f"target_abs={float(active_target.mean()):.6f}",
                    flush=True,
                )
        return result


def criterion_for(args, phase):
    criterion = linear_base.criterion_for(args, phase)
    return DetailResidualObjective(criterion) if phase in {"adapter", "joint"} else criterion


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = visual_base.save_visual
    pipeline.UnifiedGeometryContributionModel = DetailResidualLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
