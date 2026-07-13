"""Train event-normal derivatives only in local regions containing events."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_detail_normal_derivative_model import (
    DetailNormalDerivativeLinearVoxelModel,
)
from paired_token_reliability import train_linear_voxel_detail_residual as detail_base
from paired_token_reliability import train_signed_multiscale as visual_base
from paired_token_reliability import train_unified_geometry_contribution as pipeline


LOCAL_PATCH = 8
EVENT_NEIGHBORHOOD = 3


def build_model(cfg, args, device):
    model = DetailNormalDerivativeLinearVoxelModel(
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
        f"[normal derivative base] scale={float(model.metric_depth_scale):.4f} "
        f"new={len(message.missing_keys)} unused={len(message.unexpected_keys)}",
        flush=True,
    )
    return model.to(device)


def _patchwise_active_mean(value, mask, patch_size=LOCAL_PATCH):
    """Average active pixels per patch, then average only nonempty patches."""
    n = int(np.prod(value.shape[:-2]))
    height, width = value.shape[-2:]
    value = value.reshape(n, 1, height, width)
    mask = mask.reshape(n, 1, height, width).to(value)
    pad_h = (-height) % int(patch_size)
    pad_w = (-width) % int(patch_size)
    value = F.pad(value, (0, pad_w, 0, pad_h))
    mask = F.pad(mask, (0, pad_w, 0, pad_h))
    values = F.unfold(value * mask, kernel_size=patch_size, stride=patch_size).sum(dim=1)
    counts = F.unfold(mask, kernel_size=patch_size, stride=patch_size).sum(dim=1)
    active = counts > 0
    if not active.any():
        return value.sum() * 0.0
    return (values / counts.clamp_min(1.0))[active].mean()


def _vector_derivative_error(pred_delta, target_delta):
    vector = (pred_delta - target_delta).abs().mean(dim=-1)
    target_magnitude = target_delta.norm(dim=-1)
    pred_magnitude = pred_delta.norm(dim=-1)
    direction = 1.0 - (
        F.normalize(pred_delta, dim=-1, eps=1e-6)
        * F.normalize(target_delta, dim=-1, eps=1e-6)
    ).sum(dim=-1).clamp(-1, 1)
    # Direction is meaningful only on an actual GT normal edge. Flat areas
    # still supervise the derivative magnitude toward zero.
    direction_valid = (target_magnitude > .01) & (pred_magnitude.detach() > .005)
    return vector + 0.10 * direction * direction_valid.to(vector)


def normal_derivative_loss(pred, target, normal_valid, event_support):
    """Local Nx/Ny derivative loss; completely empty regions contribute zero."""
    b, v, height, width = event_support.shape
    local = F.max_pool2d(
        event_support.float().reshape(b * v, 1, height, width),
        EVENT_NEIGHBORHOOD, 1, EVENT_NEIGHBORHOOD // 2,
    ).reshape(b, v, height, width) > 0

    pred_dx = pred[..., :, 1:, :] - pred[..., :, :-1, :]
    target_dx = target[..., :, 1:, :] - target[..., :, :-1, :]
    mask_dx = (
        normal_valid[..., :, 1:] & normal_valid[..., :, :-1]
        & (local[..., :, 1:] | local[..., :, :-1])
    )
    pred_dy = pred[..., 1:, :, :] - pred[..., :-1, :, :]
    target_dy = target[..., 1:, :, :] - target[..., :-1, :, :]
    mask_dy = (
        normal_valid[..., 1:, :] & normal_valid[..., :-1, :]
        & (local[..., 1:, :] | local[..., :-1, :])
    )
    loss_x = _patchwise_active_mean(
        _vector_derivative_error(pred_dx, target_dx), mask_dx
    )
    loss_y = _patchwise_active_mean(
        _vector_derivative_error(pred_dy, target_dy), mask_dy
    )
    return .5 * (loss_x + loss_y), local


def _disable_absolute_event_normal(criterion):
    """Find the wrapped unified loss and disable its dense absolute EN/DN."""
    current = criterion
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "event_normal_weight"):
            current.event_normal_weight = 0.0
            current.depth_event_normal_weight = 0.0
            return
        if hasattr(current, "base"):
            current = current.base
        elif hasattr(current, "criterion"):
            current = current.criterion
        else:
            break
    raise RuntimeError("could not locate UnifiedGeometryContributionLoss")


class NormalDerivativeObjective:
    def __init__(self, base, event_weight, depth_weight):
        self.base = base
        self.event_weight = float(event_weight)
        self.depth_weight = float(depth_weight)

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        event_normal = torch.stack(
            [x["event_normal"] for x in output.ress], dim=1
        ).float()
        final_normal = result.aux["normal_pred_live"].float()
        gt_normal = result.aux["normal_gt_live"].float()
        normal_valid = result.aux["normal_valid_live"].bool()
        support = torch.stack(
            [x["event_normal_support"] for x in output.ress], dim=1
        ).bool()

        event_derivative, local = normal_derivative_loss(
            event_normal, gt_normal, normal_valid, support
        )
        depth_derivative, _ = normal_derivative_loss(
            final_normal, event_normal.detach(), normal_valid, support
        )
        result.loss = (
            result.loss
            + self.event_weight * event_derivative
            + self.depth_weight * depth_derivative
        )
        # Reuse existing log columns, but they now mean derivative losses.
        result.details["event_normal"] = event_derivative
        result.details["depth_event_normal"] = depth_derivative
        result.details["event_normal_derivative"] = event_derivative
        result.details["depth_event_normal_derivative"] = depth_derivative
        result.aux["event_normal_valid_live"] = normal_valid & local
        result.aux["event_normal_local_support"] = local
        return result


def criterion_for(args, phase):
    criterion = detail_base.criterion_for(args, phase)
    _disable_absolute_event_normal(criterion)
    if phase not in {"adapter", "joint"}:
        return criterion
    return NormalDerivativeObjective(
        criterion, args.event_normal_weight, args.depth_event_normal_weight
    )


def _derivative_magnitude(normal):
    dx = normal[:, 1:] - normal[:, :-1]
    dy = normal[1:, :] - normal[:-1, :]
    value = normal.new_zeros(normal.shape[:2])
    value[:, 1:] += dx.norm(dim=-1)
    value[1:, :] += dy.norm(dim=-1)
    return value


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    visual_base.save_visual(
        output_root, phase, epoch, batch_index, views, reference_views,
        event, bridge, output, aux,
    )
    pred = output.ress[0]["event_normal"][0].detach().float().cpu()
    target = aux["normal_gt_live"][0, 0].detach().float().cpu()
    support = output.ress[0]["event_normal_support"][0].detach().float().cpu()
    pred_d = _derivative_magnitude(pred)
    target_d = _derivative_magnitude(target)
    error = (pred_d - target_d).abs()
    panels = ((support, "real event support"), (pred_d, "pred normal derivative"),
              (target_d, "GT normal derivative"), (error, "derivative error"))
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image.numpy(), cmap="magma")
        axis.set_title(title); axis.axis("off")
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}_normal_derivative.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def main(argv=None):
    pipeline.build_model = build_model
    pipeline.configure_phase = detail_base.configure_phase
    pipeline.optimizer_for = detail_base.optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.UnifiedGeometryContributionModel = DetailNormalDerivativeLinearVoxelModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
