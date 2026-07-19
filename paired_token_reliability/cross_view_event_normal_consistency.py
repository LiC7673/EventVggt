"""Patchwise cross-view consistency for event-predicted normal derivatives."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import finetune_event as fe


def _frame_ids(views):
    ids = []
    for view in views:
        values = view.get("instance")
        value = values[0] if isinstance(values, (list, tuple)) else values
        match = re.search(r"(?:_|frame)(\d+)$", str(value))
        ids.append(int(match.group(1)) if match else None)
    return ids


def ensure_adjacent_views(views):
    """Fail loudly if a four-view batch is not temporally consecutive."""
    if len(views) < 2:
        raise RuntimeError("cross-view derivative consistency needs at least two views")
    ids = _frame_ids(views)
    if all(value is not None for value in ids):
        if any(right != left + 1 for left, right in zip(ids[:-1], ids[1:])):
            raise RuntimeError(f"views are not adjacent/in-order: frame ids={ids}")
    return ids


def _pool(value, patch, weight=None):
    if weight is None:
        return F.avg_pool2d(value, patch, patch)
    numerator = F.avg_pool2d(value * weight, patch, patch)
    denominator = F.avg_pool2d(weight, patch, patch)
    return numerator / denominator.clamp_min(1.0e-6), denominator


def _sample(value, grid):
    return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def cross_view_patch_loss(output, views, *, patch_size=14, min_overlap=8,
                          min_overlap_ratio=.03, depth_tolerance=.20):
    """Return consistency loss and per-pair patch diagnostics.

    Normal-derivative magnitude is rotation invariant.  Comparing it avoids an
    invalid direct L1 between derivatives expressed in different image axes.
    Camera/depth geometry is detached: this loss trains the event derivative
    branch, not a pose/depth shortcut.
    """
    ensure_adjacent_views(views)
    derivative = torch.stack(
        [item["event_normal_derivative_full"] for item in output.ress], 1
    ).float()
    derivative = derivative.reshape(*derivative.shape[:4], -1)
    magnitude = derivative.square().sum(-1).sqrt()
    depth = torch.stack([item["depth_hdr_base"][..., 0] for item in output.ress], 1).float().detach()
    pose_enc = torch.stack([item["camera_pose"] for item in output.ress], 1).float().detach()
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(depth).float().detach()
    gt_fields = [view.get("depthmap") for view in views]
    if all(torch.is_tensor(value) for value in gt_fields):
        gt_depth = fe.stack_view_field(views, "depthmap").to(depth).float()
        valid = fe.build_valid_mask(views, gt_depth).bool()
    else:
        valid = fe.build_valid_mask(views, depth).bool()
    support = torch.stack([item["event_normal_support"] for item in output.ress], 1).bool()
    confidence = torch.stack([item["normal_fusion_gate"] for item in output.ress], 1).float().detach()
    b, view_count, height, width = depth.shape
    c2w, _ = fe.pose_encoding_to_c2w(pose_enc, image_size_hw=(height, width))
    c2w = c2w.detach().clone()
    scene_scale = output.ress[0].get("metric_depth_scale")
    if torch.is_tensor(scene_scale):
        scene_scale = scene_scale.detach().to(c2w).reshape(b)
        c2w[..., :3, 3] = c2w[..., :3, 3] * scene_scale[:, None, None]
    w2c = torch.linalg.inv(c2w)
    pairs = [(index, index + 1) for index in range(view_count - 1)]
    losses, diagnostics = [], []
    for source_index, target_index in pairs:
        for batch_index in range(b):
            source_valid = (valid[batch_index, source_index] & support[batch_index, source_index]).float()[None, None]
            target_valid = (valid[batch_index, target_index] & support[batch_index, target_index]).float()[None, None]
            source_depth, source_coverage = _pool(
                depth[batch_index, source_index][None, None], patch_size, source_valid
            )
            target_depth, target_coverage = _pool(
                depth[batch_index, target_index][None, None], patch_size, target_valid
            )
            source_mag, _ = _pool(
                magnitude[batch_index, source_index][None, None], patch_size, source_valid
            )
            target_mag, _ = _pool(
                magnitude[batch_index, target_index][None, None], patch_size, target_valid
            )
            source_conf = _pool(confidence[batch_index, source_index][None, None], patch_size)
            target_conf = _pool(confidence[batch_index, target_index][None, None], patch_size)
            grid_h, grid_w = source_depth.shape[-2:]
            ys = (torch.arange(grid_h, device=depth.device, dtype=depth.dtype) + .5) * patch_size - .5
            xs = (torch.arange(grid_w, device=depth.device, dtype=depth.dtype) + .5) * patch_size - .5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            z = source_depth[0, 0]
            ks = intrinsics[batch_index, source_index]
            x = (xx - ks[0, 2]) * z / ks[0, 0].clamp_min(1.e-6)
            y = (yy - ks[1, 2]) * z / ks[1, 1].clamp_min(1.e-6)
            xyz1 = torch.stack((x, y, z, torch.ones_like(z)), -1)
            world = torch.einsum("ij,hwj->hwi", c2w[batch_index, source_index], xyz1)
            target_xyz1 = torch.einsum("ij,hwj->hwi", w2c[batch_index, target_index], world)
            target_xyz = target_xyz1[..., :3]
            target_z = target_xyz[..., 2]
            kt = intrinsics[batch_index, target_index]
            u = kt[0, 0] * target_xyz[..., 0] / target_z.clamp_min(1.e-6) + kt[0, 2]
            v = kt[1, 1] * target_xyz[..., 1] / target_z.clamp_min(1.e-6) + kt[1, 2]
            patch_x = (u + .5) / patch_size - .5
            patch_y = (v + .5) / patch_size - .5
            norm_x = 2. * patch_x / max(grid_w - 1, 1) - 1.
            norm_y = 2. * patch_y / max(grid_h - 1, 1) - 1.
            grid = torch.stack((norm_x, norm_y), -1)[None]
            warped_mag = _sample(target_mag, grid)
            warped_depth = _sample(target_depth, grid)
            warped_coverage = _sample(target_coverage, grid)
            warped_conf = _sample(target_conf, grid)
            relative_depth_error = (
                (target_z[None, None] - warped_depth).abs()
                / warped_depth.abs().clamp_min(1.e-3)
            )
            overlap = (
                (source_coverage > .20) & (warped_coverage > .20)
                & (target_z[None, None] > 1.e-4)
                & (norm_x[None, None].abs() <= 1.) & (norm_y[None, None].abs() <= 1.)
                & (relative_depth_error < depth_tolerance)
            )
            overlap_count = int(overlap.sum().detach())
            overlap_ratio = overlap_count / max(grid_h * grid_w, 1)
            accepted = overlap_count >= int(min_overlap) and overlap_ratio >= float(min_overlap_ratio)
            # Patch calibration absorbs view-dependent image-plane derivative
            # scale, while retaining the spatial pattern being supervised.
            active = overlap & (source_mag > 1.e-5) & (warped_mag > 1.e-5)
            if active.any():
                calibration = (warped_mag[active].detach() / source_mag[active].detach().clamp_min(1.e-6)).median().clamp(.25, 4.)
            else:
                calibration = source_mag.new_tensor(1.)
            calibrated = source_mag * calibration
            error = F.smooth_l1_loss(calibrated, warped_mag, beta=.01, reduction="none")
            weights = overlap.float() * torch.sqrt((source_conf * warped_conf).clamp_min(0.))
            pair_loss = (error * weights).sum() / weights.sum().clamp_min(1.e-6)
            if accepted:
                losses.append(pair_loss)
            diagnostics.append(dict(
                source=source_index, target=target_index, batch=batch_index,
                source_magnitude=source_mag[0, 0].detach(),
                warped_target_magnitude=warped_mag[0, 0].detach(),
                overlap=overlap[0, 0].detach(),
                error=(calibrated - warped_mag).abs()[0, 0].detach(),
                overlap_count=overlap_count, overlap_ratio=overlap_ratio,
                calibration=float(calibration.detach()), accepted=accepted,
            ))
    loss = torch.stack(losses).mean() if losses else magnitude.sum() * 0.
    return loss, diagnostics


@torch.no_grad()
def save_patch_diagnostics(root, phase, epoch, batch_index, diagnostics):
    root = Path(root) / "cross_view_patch_consistency" / phase / f"epoch_{epoch:03d}"
    root.mkdir(parents=True, exist_ok=True)
    for item in diagnostics:
        if item["batch"] != 0:
            continue
        panels = (
            (item["source_magnitude"], "source patch |dN|", "magma"),
            (item["warped_target_magnitude"], "target |dN| warped to source", "magma"),
            (item["overlap"].float(), "trusted overlap patches", "gray"),
            (item["error"], "calibrated consistency error", "magma"),
        )
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for axis, (image, title, cmap) in zip(axes, panels):
            shown = axis.imshow(image.cpu().numpy(), cmap=cmap)
            axis.set_title(title); axis.axis("off")
            fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
        fig.suptitle(
            f"adjacent views {item['source']}→{item['target']}  "
            f"overlap={item['overlap_count']} ({item['overlap_ratio']:.3f})  "
            f"calibration={item['calibration']:.3f} accepted={item['accepted']}"
        )
        fig.tight_layout()
        fig.savefig(root / f"batch_{batch_index:06d}_pair_{item['source']}_{item['target']}.png", dpi=140)
        plt.close(fig)


__all__ = ["cross_view_patch_loss", "save_patch_diagnostics", "ensure_adjacent_views"]
