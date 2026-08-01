                                                           
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from release_artifacts.dn_core_showcase.core_showcase import (
    compose_core_objective,
    differential_geometry_losses,
    high_frequency_log_depth_target,
)


@dataclass
class LossWeights:
    hdr: float = 1.0
    event_dn: float = 1.0
    depth_dn: float = 1.0
    high_frequency: float = 2.0
    cross_view: float = 0.2


def stack_views(batch, key):
    return torch.stack([view[key] for view in batch["views"]], dim=1)


def compute_losses(model, batch, depth_to_normals,
                   cross_view_loss_function, weights):
    images = stack_views(batch, "image")
    events = stack_views(batch, "event_voxel")
    intrinsics = stack_views(batch, "intrinsics")
    gt_depth = stack_views(batch, "depth")
    valid = stack_views(batch, "valid_depth")
    outputs = []
    for view in range(images.shape[1]):
        outputs.append(model(images[:, view], events[:, view], intrinsics[:, view]))

    final_depth = torch.stack([item["final_depth"] for item in outputs], 1)
    hdr_depth = torch.stack([item["hdr_depth"] for item in outputs], 1)
    event_dn = torch.stack([item["event_derivative"] for item in outputs], 1)
    support = events.abs().sum(2) > 0

    base_geometry = F.smooth_l1_loss(final_depth[valid], gt_depth[valid])
                                                                            
                                                   
    hdr_alignment = final_depth.new_zeros(())
    event_dn_loss, depth_dn_loss = differential_geometry_losses(
        event_dn, final_depth, gt_depth, intrinsics, valid, support,
        depth_to_normals,
    )
    hf_target = high_frequency_log_depth_target(
        hdr_depth.flatten(0, 1), gt_depth.flatten(0, 1),
        valid.flatten(0, 1),
    ).reshape_as(final_depth)
    predicted_hf = torch.stack([item["log_depth_update"] for item in outputs], 1)
    hf_loss = F.smooth_l1_loss(predicted_hf[valid], hf_target[valid])
    cross_view_loss = cross_view_loss_function(outputs, batch["views"])

    total = compose_core_objective(
        base_geometry, hdr_alignment, event_dn_loss, depth_dn_loss,
        hf_loss, cross_view_loss, weights,
    )
    return total, {
        "geometry": base_geometry.detach(),
        "event_dn": event_dn_loss.detach(),
        "depth_dn": depth_dn_loss.detach(),
        "high_frequency": hf_loss.detach(),
        "cross_view_dn": cross_view_loss.detach(),
    }


def train_one_step(model, batch, optimizer, depth_to_normals,
                   cross_view_loss_function, weights=LossWeights()):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = compute_losses(
        model, batch, depth_to_normals, cross_view_loss_function, weights
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=1.0,
    )
    optimizer.step()
    return float(loss.detach()), diagnostics
