"""Losses for unified contribution-guided DPT geometry adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import finetune_event as fe


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def event_mass_budget(contribution, event_voxel, rho: float):
    mass = event_voxel.float().abs()
    if contribution.ndim == event_voxel.ndim - 1:
        mass = mass.sum(dim=2)
    ratio = (mass * contribution.float()).flatten(1).sum(dim=1) / mass.flatten(1).sum(dim=1).clamp_min(1.0e-6)
    return (ratio - float(rho)).square().mean(), ratio


def pair_consistency(contribution_a, contribution_b, event_voxel):
    support = event_voxel.float().abs() > 0.0
    if contribution_a.ndim == event_voxel.ndim - 1:
        support = support.any(dim=2)
    return _weighted_mean((contribution_a.float() - contribution_b.float()).abs(), support)


@dataclass
class UnifiedLossOutput:
    loss: torch.Tensor
    details: dict
    aux: dict


class UnifiedGeometryContributionLoss:
    """Full-valid geometry supervision with Bridge used only as extra weight."""

    def __init__(
        self,
        *,
        depth_weight=1.0,
        normal_weight=0.25,
        point_weight=1.0,
        bridge_beta=2.0,
        budget_weight=0.05,
        pair_weight=0.2,
        update_weight=0.01,
        decomposition_weight=0.0,
        points_loss_type="l1",
    ):
        self.depth_weight = float(depth_weight)
        self.normal_weight = float(normal_weight)
        self.bridge_beta = float(bridge_beta)
        self.budget_weight = float(budget_weight)
        self.pair_weight = float(pair_weight)
        self.update_weight = float(update_weight)
        self.decomposition_weight = float(decomposition_weight)
        # Reuse the established point-map alignment/loss implementation. Depth
        # and normal are replaced below by explicitly Bridge-weighted versions.
        self.point_criterion = fe.EventSupervisedLoss(
            pose_weight=0.0,
            depth_weight=0.0,
            points_weight=float(point_weight),
            normal_weight=0.0,
            align_depth_scale_enabled=False,
            points_loss_type=points_loss_type,
        )

    def geometry(self, output, views, bridge_mask):
        point_loss, point_details, point_aux = self.point_criterion(output, views)
        depth_pred = torch.stack([item["depth"] for item in output.ress], dim=1).squeeze(-1)
        depth_gt = fe.stack_view_field(views, "depthmap").to(depth_pred)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(depth_pred)
        valid = fe.build_valid_mask(views, depth_gt)
        weight = valid.float() * (1.0 + self.bridge_beta * bridge_mask.float())
        depth_error = (depth_pred.float() - depth_gt.float()).abs()
        depth_loss = _weighted_mean(depth_error, weight)

        normal_pred = fe.depth_to_normals(depth_pred, intrinsics)
        normal_gt = fe.depth_to_normals(depth_gt, intrinsics)
        normal_valid = fe.normal_stencil_valid_mask(valid, depth_pred, eps=1.0e-6)
        pred_unit = F.normalize(normal_pred.float(), dim=-1, eps=1.0e-6)
        gt_unit = F.normalize(normal_gt.float(), dim=-1, eps=1.0e-6)
        normal_error = 1.0 - (pred_unit * gt_unit).sum(dim=-1).clamp(-1.0, 1.0)
        normal_loss = _weighted_mean(normal_error, weight * normal_valid.float())
        geometry_loss = point_loss + self.depth_weight * depth_loss + self.normal_weight * normal_loss
        return geometry_loss, {
            "depth": depth_loss,
            "normal": normal_loss,
            "point": depth_loss.new_tensor(point_details["points_loss"]),
        }, {**point_aux, "depth_pred_live": depth_pred, "valid_live": valid}

    def __call__(
        self,
        output_target,
        target_views,
        bridge_mask,
        event_voxel,
        *,
        rho: float,
        contribution_reference=None,
        geometry_reference=None,
    ) -> UnifiedLossOutput:
        geometry_target, geometry_details, aux = self.geometry(
            output_target, target_views, bridge_mask
        )
        geometry_loss = geometry_target
        if geometry_reference is not None:
            reference_loss, _, _ = geometry_reference
            geometry_loss = 0.5 * (geometry_target + reference_loss)

        contribution = torch.stack(
            [item["event_contribution"] for item in output_target.ress], dim=1
        )
        contribution_spatial = torch.stack(
            [item.get("event_contribution_spatial", item["event_contribution"])
             for item in output_target.ress], dim=1
        )
        if self.decomposition_weight > 0.0 and all(
            "contribution_target" in view for view in target_views
        ):
            decomposition_target = fe.stack_view_field(
                target_views, "contribution_target"
            ).to(contribution_spatial)
            if decomposition_target.ndim == contribution_spatial.ndim + 1:
                decomposition_target = decomposition_target.squeeze(2)
            decomposition_valid = event_voxel.float().abs().sum(dim=2) > 0.0
            if all("decomposition_valid" in view for view in target_views):
                sample_valid = fe.stack_view_field(
                    target_views, "decomposition_valid"
                ).to(device=contribution_spatial.device).bool()
                while sample_valid.ndim < decomposition_valid.ndim:
                    sample_valid = sample_valid.unsqueeze(-1)
                decomposition_valid = decomposition_valid & sample_valid
            decomposition_error = F.smooth_l1_loss(
                contribution_spatial.float(), decomposition_target.float(), reduction="none"
            )
            decomposition_loss = _weighted_mean(
                decomposition_error, decomposition_valid.float()
            )
        else:
            decomposition_target = None
            decomposition_loss = contribution.new_zeros(())
        budget_loss, ratio = event_mass_budget(contribution, event_voxel, rho)
        if contribution_reference is None:
            pair_loss = contribution.new_zeros(())
        else:
            pair_loss = pair_consistency(contribution, contribution_reference, event_voxel)
        update_loss = output_target.ress[0]["adapter_update_loss"]
        total = (
            geometry_loss
            + self.budget_weight * budget_loss
            + self.pair_weight * pair_loss
            + self.update_weight * update_loss
            + self.decomposition_weight * decomposition_loss
        )
        details = {
            **geometry_details,
            "geometry": geometry_loss,
            "budget": budget_loss,
            "pair": pair_loss,
            "update": update_loss,
            "decomposition": decomposition_loss,
            "rho": contribution.new_tensor(float(rho)),
            "contribution_mean": ratio.mean(),
            "contribution_std": contribution.float().std(unbiased=False),
        }
        aux.update({
            "contribution": contribution,
            "contribution_spatial": contribution_spatial,
            "bridge": bridge_mask,
            "decomposition_target": decomposition_target,
        })
        return UnifiedLossOutput(total, details, aux)


__all__ = [
    "UnifiedGeometryContributionLoss",
    "UnifiedLossOutput",
    "event_mass_budget",
    "pair_consistency",
]
