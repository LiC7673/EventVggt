"""Losses for unified contribution-guided DPT geometry adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import finetune_event as fe
from paired_token_reliability.contribution_stage1 import geometry_emphasis_weight


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def supervised_log_depth_derivative_losses(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    valid: torch.Tensor,
    *,
    patch_size: int = 14,
    eps: float = 1.0e-6,
):
    """Match GT slopes/curvature and explicitly suppress patch-boundary error.

    Unlike an unsupervised smoothness prior, these terms preserve real surface
    detail because the target derivatives come from GT depth.  Log depth makes
    the comparison less sensitive to absolute scene scale.
    """
    pred = torch.log(depth_pred.float().clamp_min(eps))
    target = torch.log(depth_gt.float().clamp_min(eps))
    valid = valid.bool() & torch.isfinite(pred) & torch.isfinite(target)

    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
    error_dx = (pred_dx - target_dx).abs()
    error_dy = (pred_dy - target_dy).abs()
    gradient_loss = 0.5 * (
        _weighted_mean(error_dx, valid_dx.float())
        + _weighted_mean(error_dy, valid_dy.float())
    )

    pred_dxx = pred_dx[..., :, 1:] - pred_dx[..., :, :-1]
    target_dxx = target_dx[..., :, 1:] - target_dx[..., :, :-1]
    valid_dxx = valid_dx[..., :, 1:] & valid_dx[..., :, :-1]
    pred_dyy = pred_dy[..., 1:, :] - pred_dy[..., :-1, :]
    target_dyy = target_dy[..., 1:, :] - target_dy[..., :-1, :]
    valid_dyy = valid_dy[..., 1:, :] & valid_dy[..., :-1, :]
    curvature_loss = 0.5 * (
        _weighted_mean((pred_dxx - target_dxx).abs(), valid_dxx.float())
        + _weighted_mean((pred_dyy - target_dyy).abs(), valid_dyy.float())
    )

    grid_terms = []
    patch_size = int(patch_size)
    if patch_size > 1:
        for col in range(patch_size, pred.shape[-1], patch_size):
            edge = col - 1
            grid_terms.append(
                _weighted_mean(error_dx[..., edge], valid_dx[..., edge].float())
            )
        for row in range(patch_size, pred.shape[-2], patch_size):
            edge = row - 1
            grid_terms.append(
                _weighted_mean(error_dy[..., edge, :], valid_dy[..., edge, :].float())
            )
    grid_loss = (
        torch.stack(grid_terms).mean() if grid_terms else depth_pred.new_zeros(())
    )
    return gradient_loss, curvature_loss, grid_loss


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


# def geometry_contribution_rank_loss(
#     contribution,
#     geometry_score,
#     valid,
#     event_voxel,
#     *,
#     margin=0.05,
#     difference_threshold=0.10,
# ):
#     """Enforce ordering, not equality, between GT geometry detail and C."""
#     support = event_voxel.float().abs().sum(dim=2) > 0.0
#     eligible = valid.bool() & support
#     losses = []
#     for dimension in (-1, -2):
#         geo_left = geometry_score.narrow(dimension, 0, geometry_score.shape[dimension] - 1)
#         geo_right = geometry_score.narrow(dimension, 1, geometry_score.shape[dimension] - 1)
#         contribution_left = contribution.narrow(dimension, 0, contribution.shape[dimension] - 1)
#         contribution_right = contribution.narrow(dimension, 1, contribution.shape[dimension] - 1)
#         valid_left = eligible.narrow(dimension, 0, eligible.shape[dimension] - 1)
#         valid_right = eligible.narrow(dimension, 1, eligible.shape[dimension] - 1)
#         geo_difference = geo_right - geo_left
#         pair_mask = valid_left & valid_right & (
#             geo_difference.abs() > float(difference_threshold)
#         )
#         ordered_contribution_difference = (
#             geo_difference.sign() * (contribution_right - contribution_left)
#         )
#         pair_loss = F.relu(float(margin) - ordered_contribution_difference)
#         if pair_mask.any():
#             losses.append(pair_loss[pair_mask].mean())
#     if not losses:
#         return contribution.sum() * 0.0
#     return torch.stack(losses).mean()
import torch
import torch.nn.functional as F


def geometry_contribution_rank_loss(
    contribution,
    geometry_score,
    valid,
    event_voxel,
    *,
    margin=0.05,
    difference_threshold=0.10,
):
    """
    Enforce relative ordering between geometry score and contribution:

        G_i > G_j  =>  C_i > C_j + margin

    Compared with the original implementation, this version uses
    both adjacent and slightly longer-range pixel pairs.
    """

    # Geometry score is supervision only.
    geometry_score = geometry_score.detach()

    # [B, V, C, H, W] -> [B, V, H, W]
    event_support = event_voxel.float().abs().sum(dim=2) > 0.0
    eligible = valid.bool() & event_support

    losses = []

    # Compare pixels separated by 1 and 4 pixels.
    for offset in (1, 4):
        # Horizontal and vertical directions.
        for dimension in (-1, -2):
            spatial_size = geometry_score.shape[dimension]

            if offset >= spatial_size:
                continue

            length = spatial_size - offset

            geo_a = geometry_score.narrow(
                dimension, 0, length
            )
            geo_b = geometry_score.narrow(
                dimension, offset, length
            )

            contribution_a = contribution.narrow(
                dimension, 0, length
            )
            contribution_b = contribution.narrow(
                dimension, offset, length
            )

            eligible_a = eligible.narrow(
                dimension, 0, length
            )
            eligible_b = eligible.narrow(
                dimension, offset, length
            )

            geometry_difference = geo_b - geo_a

            pair_mask = (
                eligible_a
                & eligible_b
                & (
                    geometry_difference.abs()
                    > float(difference_threshold)
                )
            )

            if not pair_mask.any():
                continue

            # Positive when contribution follows geometry ordering.
            ordered_contribution_difference = (
                geometry_difference.sign()
                * (contribution_b - contribution_a)
            )

            pair_loss = F.relu(
                float(margin)
                - ordered_contribution_difference
            )

            losses.append(pair_loss[pair_mask].mean())

    if not losses:
        return contribution.sum() * 0.0

    return torch.stack(losses).mean()

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
        geometry_rank_weight=0.10,
        geometry_rank_margin=0.05,
        geometry_rank_threshold=0.10,
        event_normal_weight=0.0,
        depth_event_normal_weight=0.0,
        depth_gradient_weight=0.5,
        depth_curvature_weight=0.1,
        patch_grid_weight=0.25,
        grid_patch_size=14,
        points_loss_type="l1",
    ):
        self.depth_weight = float(depth_weight)
        self.normal_weight = float(normal_weight)
        self.bridge_beta = float(bridge_beta)
        self.budget_weight = float(budget_weight)
        self.pair_weight = float(pair_weight)
        self.update_weight = float(update_weight)
        self.decomposition_weight = float(decomposition_weight)
        self.geometry_rank_weight = float(geometry_rank_weight)
        self.geometry_rank_margin = float(geometry_rank_margin)
        self.geometry_rank_threshold = float(geometry_rank_threshold)
        self.event_normal_weight = float(event_normal_weight)
        self.depth_event_normal_weight = float(depth_event_normal_weight)
        self.depth_gradient_weight = float(depth_gradient_weight)
        self.depth_curvature_weight = float(depth_curvature_weight)
        self.patch_grid_weight = float(patch_grid_weight)
        self.grid_patch_size = int(grid_patch_size)
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
        depth_gradient_loss, depth_curvature_loss, patch_grid_loss = (
            supervised_log_depth_derivative_losses(
                depth_pred,
                depth_gt,
                valid,
                patch_size=self.grid_patch_size,
            )
        )

        normal_pred = fe.depth_to_normals(depth_pred, intrinsics)
        normal_gt = fe.depth_to_normals(depth_gt, intrinsics)
        geometry_score = (
            geometry_emphasis_weight(depth_gt, normal_gt, valid, alpha=2.0) - 1.0
        ) / 2.0
        normal_valid = fe.normal_stencil_valid_mask(valid, depth_pred, eps=1.0e-6)
        pred_unit = F.normalize(normal_pred.float(), dim=-1, eps=1.0e-6)
        gt_unit = F.normalize(normal_gt.float(), dim=-1, eps=1.0e-6)
        normal_error = 1.0 - (pred_unit * gt_unit).sum(dim=-1).clamp(-1.0, 1.0)
        normal_loss = _weighted_mean(normal_error, weight * normal_valid.float())
        event_normal_loss = depth_loss.new_zeros(())
        depth_event_normal_loss = depth_loss.new_zeros(())
        event_normal = None
        event_normal_valid = None
        if output.ress and all("event_normal" in item for item in output.ress):
            event_normal = torch.stack(
                [item["event_normal"] for item in output.ress], dim=1
            )
            event_unit = F.normalize(event_normal.float(), dim=-1, eps=1.0e-6)
            event_support = torch.stack(
                [item["event_normal_support"] for item in output.ress], dim=1
            ).bool()
            event_reliability = torch.stack(
                [item["event_normal_reliability"] for item in output.ress], dim=1
            ).float()
            event_normal_valid = normal_valid & event_support
            # Do not let C reduce its own supervision weight to escape the
            # normal objective. Reliability still emphasizes confident events,
            # while every active event keeps a nonzero supervision floor.
            event_weight = (
                weight
                * normal_valid.float()
                * event_support.float()
                * (0.25 + 0.75 * event_reliability.detach())
            )
            event_normal_error = 1.0 - (
                event_unit * gt_unit
            ).sum(dim=-1).clamp(-1.0, 1.0)
            event_normal_loss = _weighted_mean(event_normal_error, event_weight)
            # The direct event-normal branch is anchored by GT above. Detach it
            # here so depth must follow the event normal instead of both heads
            # moving toward an arbitrary agreement solution.
            depth_event_error = 1.0 - (
                pred_unit * event_unit.detach()
            ).sum(dim=-1).clamp(-1.0, 1.0)
            depth_event_normal_loss = _weighted_mean(depth_event_error, event_weight)
        geometry_loss = (
            point_loss
            + self.depth_weight * depth_loss
            + self.normal_weight * normal_loss
            + self.event_normal_weight * event_normal_loss
            + self.depth_event_normal_weight * depth_event_normal_loss
            + self.depth_gradient_weight * depth_gradient_loss
            + self.depth_curvature_weight * depth_curvature_loss
            + self.patch_grid_weight * patch_grid_loss
        )
        return geometry_loss, {
            "depth": depth_loss,
            "normal": normal_loss,
            "point": depth_loss.new_tensor(point_details["points_loss"]),
            "event_normal": event_normal_loss,
            "depth_event_normal": depth_event_normal_loss,
            "depth_gradient": depth_gradient_loss,
            "depth_curvature": depth_curvature_loss,
            "patch_grid": patch_grid_loss,
        }, {
            **point_aux,
            "depth_pred_live": depth_pred,
            "valid_live": valid,
            "geometry_score": geometry_score,
            "normal_pred_live": normal_pred,
            "normal_gt_live": normal_gt,
            "normal_valid_live": normal_valid,
            "event_normal_live": event_normal,
            "event_normal_valid_live": event_normal_valid,
        }

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
        geometry_rank_loss = (
            geometry_contribution_rank_loss(
                contribution_spatial.float(),
                aux["geometry_score"],
                aux["valid_live"],
                event_voxel,
                margin=self.geometry_rank_margin,
                difference_threshold=self.geometry_rank_threshold,
            )
            if self.geometry_rank_weight > 0.0
            else contribution.new_zeros(())
        )
        total = (
            geometry_loss
            + self.budget_weight * budget_loss
            + self.pair_weight * pair_loss
            + self.update_weight * update_loss
            + self.decomposition_weight * decomposition_loss
            + self.geometry_rank_weight * geometry_rank_loss
        )
        details = {
            **geometry_details,
            "geometry": geometry_loss,
            "budget": budget_loss,
            "pair": pair_loss,
            "update": update_loss,
            "decomposition": decomposition_loss,
            "geometry_rank": geometry_rank_loss,
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
    "geometry_contribution_rank_loss",
    "supervised_log_depth_derivative_losses",
]
