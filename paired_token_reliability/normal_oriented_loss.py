"""Additional normal/local-invariance losses for the normal-oriented model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss, _weighted_mean


def normal_gradient_loss(pred, target, valid):
    pred_dx, target_dx = pred[..., :, 1:, :] - pred[..., :, :-1, :], target[..., :, 1:, :] - target[..., :, :-1, :]
    pred_dy, target_dy = pred[..., 1:, :, :] - pred[..., :-1, :, :], target[..., 1:, :, :] - target[..., :-1, :, :]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
    return 0.5 * (
        _weighted_mean((pred_dx - target_dx).abs().mean(-1), valid_dx)
        + _weighted_mean((pred_dy - target_dy).abs().mean(-1), valid_dy)
    )


class NormalOrientedGeometryLoss(UnifiedGeometryContributionLoss):
    def __init__(self, *, normal_gradient_weight=0.5, depth_normal_consistency_weight=0.5,
                 depth_outside_support_weight=0.2, detach_normal_target=True, **kwargs):
        super().__init__(**kwargs)
        self.normal_gradient_weight = float(normal_gradient_weight)
        self.depth_normal_consistency_weight = float(depth_normal_consistency_weight)
        self.depth_outside_support_weight = float(depth_outside_support_weight)
        self.detach_normal_target = bool(detach_normal_target)

    def geometry(self, output, views, bridge_mask):
        base, details, aux = super().geometry(output, views, bridge_mask)
        if not output.ress or not all("event_normal" in item for item in output.ress):
            zero = base.new_zeros(())
            details.update(normal_gradient=zero, depth_normal_consistency=zero,
                           depth_outside_support=zero)
            return base, details, aux
        normal = torch.stack([item["event_normal"] for item in output.ress], 1)
        support = torch.stack([item["event_normal_reliability"] for item in output.ress], 1).float()
        valid = aux["normal_valid_live"].bool() & (support > 0)
        normal_grad = normal_gradient_loss(normal.float(), aux["normal_gt_live"].float(), valid)
        depth_normal = aux["normal_pred_live"].float()
        target = normal.detach() if self.detach_normal_target else normal
        consistency = _weighted_mean(
            1.0 - (F.normalize(depth_normal, dim=-1, eps=1e-6) *
                   F.normalize(target.float(), dim=-1, eps=1e-6)).sum(-1).clamp(-1, 1),
            valid.float() * support,
        )
        coarse = torch.stack([item["depth_coarse"] for item in output.ress], 1).squeeze(-1)
        depth_outside = _weighted_mean(
            (1.0 - support) * (aux["depth_pred_live"].float() - coarse.float()).abs(),
            aux["valid_live"],
        )
        total = (base + self.normal_gradient_weight * normal_grad
                 + self.depth_normal_consistency_weight * consistency
                 + self.depth_outside_support_weight * depth_outside)
        details.update(normal_gradient=normal_grad, depth_normal_consistency=consistency,
                       depth_outside_support=depth_outside)
        return total, details, aux


__all__ = ["NormalOrientedGeometryLoss", "normal_gradient_loss"]
