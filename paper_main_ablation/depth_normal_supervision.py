"""Depth-derived normal supervision shared by every main-table row.

The dataset may expose rendered normals whose convention differs from normals
computed from metric depth.  This loss always derives both the target normal
and its spatial gradients from GT depth, so it suppresses event-texture noise
without smoothing away genuine GT geometry transitions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import finetune_event as fe


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class DepthDerivedNormalSupervisionMixin:
    def _init_depth_normal_supervision(
        self,
        *,
        normal_weight: float,
        normal_gradient_weight: float,
    ) -> None:
        self.depth_normal_weight = float(normal_weight)
        self.depth_normal_gradient_weight = float(normal_gradient_weight)

    def forward(self, model_output, views):
        total_loss, details, aux = super().forward(model_output, views)
        if self.depth_normal_weight <= 0.0 and self.depth_normal_gradient_weight <= 0.0:
            details.update(
                {
                    "depth_gt_normal_loss": 0.0,
                    "depth_gt_normal_gradient_loss": 0.0,
                    "depth_gt_normal_total": 0.0,
                }
            )
            return total_loss, details, aux

        depth_pred = torch.stack(
            [result["depth"] for result in model_output.ress], dim=1
        ).squeeze(-1)
        depth_gt = fe.stack_view_field(views, "depthmap").to(
            device=depth_pred.device, dtype=depth_pred.dtype
        )
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(
            device=depth_pred.device, dtype=depth_pred.dtype
        )
        valid = fe.build_valid_mask(
            views,
            depth_gt,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        )
        pred_normal = fe.depth_to_normals(depth_pred.clamp_min(self.depth_min), intrinsics)
        gt_normal = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics).detach()

        normal_valid = valid.clone()
        normal_valid[..., 0, :] = False
        normal_valid[..., -1, :] = False
        normal_valid[..., :, 0] = False
        normal_valid[..., :, -1] = False
        cosine = 1.0 - (
            F.normalize(pred_normal, dim=-1, eps=1.0e-6)
            * F.normalize(gt_normal, dim=-1, eps=1.0e-6)
        ).sum(dim=-1).clamp(-1.0, 1.0)
        normal_loss = _masked_mean(cosine, normal_valid)

        pred_dx = pred_normal[..., :, 1:, :] - pred_normal[..., :, :-1, :]
        gt_dx = gt_normal[..., :, 1:, :] - gt_normal[..., :, :-1, :]
        pred_dy = pred_normal[..., 1:, :, :] - pred_normal[..., :-1, :, :]
        gt_dy = gt_normal[..., 1:, :, :] - gt_normal[..., :-1, :, :]
        dx_valid = normal_valid[..., :, 1:] & normal_valid[..., :, :-1]
        dy_valid = normal_valid[..., 1:, :] & normal_valid[..., :-1, :]
        dx_loss = _masked_mean((pred_dx - gt_dx).abs().mean(dim=-1), dx_valid)
        dy_loss = _masked_mean((pred_dy - gt_dy).abs().mean(dim=-1), dy_valid)
        gradient_loss = 0.5 * (dx_loss + dy_loss)

        extra = (
            self.depth_normal_weight * normal_loss
            + self.depth_normal_gradient_weight * gradient_loss
        )
        total_loss = total_loss + extra
        details.update(
            {
                "depth_gt_normal_loss": float(normal_loss.detach()),
                "depth_gt_normal_gradient_loss": float(gradient_loss.detach()),
                "depth_gt_normal_total": float(extra.detach()),
                "extra_loss_total": float(details.get("extra_loss_total", 0.0))
                + float(extra.detach()),
                "total_loss_with_extra": float(total_loss.detach()),
            }
        )
        return total_loss, details, aux


def wrap_depth_normal_supervision(base_loss, cfg):
    class ConfiguredDepthNormalLoss(DepthDerivedNormalSupervisionMixin, base_loss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_depth_normal_supervision(
                normal_weight=float(cfg.loss.depth_gt_normal_weight),
                normal_gradient_weight=float(cfg.loss.depth_gt_normal_gradient_weight),
            )

    return ConfiguredDepthNormalLoss


__all__ = ["wrap_depth_normal_supervision"]
