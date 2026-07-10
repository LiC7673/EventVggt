"""Stage-2 geometry loss plus a light low-contribution adapter constraint."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

import finetune_event as fe


class GeometryAdapterStage2Loss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        adapter_update_weight: float = 0.01,
        saturation_boost: float = 1.0,
        saturation_threshold: float = 0.98,
        saturation_normal_weight: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.adapter_update_weight = float(adapter_update_weight)
        self.saturation_boost = float(saturation_boost)
        self.saturation_threshold = float(saturation_threshold)
        self.saturation_normal_weight = float(saturation_normal_weight)

    @staticmethod
    def _weighted_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(device=value.device, dtype=value.dtype)
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _gt_normals(views, depth_gt, intrinsics):
        if all("normal" in view for view in views):
            normals = fe.stack_view_field(views, "normal").to(depth_gt)
            if normals.ndim == 5 and normals.shape[2] == 3:
                normals = normals.permute(0, 1, 3, 4, 2)
            return normals
        return fe.depth_to_normals(depth_gt, intrinsics)

    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]) -> Tuple:
        base_loss, details, aux = super().forward(model_output, views)
        predictions = model_output.ress
        depth_pred = torch.stack([item["depth"] for item in predictions], dim=1).squeeze(-1)
        depth_gt = fe.stack_view_field(views, "depthmap").to(depth_pred)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(depth_pred)
        valid = fe.build_valid_mask(
            views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max
        )
        images = fe.stack_view_field(views, "img").to(depth_pred)
        saturated = (images.amax(dim=2) >= self.saturation_threshold) & valid

        log_depth_error = (
            depth_pred.clamp_min(self.depth_min).log()
            - depth_gt.clamp_min(self.depth_min).log()
        ).abs()
        saturated_depth_loss = self._weighted_mean(log_depth_error, saturated)

        normal_gt = self._gt_normals(views, depth_gt, intrinsics)
        normal_pred = fe.depth_to_normals(depth_pred, intrinsics)
        normal_valid = fe.normal_stencil_valid_mask(valid, depth_pred, eps=self.depth_min)
        saturated_normal_loss = fe.masked_cosine_loss(
            normal_pred, normal_gt, normal_valid & saturated
        )

        update_loss = predictions[0].get("adapter_update_loss", depth_pred.new_zeros(()))
        total = (
            base_loss
            + self.saturation_boost
            * (
                self.depth_weight * saturated_depth_loss
                + self.saturation_normal_weight * saturated_normal_loss
            )
            + self.adapter_update_weight * update_loss
        )
        details.update(
            {
                "saturated_depth_loss": float(saturated_depth_loss.detach()),
                "saturated_normal_loss": float(saturated_normal_loss.detach()),
                "adapter_update_loss": float(update_loss.detach()),
                "saturated_ratio": float(saturated.float().mean().detach()),
                "adapter_depth_alpha_abs": float(
                    predictions[0]["adapter_alpha_depth"].detach().abs().mean()
                ),
                "adapter_point_alpha_abs": float(
                    predictions[0]["adapter_alpha_point"].detach().abs().mean()
                ),
                "adapter_depth_update_abs": float(
                    predictions[0]["adapter_depth_update_magnitudes"].detach().mean()
                ),
                "adapter_point_update_abs": float(
                    predictions[0]["adapter_point_update_magnitudes"].detach().mean()
                ),
            }
        )
        aux["event_contribution"] = torch.stack(
            [item["event_contribution"] for item in predictions], dim=1
        ).detach()
        aux["event_reliability"] = aux["event_contribution"]
        aux["depth_coarse"] = torch.stack(
            [item["depth_coarse"] for item in predictions], dim=1
        ).squeeze(-1).detach()
        return total, details, aux


def make_geometry_adapter_loss(cfg):
    class ConfiguredGeometryAdapterLoss(GeometryAdapterStage2Loss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                adapter_update_weight=float(getattr(cfg.loss, "adapter_update_weight", 0.01)),
                saturation_boost=float(getattr(cfg.loss, "adapter_saturation_boost", 1.0)),
                saturation_threshold=float(
                    getattr(cfg.loss, "adapter_saturation_threshold", 0.98)
                ),
                saturation_normal_weight=float(
                    getattr(cfg.loss, "adapter_saturation_normal_weight", 0.25)
                ),
                **kwargs,
            )

    return ConfiguredGeometryAdapterLoss


__all__ = ["GeometryAdapterStage2Loss", "make_geometry_adapter_loss"]
