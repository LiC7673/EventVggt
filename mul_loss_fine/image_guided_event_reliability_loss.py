"""Image-guided reliability supervision for simple temporal-event detail.

Unlike the V2 reliability losses, this does not use reverse/swap event
counterfactuals.  It builds a soft target from three stable cues:

* event support: only supervise pixels where events are present;
* GT geometry detail: reliable events should coincide with geometric detail;
* RGB edge/saturation: image edges support reliability, saturated non-geometry
  event regions are more likely to be highlight/noise events.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F

import finetune_event as fe
from mul_loss_fine.event_supported_mv_loss import (
    _make_event_support,
    _normal_gradient_magnitude,
    _normalize_detail_support,
    _stack_output_field,
    _weighted_mean,
)
from mul_loss_fine.launcher import make_configured_loss


def _image_edge_and_saturation(
    views: List[Dict[str, torch.Tensor]],
    *,
    valid_mask: torch.Tensor,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    images = fe.stack_view_field(views, "img").to(device=valid_mask.device, dtype=dtype)
    if images.detach().amin() < -0.05:
        images01 = ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        images01 = images.clamp(0.0, 1.0)

    gray = images01.mean(dim=2, keepdim=True)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    edge = torch.sqrt((dx.square() + dy.square()).clamp_min(1e-12))
    edge_support = _normalize_detail_support(edge, valid_mask, threshold=0.02, power=1.0)

    sat_high = (images01 > 0.97).to(dtype=dtype).mean(dim=2, keepdim=True)
    sat_low = (images01 < 0.03).to(dtype=dtype).mean(dim=2, keepdim=True)
    saturation = (sat_high + sat_low).clamp(0.0, 1.0)
    saturation = F.avg_pool2d(
        saturation.flatten(0, 1),
        kernel_size=5,
        stride=1,
        padding=2,
    ).view_as(saturation)
    saturation = saturation * valid_mask.unsqueeze(2).to(dtype=dtype)
    return {"edge": edge_support.detach(), "saturation": saturation.detach()}


class ImageGuidedEventReliabilityLossMixin:
    def _init_image_guided_event_reliability_loss(
        self,
        *,
        reliability_weight: float,
        reject_weight: float,
        geometry_threshold: float,
        event_threshold: float,
        image_support_floor: float,
        saturation_reject_boost: float,
    ) -> None:
        self.img_rel_weight = float(reliability_weight)
        self.img_rel_reject_weight = float(reject_weight)
        self.img_rel_geometry_threshold = float(geometry_threshold)
        self.img_rel_event_threshold = float(event_threshold)
        self.img_rel_image_support_floor = min(max(float(image_support_floor), 0.0), 1.0)
        self.img_rel_saturation_reject_boost = float(saturation_reject_boost)

    def forward(self, model_output, views):
        total_loss, details, aux = super().forward(model_output, views)
        reliability = _stack_output_field(model_output, "event_reliability")
        if reliability is None or self.img_rel_weight <= 0.0:
            details.update(
                {
                    "img_event_rel_loss": 0.0,
                    "img_event_rel_bce_loss": 0.0,
                    "img_event_rel_reject_loss": 0.0,
                    "img_event_rel_target_mean": 0.0,
                    "img_event_rel_pos_mean": 0.0,
                    "img_event_rel_neg_mean": 0.0,
                    "img_event_weight_mean": 0.0,
                    "img_edge_support_mean": 0.0,
                    "img_saturation_mean": 0.0,
                }
            )
            return total_loss, details, aux

        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)
        dtype = depth_pred.dtype
        device = depth_pred.device
        batch, seq_len, height, width = depth_pred.shape
        reliability = reliability.to(device=device, dtype=dtype)

        depth_gt = fe.stack_view_field(views, "depthmap").to(device=device, dtype=dtype)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=device, dtype=dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)
        valid_weight = valid_mask.unsqueeze(2).to(dtype=dtype)

        gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics)
        gt_grad = _normal_gradient_magnitude(gt_normals.flatten(0, 1)).view(batch, seq_len, 1, height, width)
        geo_support = _normalize_detail_support(
            gt_grad,
            valid_mask,
            threshold=self.img_rel_geometry_threshold,
            power=1.0,
        ).detach()

        event_support = _make_event_support(
            views,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            blur_kernel=self.mv_loss.event_blur_kernel,
            dilate_kernel=self.mv_loss.event_dilate_kernel,
            threshold=self.img_rel_event_threshold,
            power=self.mv_loss.event_power,
            top_fraction=self.mv_loss.event_top_fraction,
            mode=self.mv_loss.event_support_mode,
        ).unsqueeze(2).detach()
        image_cues = _image_edge_and_saturation(views, valid_mask=valid_mask, dtype=dtype)
        image_support = image_cues["edge"]
        saturation = image_cues["saturation"]

        image_factor = self.img_rel_image_support_floor + (
            1.0 - self.img_rel_image_support_floor
        ) * image_support
        reliable_target = (geo_support * image_factor).clamp(0.0, 1.0).detach()

        event_weight = valid_weight * event_support
        positive_weight = event_weight * (0.10 + geo_support)
        bce_loss = _weighted_mean(
            F.binary_cross_entropy(
                reliability.clamp(1e-5, 1.0 - 1e-5).unsqueeze(2),
                reliable_target,
                reduction="none",
            ),
            positive_weight,
        )

        non_geometry = (1.0 - geo_support).detach()
        reject_weight = event_weight * non_geometry * (
            1.0 + self.img_rel_saturation_reject_boost * saturation
        )
        reject_loss = _weighted_mean(reliability.unsqueeze(2), reject_weight)
        rel_loss = self.img_rel_weight * bce_loss + self.img_rel_reject_weight * reject_loss
        total_loss = total_loss + rel_loss

        pos_mask = event_weight * (geo_support > 0.5).to(dtype=dtype)
        neg_mask = reject_weight
        details.update(
            {
                "img_event_rel_loss": float(rel_loss.detach()),
                "img_event_rel_bce_loss": float(bce_loss.detach()),
                "img_event_rel_reject_loss": float(reject_loss.detach()),
                "img_event_rel_target_mean": float(_weighted_mean(reliable_target, valid_weight).detach()),
                "img_event_rel_pos_mean": float(_weighted_mean(reliability.unsqueeze(2), pos_mask).detach()),
                "img_event_rel_neg_mean": float(_weighted_mean(reliability.unsqueeze(2), neg_mask).detach()),
                "img_event_weight_mean": float(_weighted_mean(event_support, valid_weight).detach()),
                "img_edge_support_mean": float(_weighted_mean(image_support, valid_weight).detach()),
                "img_saturation_mean": float(_weighted_mean(saturation, valid_weight).detach()),
                "extra_loss_total": float(details.get("extra_loss_total", 0.0)) + float(rel_loss.detach()),
                "total_loss_with_extra": float(total_loss.detach()),
            }
        )
        aux["event_reliability"] = reliability.detach()
        aux["img_event_reliability_target"] = reliable_target.squeeze(2).detach()
        return total_loss, details, aux


def make_configured_image_guided_event_reliability_loss(cfg):
    configured_base = make_configured_loss(cfg)

    class ConfiguredImageGuidedEventReliabilityLoss(
        ImageGuidedEventReliabilityLossMixin,
        configured_base,
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_image_guided_event_reliability_loss(
                reliability_weight=float(getattr(cfg.loss, "img_event_reliability_weight", 0.25)),
                reject_weight=float(getattr(cfg.loss, "img_event_reject_weight", 0.10)),
                geometry_threshold=float(getattr(cfg.loss, "img_event_geometry_threshold", 0.02)),
                event_threshold=float(getattr(cfg.loss, "img_event_event_threshold", 0.20)),
                image_support_floor=float(getattr(cfg.loss, "img_event_image_support_floor", 0.35)),
                saturation_reject_boost=float(getattr(cfg.loss, "img_event_saturation_reject_boost", 1.5)),
            )

    return ConfiguredImageGuidedEventReliabilityLoss
