"""Geometry-contribution supervision for temporal event reliability."""

from typing import Dict, List

import torch
import torch.nn.functional as F

import finetune_event as fe
from mul_loss_fine.event_supported_mv_loss import (
    _normal_gradient_magnitude,
    _normalize_detail_support,
    _stack_output_field,
    _weighted_mean,
)
from mul_loss_fine.reliability_residual_v2_loss import (
    TemporalReliabilityV2LossMixin,
    _batch_group_keys,
)
from mul_loss_fine.launcher import make_configured_loss


def _format_ldr(value) -> str:
    value = str(value)
    return value if value.startswith("ev_") else f"ev_{value}"


def _numeric_ldr_id(value: str) -> int:
    value = str(value)
    if value.startswith("ev_"):
        value = value[3:]
    try:
        return int(value)
    except ValueError:
        return -10**9


def _batch_ldr_ids(views: List[Dict[str, torch.Tensor]], batch_size: int):
    value = views[0].get("ldr_event_id", "")
    if isinstance(value, (list, tuple)):
        return [str(value[idx]) for idx in range(batch_size)]
    return [str(value) for _ in range(batch_size)]


class GeometryContributionEventLossMixin(TemporalReliabilityV2LossMixin):
    """Learn which event pixels are useful for geometry, then transfer to LDR."""

    def _init_geo_contribution_loss(
        self,
        *,
        teacher_ldr_id: str,
        geo_target_weight: float,
        geo_reject_weight: float,
        teacher_consistency_weight: float,
        event_delta_weight: float,
        geo_teacher_boost: float,
        geo_detail_threshold: float,
        geo_positive_floor: float,
        geo_negative_margin: float,
    ):
        self.geo_teacher_ldr_id = _format_ldr(teacher_ldr_id)
        self.geo_target_weight = float(geo_target_weight)
        self.geo_reject_weight = float(geo_reject_weight)
        self.geo_teacher_consistency_weight = float(teacher_consistency_weight)
        self.geo_event_delta_weight = float(event_delta_weight)
        self.geo_teacher_boost = float(geo_teacher_boost)
        self.geo_detail_threshold = float(geo_detail_threshold)
        self.geo_positive_floor = min(max(float(geo_positive_floor), 0.0), 1.0)
        self.geo_negative_margin = min(max(float(geo_negative_margin), 0.0), 1.0)

    def forward(self, model_output, views):
        total_loss, details, aux = super().forward(model_output, views)
        reliability = _stack_output_field(model_output, "event_reliability")
        temporal_quality = _stack_output_field(model_output, "event_temporal_quality")
        presence = _stack_output_field(model_output, "event_presence")
        depth_coarse = _stack_output_field(model_output, "depth_coarse")
        event_delta = _stack_output_field(model_output, "event_delta_log")
        rgb_delta = _stack_output_field(model_output, "rgb_delta_log")
        if reliability is None or presence is None or depth_coarse is None:
            details.update(
                {
                    "geo_event_loss": 0.0,
                    "geo_event_target_loss": 0.0,
                    "geo_event_reject_loss": 0.0,
                    "geo_event_delta_loss": 0.0,
                    "geo_teacher_consistency_loss": 0.0,
                    "geo_event_target_mean": 0.0,
                    "geo_event_reliability_pos_mean": 0.0,
                    "geo_event_reliability_neg_mean": 0.0,
                    "geo_event_delta_abs": 0.0,
                    "geo_rgb_delta_abs": 0.0,
                    "geo_teacher_pair_count": 0.0,
                }
            )
            return total_loss, details, aux

        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)
        dtype = depth_pred.dtype
        device = depth_pred.device
        batch, seq_len, height, width = depth_pred.shape
        reliability = reliability.to(device=device, dtype=dtype)
        presence = presence.to(device=device, dtype=dtype).detach().clamp(0.0, 1.0)
        if temporal_quality is None:
            temporal_quality = torch.ones_like(reliability)
        temporal_quality = temporal_quality.to(device=device, dtype=dtype).detach().clamp(0.0, 1.0)
        depth_coarse = depth_coarse.to(device=device, dtype=dtype)
        event_delta = event_delta.to(device=device, dtype=dtype) if event_delta is not None else None
        rgb_delta = rgb_delta.to(device=device, dtype=dtype) if rgb_delta is not None else None

        depth_gt = fe.stack_view_field(views, "depthmap").to(device=device, dtype=dtype)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=device, dtype=dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)
        valid_weight = valid_mask.unsqueeze(2).to(dtype=dtype)

        gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics)
        coarse_normals = fe.depth_to_normals(depth_coarse.clamp_min(self.depth_min), intrinsics)
        gt_gradient = _normal_gradient_magnitude(gt_normals.flatten(0, 1)).view(
            batch, seq_len, 1, height, width
        )
        detail_support = _normalize_detail_support(
            gt_gradient,
            valid_mask,
            threshold=self.geo_detail_threshold,
            power=1.0,
        ).detach()
        signed_target_delta = (
            torch.log(depth_gt.clamp_min(self.depth_min))
            - torch.log(depth_coarse.detach().clamp_min(self.depth_min))
        ).clamp(-self.v2_residual_scale, self.v2_residual_scale)
        target_strength = (signed_target_delta.abs() / self.v2_residual_scale).unsqueeze(2).clamp(0.0, 1.0)
        coarse_cos_error = 1.0 - (
            F.normalize(coarse_normals.detach(), dim=-1, eps=1e-6)
            * F.normalize(gt_normals.detach(), dim=-1, eps=1e-6)
        ).sum(dim=-1).clamp(-1.0, 1.0)
        normal_need = _normalize_detail_support(
            coarse_cos_error.unsqueeze(2),
            valid_mask,
            threshold=0.0,
            power=1.0,
        )
        geometry_need = torch.maximum(target_strength, normal_need)
        geometry_need = (geometry_need * (0.20 + 0.80 * detail_support)).detach().clamp(0.0, 1.0)

        # Temporal quality is itself a learned/heuristic event statistic and can
        # be very small at the beginning of training.  If we multiply the geo
        # teacher target by it directly, the teacher supervision almost vanishes
        # exactly in the cases where we need to learn a better reliability map.
        quality_weight = self.v2_temporal_quality_floor + (
            1.0 - self.v2_temporal_quality_floor
        ) * temporal_quality
        event_weight = presence.unsqueeze(2) * quality_weight.unsqueeze(2)
        geo_target = (self.geo_positive_floor + (1.0 - self.geo_positive_floor) * geometry_need).detach()
        target_weight = valid_weight * event_weight * (0.10 + geometry_need)

        teacher_ids = _batch_ldr_ids(views, batch)
        teacher_mask = torch.tensor(
            [ldr_id == self.geo_teacher_ldr_id for ldr_id in teacher_ids],
            device=device,
            dtype=dtype,
        ).view(batch, 1, 1, 1, 1)
        sample_boost = 1.0 + self.geo_teacher_boost * teacher_mask
        target_loss = _weighted_mean(
            F.binary_cross_entropy(
                reliability.clamp(1e-5, 1.0 - 1e-5).unsqueeze(2),
                geo_target,
                reduction="none",
            ),
            target_weight * sample_boost,
        )

        non_geometry = (1.0 - geometry_need).detach()
        reject_weight = valid_weight * presence.unsqueeze(2) * quality_weight.unsqueeze(2) * non_geometry.square()
        reject_map = F.relu(reliability.unsqueeze(2) - self.geo_negative_margin)
        reject_loss = _weighted_mean(reject_map, reject_weight)

        groups = {}
        for batch_idx, key in enumerate(_batch_group_keys(views, batch)):
            groups.setdefault(key, []).append(batch_idx)
        consistency_terms = []
        pair_count = 0
        for indices in groups.values():
            teacher_candidates = [idx for idx in indices if teacher_ids[idx] == self.geo_teacher_ldr_id]
            if teacher_candidates:
                teacher_idx = teacher_candidates[0]
            else:
                teacher_idx = max(indices, key=lambda idx: _numeric_ldr_id(teacher_ids[idx]))
            teacher_rel = reliability[teacher_idx].detach()
            teacher_need = geometry_need[teacher_idx].detach()
            for idx in indices:
                if idx == teacher_idx:
                    continue
                pair_weight = (
                    valid_mask[idx].unsqueeze(0).to(dtype=dtype)
                    * presence[idx].unsqueeze(0).detach()
                    * (0.10 + teacher_need.squeeze(1))
                )
                consistency_terms.append(
                    _weighted_mean((reliability[idx] - teacher_rel).abs().unsqueeze(0), pair_weight)
                )
                pair_count += 1
        consistency_loss = torch.stack(consistency_terms).mean() if consistency_terms else depth_pred.new_tensor(0.0)
        event_delta_loss = depth_pred.new_tensor(0.0)
        event_delta_abs = depth_pred.new_tensor(0.0)
        rgb_delta_abs = depth_pred.new_tensor(0.0)
        if event_delta is not None:
            if rgb_delta is not None:
                event_target = (signed_target_delta - rgb_delta.detach()).clamp(
                    -self.v2_residual_scale,
                    self.v2_residual_scale,
                )
                rgb_delta_abs = _weighted_mean(rgb_delta.abs().unsqueeze(2), valid_weight)
            else:
                event_target = signed_target_delta
            event_delta_weight = valid_weight * event_weight * (0.10 + geometry_need)
            event_delta_loss = _weighted_mean(
                ((event_delta - event_target).abs() / self.v2_residual_scale).unsqueeze(2),
                event_delta_weight * sample_boost,
            )
            event_delta_abs = _weighted_mean(event_delta.abs().unsqueeze(2), valid_weight)

        geo_event_loss = (
            self.geo_target_weight * target_loss
            + self.geo_reject_weight * reject_loss
            + self.geo_teacher_consistency_weight * consistency_loss
            + self.geo_event_delta_weight * event_delta_loss
        )
        total_loss = total_loss + geo_event_loss

        pos_weight = valid_weight * event_weight * (geometry_need > 0.5).to(dtype=dtype)
        neg_weight = reject_weight
        aux["event_reliability"] = reliability.detach()
        aux["event_presence"] = presence.detach()
        aux["event_temporal_quality"] = temporal_quality.detach()
        details.update(
            {
                "geo_event_loss": float(geo_event_loss.detach()),
                "geo_event_target_loss": float(target_loss.detach()),
                "geo_event_reject_loss": float(reject_loss.detach()),
                "geo_event_delta_loss": float(event_delta_loss.detach()),
                "geo_teacher_consistency_loss": float(consistency_loss.detach()),
                "geo_event_target_mean": float(_weighted_mean(geo_target, valid_weight).detach()),
                "geo_event_weight_mean": float(_weighted_mean(event_weight, valid_weight).detach()),
                "geo_temporal_quality_weight_mean": float(
                    _weighted_mean(quality_weight.unsqueeze(2), valid_weight).detach()
                ),
                "geo_event_reliability_pos_mean": float(
                    _weighted_mean(reliability.unsqueeze(2), pos_weight).detach()
                ),
                "geo_event_reliability_neg_mean": float(
                    _weighted_mean(reliability.unsqueeze(2), neg_weight).detach()
                ),
                "geo_event_delta_abs": float(event_delta_abs.detach()),
                "geo_rgb_delta_abs": float(rgb_delta_abs.detach()),
                "geo_teacher_pair_count": float(pair_count),
                "extra_loss_total": float(details.get("extra_loss_total", 0.0)) + float(geo_event_loss.detach()),
                "total_loss_with_extra": float(total_loss.detach()),
            }
        )
        return total_loss, details, aux


def make_configured_geo_contribution_loss(cfg):
    configured_base = make_configured_loss(cfg)

    class ConfiguredGeometryContributionEventLoss(GeometryContributionEventLossMixin, configured_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_reliability_v2_loss(
                residual_scale=float(getattr(cfg.model, "refiner_residual_scale", 0.015)),
                residual_target_weight=float(getattr(cfg.loss, "v2_residual_target_weight", 0.40)),
                gate_reliability_weight=float(getattr(cfg.loss, "v2_gate_reliability_weight", 0.10)),
                gate_need_floor=float(getattr(cfg.loss, "v2_gate_need_floor", 0.10)),
                gate_positive_boost=float(getattr(cfg.loss, "v2_gate_positive_boost", 2.0)),
                ldr_depth_weight=float(getattr(cfg.loss, "v2_ldr_final_depth_weight", 0.10)),
                ldr_normal_weight=float(getattr(cfg.loss, "v2_ldr_final_normal_weight", 0.10)),
                ldr_correction_weight=float(getattr(cfg.loss, "v2_ldr_correction_weight", 0.20)),
                ldr_base_weight=float(getattr(cfg.loss, "v2_ldr_base_weight", 0.10)),
                non_detail_smooth_weight=float(getattr(cfg.loss, "v2_non_detail_smooth_weight", 0.05)),
                non_detail_second_order_weight=float(
                    getattr(cfg.loss, "v2_non_detail_second_order_weight", 0.05)
                ),
                target_detail_threshold=float(getattr(cfg.loss, "v2_target_detail_threshold", 0.02)),
                temporal_quality_floor=float(getattr(cfg.loss, "v2_temporal_quality_floor", 0.25)),
                counterfactual_weight=float(getattr(cfg.loss, "v2_counterfactual_weight", 0.20)),
                counterfactual_margin=float(getattr(cfg.loss, "v2_counterfactual_margin", 0.08)),
            )
            self._init_geo_contribution_loss(
                teacher_ldr_id=str(getattr(cfg.loss, "geo_teacher_ldr_id", "ev_10")),
                geo_target_weight=float(getattr(cfg.loss, "geo_event_target_weight", 0.30)),
                geo_reject_weight=float(getattr(cfg.loss, "geo_event_reject_weight", 0.20)),
                teacher_consistency_weight=float(
                    getattr(cfg.loss, "geo_teacher_consistency_weight", 0.15)
                ),
                event_delta_weight=float(getattr(cfg.loss, "geo_event_delta_weight", 0.0)),
                geo_teacher_boost=float(getattr(cfg.loss, "geo_teacher_boost", 1.5)),
                geo_detail_threshold=float(getattr(cfg.loss, "geo_detail_threshold", 0.02)),
                geo_positive_floor=float(getattr(cfg.loss, "geo_positive_floor", 0.05)),
                geo_negative_margin=float(getattr(cfg.loss, "geo_negative_margin", 0.05)),
            )

    return ConfiguredGeometryContributionEventLoss
