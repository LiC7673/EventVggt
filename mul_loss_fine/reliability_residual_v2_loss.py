"""Losses for temporal-reliability gated multi-LDR detail refinement."""

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
from mul_loss_fine.launcher import make_configured_loss


def _batch_group_keys(views: List[Dict[str, torch.Tensor]], batch_size: int):
    keys = []
    for batch_idx in range(batch_size):
        instances = []
        for view in views:
            instance = view.get("instance", "")
            if isinstance(instance, (list, tuple)):
                instances.append(str(instance[batch_idx]))
            else:
                instances.append(str(instance))
        keys.append(tuple(instances))
    return keys


def _masked_log_residual_smoothness(
    delta_log: torch.Tensor,
    valid_mask: torch.Tensor,
    detail_support: torch.Tensor,
):
    smooth_mask = valid_mask.to(dtype=delta_log.dtype) * (1.0 - detail_support.squeeze(2).detach())
    dx = (delta_log[..., :, 1:] - delta_log[..., :, :-1]).abs()
    dy = (delta_log[..., 1:, :] - delta_log[..., :-1, :]).abs()
    wx = smooth_mask[..., :, 1:] * smooth_mask[..., :, :-1]
    wy = smooth_mask[..., 1:, :] * smooth_mask[..., :-1, :]
    first = (dx * wx).sum() / wx.sum().clamp_min(1.0) + (dy * wy).sum() / wy.sum().clamp_min(1.0)

    dxx = (delta_log[..., :, 2:] - 2.0 * delta_log[..., :, 1:-1] + delta_log[..., :, :-2]).abs()
    dyy = (delta_log[..., 2:, :] - 2.0 * delta_log[..., 1:-1, :] + delta_log[..., :-2, :]).abs()
    wxx = smooth_mask[..., :, 2:] * smooth_mask[..., :, 1:-1] * smooth_mask[..., :, :-2]
    wyy = smooth_mask[..., 2:, :] * smooth_mask[..., 1:-1, :] * smooth_mask[..., :-2, :]
    second = (dxx * wxx).sum() / wxx.sum().clamp_min(1.0) + (
        dyy * wyy
    ).sum() / wyy.sum().clamp_min(1.0)
    return first, second


class TemporalReliabilityV2LossMixin:
    """Teach the event gate where the frozen coarse estimate needs correction."""

    def _init_reliability_v2_loss(
        self,
        *,
        residual_scale: float,
        residual_target_weight: float,
        gate_reliability_weight: float,
        gate_need_floor: float,
        gate_positive_boost: float,
        ldr_depth_weight: float,
        ldr_normal_weight: float,
        ldr_correction_weight: float,
        ldr_base_weight: float,
        non_detail_smooth_weight: float,
        non_detail_second_order_weight: float,
        target_detail_threshold: float,
        temporal_quality_floor: float,
        counterfactual_weight: float,
        counterfactual_margin: float,
    ):
        self.v2_residual_scale = max(float(residual_scale), 1e-6)
        self.v2_residual_target_weight = float(residual_target_weight)
        self.v2_gate_reliability_weight = float(gate_reliability_weight)
        self.v2_gate_need_floor = min(max(float(gate_need_floor), 0.0), 1.0)
        self.v2_gate_positive_boost = float(gate_positive_boost)
        self.v2_ldr_depth_weight = float(ldr_depth_weight)
        self.v2_ldr_normal_weight = float(ldr_normal_weight)
        self.v2_ldr_correction_weight = float(ldr_correction_weight)
        self.v2_ldr_base_weight = float(ldr_base_weight)
        self.v2_non_detail_smooth_weight = float(non_detail_smooth_weight)
        self.v2_non_detail_second_order_weight = float(non_detail_second_order_weight)
        self.v2_target_detail_threshold = float(target_detail_threshold)
        self.v2_temporal_quality_floor = min(max(float(temporal_quality_floor), 0.0), 1.0)
        self.v2_counterfactual_weight = float(counterfactual_weight)
        self.v2_counterfactual_margin = float(counterfactual_margin)

    def forward(self, model_output, views):
        total_loss, details, aux = super().forward(model_output, views)
        depth_coarse = _stack_output_field(model_output, "depth_coarse")
        delta_log = _stack_output_field(model_output, "depth_delta_log")
        reliability = _stack_output_field(model_output, "event_reliability")
        reliability_reverse = _stack_output_field(model_output, "event_reliability_reverse_time")
        reliability_swap = _stack_output_field(model_output, "event_reliability_swap_polarity")
        temporal_quality = _stack_output_field(model_output, "event_temporal_quality")
        presence = _stack_output_field(model_output, "event_presence")
        if depth_coarse is None or delta_log is None or reliability is None or presence is None:
            details.update(
                {
                    "v2_extra_loss": 0.0,
                    "residual_target_loss": 0.0,
                    "gate_reliability_loss": 0.0,
                    "counterfactual_reliability_loss": 0.0,
                    "non_detail_residual_smooth_loss": 0.0,
                    "non_detail_residual_second_order_loss": 0.0,
                    "ldr_final_depth_loss": 0.0,
                    "ldr_final_normal_loss": 0.0,
                    "ldr_correction_consistency_loss": 0.0,
                    "ldr_pair_count": 0.0,
                    "event_temporal_quality_mean": 0.0,
                    "event_reverse_reliability_mean": 0.0,
                    "event_swap_reliability_mean": 0.0,
                }
            )
            return total_loss, details, aux

        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)
        dtype = depth_pred.dtype
        device = depth_pred.device
        batch, seq_len, height, width = depth_pred.shape
        depth_coarse = depth_coarse.to(device=device, dtype=dtype)
        delta_log = delta_log.to(device=device, dtype=dtype)
        reliability = reliability.to(device=device, dtype=dtype)
        reliability_reverse = (
            reliability_reverse.to(device=device, dtype=dtype) if reliability_reverse is not None else None
        )
        reliability_swap = reliability_swap.to(device=device, dtype=dtype) if reliability_swap is not None else None
        temporal_quality = (
            temporal_quality.to(device=device, dtype=dtype).detach().clamp(0.0, 1.0)
            if temporal_quality is not None
            else torch.ones_like(reliability).detach()
        )
        presence = presence.to(device=device, dtype=dtype).detach().clamp(0.0, 1.0)
        depth_gt = fe.stack_view_field(views, "depthmap").to(device=device, dtype=dtype)
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=device, dtype=dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)
        valid_weight = valid_mask.unsqueeze(2).to(dtype=dtype)

        gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics)
        coarse_normals = fe.depth_to_normals(depth_coarse.clamp_min(self.depth_min), intrinsics)
        final_normals = fe.depth_to_normals(depth_pred.clamp_min(self.depth_min), intrinsics)
        gt_gradient = _normal_gradient_magnitude(gt_normals.flatten(0, 1)).view(
            batch, seq_len, 1, height, width
        )
        detail_support = _normalize_detail_support(
            gt_gradient,
            valid_mask,
            threshold=self.v2_target_detail_threshold,
            power=1.0,
        ).detach()

        target_delta = (
            torch.log(depth_gt.clamp_min(self.depth_min))
            - torch.log(depth_coarse.detach().clamp_min(self.depth_min))
        ).clamp(-self.v2_residual_scale, self.v2_residual_scale)
        target_strength = (target_delta.abs() / self.v2_residual_scale).unsqueeze(2).clamp(0.0, 1.0)
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
        correction_need = torch.maximum(target_strength, normal_need)
        correction_need = correction_need * (0.25 + 0.75 * detail_support)

        event_reachable = 0.05 + 0.95 * presence.unsqueeze(2)
        target_weight = valid_weight * event_reachable * (0.10 + detail_support)
        residual_target_loss = _weighted_mean(
            ((delta_log - target_delta).abs() / self.v2_residual_scale).unsqueeze(2),
            target_weight,
        )

        quality_gain = self.v2_temporal_quality_floor + (1.0 - self.v2_temporal_quality_floor) * temporal_quality.unsqueeze(2)
        reliability_target = self.v2_gate_need_floor + (
            1.0 - self.v2_gate_need_floor
        ) * correction_need * quality_gain
        reliability_weight = valid_weight * presence.unsqueeze(2) * (
            1.0 + self.v2_gate_positive_boost * correction_need
        )
        gate_reliability_loss = _weighted_mean(
            F.binary_cross_entropy(
                reliability.clamp(1e-5, 1.0 - 1e-5).unsqueeze(2),
                reliability_target.detach(),
                reduction="none",
            ),
            reliability_weight,
        )

        residual_smooth_loss, residual_second_order_loss = _masked_log_residual_smoothness(
            delta_log,
            valid_mask,
            detail_support,
        )
        counterfactual_terms = []
        contrast_weight = valid_weight * presence.unsqueeze(2) * correction_need.detach()
        for counterfactual_reliability in (reliability_reverse, reliability_swap):
            if counterfactual_reliability is None:
                continue
            margin_loss = F.relu(
                self.v2_counterfactual_margin
                + counterfactual_reliability.unsqueeze(2)
                - reliability.unsqueeze(2)
            )
            counterfactual_terms.append(_weighted_mean(margin_loss, contrast_weight))
        counterfactual_loss = (
            torch.stack(counterfactual_terms).mean()
            if counterfactual_terms
            else depth_pred.new_tensor(0.0)
        )

        groups = {}
        for batch_idx, key in enumerate(_batch_group_keys(views, batch)):
            groups.setdefault(key, []).append(batch_idx)
        pairs = [(indices[0], other) for indices in groups.values() for other in indices[1:]]
        log_depth_pred = torch.log(depth_pred.clamp_min(self.depth_min))
        correction_error = delta_log - target_delta
        depth_terms = []
        normal_terms = []
        correction_terms = []
        for first, second in pairs:
            pair_valid = (valid_mask[first] & valid_mask[second]).unsqueeze(1).to(dtype=dtype)
            pair_detail = torch.maximum(detail_support[first], detail_support[second])
            pair_presence = torch.maximum(presence[first], presence[second]).unsqueeze(1)
            pair_event_reachable = 0.05 + 0.95 * pair_presence
            pair_weight = pair_valid * pair_event_reachable * (
                self.v2_ldr_base_weight + pair_detail * (0.5 + 0.5 * pair_presence)
            )
            depth_terms.append(
                _weighted_mean((log_depth_pred[first] - log_depth_pred[second]).abs().unsqueeze(1), pair_weight)
            )
            normal_map = 1.0 - (
                F.normalize(final_normals[first], dim=-1, eps=1e-6)
                * F.normalize(final_normals[second], dim=-1, eps=1e-6)
            ).sum(dim=-1).clamp(-1.0, 1.0)
            normal_terms.append(_weighted_mean(normal_map.unsqueeze(1), pair_weight))
            correction_terms.append(
                _weighted_mean(
                    ((correction_error[first] - correction_error[second]).abs() / self.v2_residual_scale).unsqueeze(1),
                    pair_weight,
                )
            )

        zero = depth_pred.new_tensor(0.0)
        ldr_depth_loss = torch.stack(depth_terms).mean() if depth_terms else zero
        ldr_normal_loss = torch.stack(normal_terms).mean() if normal_terms else zero
        ldr_correction_loss = torch.stack(correction_terms).mean() if correction_terms else zero
        v2_extra = (
            self.v2_residual_target_weight * residual_target_loss
            + self.v2_gate_reliability_weight * gate_reliability_loss
            + self.v2_non_detail_smooth_weight * residual_smooth_loss
            + self.v2_non_detail_second_order_weight * residual_second_order_loss
            + self.v2_counterfactual_weight * counterfactual_loss
            + self.v2_ldr_depth_weight * ldr_depth_loss
            + self.v2_ldr_normal_weight * ldr_normal_loss
            + self.v2_ldr_correction_weight * ldr_correction_loss
        )
        total_loss = total_loss + v2_extra

        persistence = _stack_output_field(model_output, "event_persistence")
        entropy = _stack_output_field(model_output, "event_entropy")
        aux["event_reliability"] = reliability.detach()
        aux["event_persistence"] = persistence.detach() if persistence is not None else presence.detach()
        aux["event_temporal_quality"] = temporal_quality.detach()
        details.update(
            {
                "v2_extra_loss": float(v2_extra.detach()),
                "residual_target_loss": float(residual_target_loss.detach()),
                "gate_reliability_loss": float(gate_reliability_loss.detach()),
                "counterfactual_reliability_loss": float(counterfactual_loss.detach()),
                "event_reliability_mean": float(_weighted_mean(reliability.unsqueeze(2), valid_weight).detach()),
                "event_reverse_reliability_mean": float(reliability_reverse.mean().detach())
                if reliability_reverse is not None
                else 0.0,
                "event_swap_reliability_mean": float(reliability_swap.mean().detach())
                if reliability_swap is not None
                else 0.0,
                "event_temporal_quality_mean": float(_weighted_mean(temporal_quality.unsqueeze(2), valid_weight).detach()),
                "event_persistence_mean": float(persistence.mean().detach()) if persistence is not None else 0.0,
                "event_entropy_mean": float(entropy.mean().detach()) if entropy is not None else 0.0,
                "event_conditioned_delta_log_abs": float(
                    _weighted_mean(delta_log.abs().unsqueeze(2), valid_weight * presence.unsqueeze(2)).detach()
                ),
                "non_detail_residual_smooth_loss": float(residual_smooth_loss.detach()),
                "non_detail_residual_second_order_loss": float(residual_second_order_loss.detach()),
                "ldr_final_depth_loss": float(ldr_depth_loss.detach()),
                "ldr_final_normal_loss": float(ldr_normal_loss.detach()),
                "ldr_correction_consistency_loss": float(ldr_correction_loss.detach()),
                "ldr_pair_count": float(len(pairs)),
                "extra_loss_total": float(details.get("extra_loss_total", 0.0)) + float(v2_extra.detach()),
                "total_loss_with_extra": float(total_loss.detach()),
            }
        )
        return total_loss, details, aux


def make_configured_reliability_v2_loss(cfg):
    configured_base = make_configured_loss(cfg)

    class ConfiguredTemporalReliabilityV2Loss(TemporalReliabilityV2LossMixin, configured_base):
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

    return ConfiguredTemporalReliabilityV2Loss
