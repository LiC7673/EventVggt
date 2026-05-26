"""Loss terms for multi-LDR exposure-invariant temporal event gating."""

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


def _resize_sequence_weight(weight: torch.Tensor, size) -> torch.Tensor:
    batch, seq_len = weight.shape[:2]
    resized = F.interpolate(weight.flatten(0, 1), size=size, mode="area")
    return resized.view(batch, seq_len, 1, size[0], size[1])


class ExposureInvariantLossMixin:
    """Add training-only paired-exposure supervision to the standard detail loss."""

    def _init_exposure_invariant_loss(
        self,
        *,
        exposure_feature_weight: float,
        exposure_event_match_weight: float,
        exposure_depth_weight: float,
        exposure_normal_weight: float,
        event_reliability_weight: float,
        exposure_output_base_weight: float,
        reliability_detail_threshold: float,
        reliability_target_floor: float,
    ) -> None:
        self.exposure_feature_weight = float(exposure_feature_weight)
        self.exposure_event_match_weight = float(exposure_event_match_weight)
        self.exposure_depth_weight = float(exposure_depth_weight)
        self.exposure_normal_weight = float(exposure_normal_weight)
        self.event_reliability_weight = float(event_reliability_weight)
        self.exposure_output_base_weight = float(exposure_output_base_weight)
        self.reliability_detail_threshold = float(reliability_detail_threshold)
        self.reliability_target_floor = min(max(float(reliability_target_floor), 0.0), 1.0)

    @staticmethod
    def _zero_details(details):
        details.update(
            {
                "ldr_invariant_loss": 0.0,
                "ldr_feature_consistency_loss": 0.0,
                "ldr_event_match_loss": 0.0,
                "ldr_output_depth_loss": 0.0,
                "ldr_output_normal_loss": 0.0,
                "event_reliability_loss": 0.0,
                "event_agreement_mean": 0.0,
                "ldr_pair_count": 0.0,
            }
        )

    def forward(self, model_output, views):
        total_loss, details, aux = super().forward(model_output, views)
        appearance_feature = _stack_output_field(model_output, "exposure_feature")
        event_feature = _stack_output_field(model_output, "exposure_event_feature")
        event_agreement = _stack_output_field(model_output, "event_agreement")
        event_presence = _stack_output_field(model_output, "event_presence")
        if appearance_feature is None or event_feature is None or event_agreement is None or event_presence is None:
            self._zero_details(details)
            return total_loss, details, aux

        depth_pred = torch.stack([res["depth"] for res in model_output.ress], dim=1).squeeze(-1)
        batch, seq_len, height, width = depth_pred.shape
        groups = {}
        for batch_idx, key in enumerate(_batch_group_keys(views, batch)):
            groups.setdefault(key, []).append(batch_idx)
        pairs = [(indices[0], other) for indices in groups.values() for other in indices[1:]]

        event_agreement = event_agreement.to(device=depth_pred.device, dtype=depth_pred.dtype)
        event_presence = event_presence.to(device=depth_pred.device, dtype=depth_pred.dtype).detach()
        aux["event_agreement"] = event_agreement.detach()
        details["event_agreement_mean"] = float(event_agreement.mean().detach())
        if not pairs:
            self._zero_details(details)
            details["event_agreement_mean"] = float(event_agreement.mean().detach())
            return total_loss, details, aux

        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=depth_pred.device, dtype=depth_pred.dtype)
        depth_gt = fe.stack_view_field(views, "depthmap").to(device=depth_pred.device, dtype=depth_pred.dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)
        pred_normals = fe.depth_to_normals(depth_pred.clamp_min(self.depth_min), intrinsics)
        gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics)
        gt_gradient = _normal_gradient_magnitude(gt_normals.flatten(0, 1)).view(
            batch, seq_len, 1, height, width
        ).detach()
        detail_support = _normalize_detail_support(
            gt_gradient,
            valid_mask,
            threshold=self.reliability_detail_threshold,
            power=1.0,
        ).detach()
        valid_weight = valid_mask.unsqueeze(2).to(dtype=depth_pred.dtype)
        presence_weight = event_presence.unsqueeze(2).clamp(0.0, 1.0)
        reliable_support = detail_support * presence_weight * valid_weight

        # During training GT geometry teaches the event agreement branch to
        # reject active but non-geometric event evidence.
        reliability_target = self.reliability_target_floor + (
            1.0 - self.reliability_target_floor
        ) * detail_support.squeeze(2)
        reliability_weight = presence_weight * valid_weight
        reliability_map = F.binary_cross_entropy(
            event_agreement.clamp(1e-5, 1.0 - 1e-5),
            reliability_target,
            reduction="none",
        ).unsqueeze(2)
        reliability_loss = _weighted_mean(reliability_map, reliability_weight)

        appearance_feature = appearance_feature.to(device=depth_pred.device, dtype=depth_pred.dtype)
        event_feature = event_feature.to(device=depth_pred.device, dtype=depth_pred.dtype)
        feature_size = appearance_feature.shape[-2:]
        reliable_low = _resize_sequence_weight(reliable_support, feature_size)

        feature_terms = []
        match_terms = []
        depth_terms = []
        normal_terms = []
        log_depth = torch.log(depth_pred.clamp_min(self.depth_min))
        for first, second in pairs:
            low_weight = torch.maximum(reliable_low[first], reliable_low[second])
            first_feature = F.normalize(appearance_feature[first], dim=1, eps=1e-6)
            second_feature = F.normalize(appearance_feature[second], dim=1, eps=1e-6)
            feature_cos = 1.0 - (first_feature * second_feature).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            feature_terms.append(_weighted_mean(feature_cos, low_weight))

            shared_event = F.normalize(
                0.5 * (event_feature[first] + event_feature[second]),
                dim=1,
                eps=1e-6,
            )
            for exposure_feature in (first_feature, second_feature):
                match_cos = 1.0 - (exposure_feature * shared_event).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                match_terms.append(_weighted_mean(match_cos, low_weight))

            pair_valid = (valid_mask[first] & valid_mask[second]).unsqueeze(1).to(dtype=depth_pred.dtype)
            full_weight = pair_valid * (
                self.exposure_output_base_weight
                + torch.maximum(reliable_support[first], reliable_support[second])
            )
            depth_terms.append(_weighted_mean((log_depth[first] - log_depth[second]).abs().unsqueeze(1), full_weight))
            first_normal = F.normalize(pred_normals[first], dim=-1, eps=1e-6)
            second_normal = F.normalize(pred_normals[second], dim=-1, eps=1e-6)
            normal_map = 1.0 - (first_normal * second_normal).sum(dim=-1).clamp(-1.0, 1.0)
            normal_terms.append(_weighted_mean(normal_map.unsqueeze(1), full_weight))

        zero = depth_pred.new_tensor(0.0)
        feature_loss = torch.stack(feature_terms).mean() if feature_terms else zero
        match_loss = torch.stack(match_terms).mean() if match_terms else zero
        depth_loss = torch.stack(depth_terms).mean() if depth_terms else zero
        normal_loss = torch.stack(normal_terms).mean() if normal_terms else zero
        invariant_loss = (
            self.exposure_feature_weight * feature_loss
            + self.exposure_event_match_weight * match_loss
            + self.exposure_depth_weight * depth_loss
            + self.exposure_normal_weight * normal_loss
            + self.event_reliability_weight * reliability_loss
        )
        total_loss = total_loss + invariant_loss
        details.update(
            {
                "ldr_invariant_loss": float(invariant_loss.detach()),
                "ldr_feature_consistency_loss": float(feature_loss.detach()),
                "ldr_event_match_loss": float(match_loss.detach()),
                "ldr_output_depth_loss": float(depth_loss.detach()),
                "ldr_output_normal_loss": float(normal_loss.detach()),
                "event_reliability_loss": float(reliability_loss.detach()),
                "event_agreement_mean": float(event_agreement.mean().detach()),
                "ldr_pair_count": float(len(pairs)),
                "extra_loss_total": float(details.get("extra_loss_total", 0.0)) + float(invariant_loss.detach()),
                "total_loss_with_extra": float(total_loss.detach()),
            }
        )
        return total_loss, details, aux


def make_configured_exposure_invariant_loss(cfg):
    configured_base = make_configured_loss(cfg)

    class ConfiguredExposureInvariantLoss(ExposureInvariantLossMixin, configured_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_exposure_invariant_loss(
                exposure_feature_weight=float(getattr(cfg.loss, "ldr_feature_consistency_weight", 0.10)),
                exposure_event_match_weight=float(getattr(cfg.loss, "ldr_event_match_weight", 0.05)),
                exposure_depth_weight=float(getattr(cfg.loss, "ldr_output_depth_weight", 0.10)),
                exposure_normal_weight=float(getattr(cfg.loss, "ldr_output_normal_weight", 0.10)),
                event_reliability_weight=float(getattr(cfg.loss, "ldr_event_reliability_weight", 0.05)),
                exposure_output_base_weight=float(getattr(cfg.loss, "ldr_output_base_weight", 0.10)),
                reliability_detail_threshold=float(getattr(cfg.loss, "ldr_reliability_detail_threshold", 0.02)),
                reliability_target_floor=float(getattr(cfg.loss, "ldr_reliability_target_floor", 0.25)),
            )

    return ConfiguredExposureInvariantLoss
