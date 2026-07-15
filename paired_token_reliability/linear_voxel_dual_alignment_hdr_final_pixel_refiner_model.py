"""Final dual-path model: HDR base geometry plus event-only pixel refinement."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_dual_alignment_hdr_pixel_hf_model import (
    PixelHighFrequencyDerivativeV10Model,
)
from stage2_geometry_adapter.model import depth_to_normals


class EventGeometryPixelRefiner(nn.Module):
    """Predict dense log-depth detail without consuming RGB pixels/tokens."""

    def __init__(self, event_channels=32, hidden=64):
        super().__init__()
        # event feature + dN(dx,dy) + log base depth + base normal + C
        channels = int(event_channels) + 6 + 1 + 3 + 1
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1), nn.GELU(),
        )
        self.context2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, bias=False), nn.GELU(),
        )
        self.context4 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4, bias=False), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(3 * hidden, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # Small, not zero: all preceding layers receive gradients at step one.
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, value):
        x = self.stem(value)
        return self.head(torch.cat((self.local(x), self.context2(x), self.context4(x)), 1))


class FinalEventGeometryPixelRefinerModel(PixelHighFrequencyDerivativeV10Model):
    checkpoint_schema = "dual_hdr_pixel_refiner_gt_scene_scale_aligned_final_v2"

    def __init__(self, *args, pixel_hidden=32, pixel_refiner_hidden=64,
                 pixel_refine_log_limit=.20, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        self.pixel_depth_refiner = EventGeometryPixelRefiner(
            event_channels=int(pixel_hidden), hidden=int(pixel_refiner_hidden)
        )
        self.pixel_refine_log_limit = float(pixel_refine_log_limit)
        self._aligned_event_feature_for_refiner = None

    def _event_patch_tokens(self, feature, reliability, patch_count, image_hw):
        # ``feature`` is the aligned full-resolution event feature immediately
        # before pooling. Keep the live tensor for the independent pixel path.
        self._aligned_event_feature_for_refiner = feature
        return super()._event_patch_tokens(feature, reliability, patch_count, image_hw)

    def _recency_gate(self, representation):
        bins = int(self.voxel_bins)
        recent = torch.cat((
            representation[:, :, bins - 2:bins],
            representation[:, :, 2 * bins - 2:2 * bins],
        ), dim=2).abs().sum(2)
        old = torch.cat((
            representation[:, :, :bins - 2],
            representation[:, :, bins:2 * bins - 2],
        ), dim=2).abs().sum(2)
        total = recent + old
        ratio = recent / total.clamp_min(1.0e-6)
        soft = torch.sigmoid(8.0 * (ratio - .30))
        return torch.where(total > 0, soft, torch.zeros_like(soft))

    def forward(self, views, *args, **kwargs):
        self._aligned_event_feature_for_refiner = None
        output = super().forward(views, *args, **kwargs)
        feature = self._aligned_event_feature_for_refiner
        if feature is None:
            raise RuntimeError("HDR event path did not expose its full-resolution feature")

        current_scale = self.metric_depth_scale
        raw_coarse = torch.stack(
            [item["depth_coarse_raw"][..., 0] for item in output.ress], 1
        ).float()
        gt_fields = [view.get("depthmap") for view in views]
        if not all(torch.is_tensor(value) for value in gt_fields):
            raise RuntimeError(
                "GT scene-scale protocol requires depthmap for every view at train/val/test time"
            )
        gt_depth = torch.stack(gt_fields, dim=1).to(raw_coarse).float()
        valid_scale = (
            torch.isfinite(raw_coarse) & torch.isfinite(gt_depth)
            & (raw_coarse > 1.0e-6) & (gt_depth > 1.0e-6)
        )
        weight = valid_scale.float()
        reduce_dims = tuple(range(1, raw_coarse.ndim))
        numerator = (weight * raw_coarse * gt_depth).sum(reduce_dims)
        denominator = (weight * raw_coarse.square()).sum(reduce_dims)
        scene_scale = (numerator / denominator.clamp_min(1.0e-6)).detach()
        valid_count = weight.sum(reduce_dims)
        if bool((valid_count <= 0).any()):
            raise RuntimeError("GT scene-scale protocol found a sample with no valid depth pixels")
        scale_view = scene_scale.view(-1, *([1] * (raw_coarse.ndim - 1)))
        coarse = raw_coarse * scale_view

        # The parent already applied its learned dataset scale. Remove it and
        # apply the one GT-derived scene scale shared by coarse/base/final.
        parent_base = torch.stack(
            [item["depth_hdr_base"][..., 0] for item in output.ress], 1
        ).float()
        base = parent_base / current_scale.clamp_min(1.0e-6) * scale_view
        contribution = torch.stack([item["event_contribution"] for item in output.ress], 1)
        representation = torch.stack([item["signed_event"] for item in output.ress], 1)
        derivative = torch.stack(
            [item["event_normal_derivative_full"] for item in output.ress], 1
        )
        recency = self._recency_gate(representation)
        recent_support = recency > .10

        # Suppress old-only trails only on the pixel-detail path. The complete
        # history remains untouched in full->geo and HDR-token alignment.
        feature = feature * recency.unsqueeze(2)
        derivative = derivative * recency.unsqueeze(-1).unsqueeze(-1)

        intrinsics = torch.stack(
            [view["camera_intrinsics"].to(base) for view in views], dim=1
        ).float()
        base_normal = depth_to_normals(base.float(), intrinsics)
        b, v, h, w = base.shape
        bv = b * v
        log_base_scene = torch.log(base.clamp_min(1.0e-6))
        valid_base = base > 1.0e-6
        center_weight = valid_base.float()
        center = (
            (log_base_scene * center_weight).sum((1, 2, 3))
            / center_weight.sum((1, 2, 3)).clamp_min(1.0)
        ).detach().view(b, 1, 1, 1)
        relative_log_base = log_base_scene - center
        log_base = log_base_scene.reshape(bv, 1, h, w)
        relative_log_ch = relative_log_base.reshape(bv, 1, h, w)
        normal_ch = base_normal.movedim(-1, 2).reshape(bv, 3, h, w)
        event_ch = feature.reshape(bv, feature.shape[2], h, w).float()
        derivative_ch = derivative.reshape(b, v, h, w, 6).movedim(-1, 2).reshape(bv, 6, h, w)

        warmup = bool(float(output.ress[0]["contribution_warmup_active"].detach()))
        c_live = contribution.detach() if warmup else contribution
        c_ch = c_live.reshape(bv, 1, h, w)
        step = int(float(output.ress[0]["pixel_hf_train_step"].detach()))
        coupling = max(0.0, min(1.0, (step - 1000) / 1000.0))
        if coupling <= 0.0:
            # Do not evaluate an inactive random refiner. Keep a zero-gradient
            # dependency so DDP still regards every parameter as used.
            zero = sum(
                (parameter.sum() * 0.0 for parameter in self.pixel_depth_refiner.parameters()),
                log_base.new_zeros(()),
            )
            raw = torch.zeros_like(log_base) + zero
            bounded = torch.zeros_like(log_base) + zero
        else:
            actual = torch.cat((event_ch, derivative_ch, relative_log_ch, normal_ch, c_ch), 1)
            # Same coarse geometry, but no event evidence. Subtraction makes
            # the update zero for absent events without a hard output mask.
            baseline = torch.cat((
                torch.zeros_like(event_ch), torch.zeros_like(derivative_ch),
                relative_log_ch, normal_ch, torch.zeros_like(c_ch),
            ), 1)
            raw = self.pixel_depth_refiner(actual) - self.pixel_depth_refiner(baseline)
            limit = max(self.pixel_refine_log_limit, 1.0e-6)
            bounded = limit * torch.tanh(raw / limit)
        delta_log = coupling * bounded
        refined = torch.exp(log_base + delta_log)[:, 0].reshape(b, v, h, w)
        # Do not turn invalid/background zero depth into a positive epsilon map.
        final = torch.where(base > 1.0e-6, refined, base)
        final_normal = depth_to_normals(final.float(), intrinsics)
        pixel_update = final - base
        ratio = final / base.clamp_min(1.0e-6) - 1.0
        ratio = torch.where(base > 1.0e-6, ratio, torch.zeros_like(ratio))
        tv = .5 * (
            (ratio[..., :, 1:] - ratio[..., :, :-1]).abs().mean()
            + (ratio[..., 1:, :] - ratio[..., :-1, :]).abs().mean()
        )
        regularizer = ratio.abs().mean() + .25 * tv

        for index, item in enumerate(output.ress):
            point_scale = (
                scene_scale / current_scale.detach().clamp_min(1.0e-6)
            ).view(b, 1, 1, 1)
            coarse_points = item["pts3d_coarse"] * point_scale
            hdr_points = item["pts3d_hdr_base"] * point_scale
            item["pts3d_coarse"] = coarse_points
            item["pts3d_hdr_base"] = hdr_points
            item["pts3d_in_other_view"] = hdr_points
            item["point_token_map_update"] = hdr_points - coarse_points
            item["point_total_update"] = hdr_points - coarse_points
            item["event_normal_derivative"] = derivative[:, index]
            item["event_normal_derivative_full"] = derivative[:, index]
            item["event_normal_support"] = recent_support[:, index]
            item["event_detail_recency"] = recency[:, index]
            item["depth_coarse"] = coarse[:, index].unsqueeze(-1)
            item["depth_hdr_base"] = base[:, index].unsqueeze(-1)
            item["gt_scene_scale"] = scene_scale
            item["metric_depth_scale"] = scene_scale
            item["learned_dataset_scale_diagnostic"] = current_scale
            item["depth"] = final[:, index].unsqueeze(-1)
            item["normal"] = final_normal[:, index]
            item["depth_geometry_update"] = pixel_update[:, index]
            item["depth_pixel_update"] = pixel_update[:, index]
            item["depth_update_final_absolute"] = pixel_update[:, index]
            item["depth_delta_ratio"] = ratio[:, index]
            item["depth_update_centered_ratio"] = ratio[:, index]
            item["depth_update_detail_ratio"] = ratio[:, index]
            item["depth_total_update"] = final[:, index] - coarse[:, index]
            item["depth_update_tv"] = tv
            item["adapter_update_loss"] = regularizer
            item["pixel_refiner_raw_update"] = raw[:, 0].reshape(b, v, h, w)[:, index]
            item["pixel_refiner_bounded_update"] = bounded[:, 0].reshape(b, v, h, w)[:, index]
            item["pixel_refiner_coupling"] = final.new_tensor(coupling)
        return output


__all__ = ["FinalEventGeometryPixelRefinerModel", "EventGeometryPixelRefiner"]
