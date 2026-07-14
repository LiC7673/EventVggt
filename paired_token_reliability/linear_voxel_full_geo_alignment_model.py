"""One-stage full-to-geometry event alignment with normal-bottleneck depth."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_conditioned_dense_scale_warmup_model import (
    ConditionedDenseScaleWarmupLinearVoxelModel,
)
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class FullToGeoAlignment(nn.Module):
    """Map full-event features to geo-like features and predict alignment quality."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.residual = nn.Conv2d(channels, channels, 1)
        self.reliability = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    def forward(self, full_feature):
        correction = self.residual(self.body(full_feature))
        aligned = full_feature + correction
        reliability = torch.sigmoid(
            self.reliability(torch.cat((full_feature, correction), dim=1))
        )
        return aligned, correction, reliability


class FullGeoAlignmentLinearVoxelModel(ConditionedDenseScaleWarmupLinearVoxelModel):
    checkpoint_schema = "linear_time_voxel_full_geo_alignment_v1"

    def __init__(self, *args, pixel_hidden=32, normal_refine_iterations=3,
                 normal_refine_step_limit=.05, alignment_confidence_tau=.10,
                 normal_bottleneck_warmup_steps=1000, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        self.full_geo_aligner = FullToGeoAlignment(hidden)
        # normal residual (3), current normal (3), reliability (1), log ratio (1)
        self.normal_bottleneck_refiner = nn.Sequential(
            nn.Conv2d(8, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.normal_bottleneck_refiner[-1].weight)
        nn.init.zeros_(self.normal_bottleneck_refiner[-1].bias)
        self.normal_refine_iterations = max(1, int(normal_refine_iterations))
        self.normal_refine_step_limit = float(normal_refine_step_limit)
        self.alignment_confidence_tau = max(float(alignment_confidence_tau), 1e-4)
        self.normal_bottleneck_warmup_steps = max(
            int(self.scale_warmup_steps), int(normal_bottleneck_warmup_steps)
        )

    def _decode_normal(self, feature):
        b, v, channels, height, width = feature.shape
        flat = feature.reshape(b * v, channels, height, width)
        normal = F.normalize(
            self.event_normal_decoder(flat).float(), dim=1, eps=1e-6
        )
        return normal.reshape(b, v, 3, height, width).movedim(2, -1)

    def _encode_geo_teacher(self, views, full_feature):
        fields = [view.get("geometry_event_voxel") for view in views]
        if not all(torch.is_tensor(value) for value in fields):
            if self.training:
                raise RuntimeError(
                    "one-stage alignment training requires geometry_event_voxel"
                )
            return None, None
        geo_voxel = torch.stack(fields, dim=1).to(full_feature)
        geo_signed, _ = self._decayed_signed(views, geo_voxel)
        geo_support = self._last_filtered_event_support.detach().clone()
        geo_feature = self.event_encoder(geo_signed)
        # The hook follows the most recent encoder call; restore its semantic
        # value for diagnostics that expect the deployed full-event feature.
        self._captured_event_feature = full_feature
        return geo_feature, geo_support

    def _zero_dependency(self):
        value = self.depth_log_scale * 0.0
        for parameter in self.normal_bottleneck_refiner.parameters():
            value = value + parameter.sum() * 0.0
        return value

    @staticmethod
    def _write_depth(views, output, coarse, final, ratio, regularizer):
        for index, (view, item) in enumerate(zip(views, output.ress)):
            final_i, ratio_i = final[:, index], ratio[:, index]
            item["depth"] = final_i.unsqueeze(-1)
            item["normal"] = depth_to_normals(
                final_i.float(), view["camera_intrinsics"].to(final_i).float()
            )
            item["depth_delta_ratio"] = ratio_i
            item["depth_pixel_update"] = coarse[:, index] * ratio_i
            item["depth_total_update"] = item["depth_pixel_update"]
            item["depth_update_final_absolute"] = item["depth_pixel_update"]
            item["depth_update_centered_ratio"] = ratio_i
            item["depth_update_detail_ratio"] = ratio_i
            item["adapter_update_loss"] = regularizer
            item["depth_update_tv"] = regularizer

    def forward(self, views, *args, **kwargs):
        # target views keep E_full in event_voxel; the parent computes RGB
        # coarse geometry and the deployed full-event feature once.
        output = super().forward(views, *args, **kwargs)
        if not output.ress:
            return output
        full_feature = self._captured_event_feature
        if full_feature is None:
            raise RuntimeError("full-event feature was not captured")
        full_support = self._last_filtered_event_support.detach().clone()
        b, v, hidden, height, width = full_feature.shape
        flat_full = full_feature.reshape(b * v, hidden, height, width)
        aligned_flat, correction_flat, reliability_flat = self.full_geo_aligner(flat_full)
        aligned_feature = aligned_flat.reshape(b, v, hidden, height, width)
        correction = correction_flat.reshape(b, v, hidden, height, width)
        reliability = reliability_flat[:, 0].reshape(b, v, height, width)

        geo_feature, geo_support = self._encode_geo_teacher(views, full_feature)
        full_normal = self._decode_normal(aligned_feature)
        geo_normal = self._decode_normal(geo_feature) if geo_feature is not None else None

        if geo_feature is not None:
            # Teacher is stop-gradient only for alignment. Its own normal loss
            # still trains the shared encoder/normal decoder.
            feature_error = F.smooth_l1_loss(
                aligned_feature, geo_feature.detach(), beta=.05, reduction="none"
            ).mean(dim=2)
            reliability_target = torch.exp(
                -feature_error.detach() / self.alignment_confidence_tau
            ).clamp(0.0, 1.0)
        else:
            feature_error = reliability.new_zeros(reliability.shape)
            reliability_target = reliability.detach()

        coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], dim=1
        ).float()
        intrinsics = torch.stack(
            [view["camera_intrinsics"] for view in views], dim=1
        ).to(coarse).float()
        coarse_normal = depth_to_normals(coarse, intrinsics)
        coarse_n = coarse_normal.movedim(-1, 2).reshape(b * v, 3, height, width)
        full_n = full_normal.movedim(-1, 2).reshape(b * v, 3, height, width)
        reliability_4d = reliability.reshape(b * v, 1, height, width)
        fused_normal = F.normalize(
            coarse_n + reliability_4d * (full_n - coarse_n), dim=1, eps=1e-6
        )

        kernel = max(1, int(self.support_dilation_kernel))
        full_local = full_support.float().reshape(b * v, 1, height, width)
        if kernel > 1:
            full_local = F.max_pool2d(full_local, kernel, 1, kernel // 2)
        full_local = full_local[:, 0].reshape(b, v, height, width) > 0
        if geo_support is not None:
            geo_local = geo_support.float().reshape(b * v, 1, height, width)
            if kernel > 1:
                geo_local = F.max_pool2d(geo_local, kernel, 1, kernel // 2)
            geo_local = geo_local[:, 0].reshape(b, v, height, width) > 0
        else:
            geo_local = full_local

        bottleneck_active = not (
            self.training
            and self._scale_warmup_forward_step <= self.normal_bottleneck_warmup_steps
        )
        log_coarse = torch.log(coarse.clamp_min(1e-6))
        if not bottleneck_active:
            final = coarse + self._zero_dependency()
            ratio = final * 0.0
            self._write_depth(views, output, coarse, final, ratio, ratio.mean())
            iteration_updates = None
        else:
            log_coarse_flat = log_coarse.reshape(b * v, 1, height, width)
            log_depth = log_coarse_flat.clone()
            intrinsics_flat = intrinsics.reshape(b * v, 3, 3)
            step_limit = max(self.normal_refine_step_limit, 1e-6)
            updates = []
            for _ in range(self.normal_refine_iterations):
                current_depth = torch.exp(log_depth[:, 0])
                current_normal = depth_to_normals(
                    current_depth.unsqueeze(1), intrinsics_flat.unsqueeze(1)
                )[:, 0].movedim(-1, 1)
                normal_residual = fused_normal - current_normal
                log_ratio = log_depth - log_coarse_flat
                refine_input = torch.cat((
                    normal_residual, current_normal, reliability_4d, log_ratio,
                ), dim=1)
                baseline_input = torch.cat((
                    torch.zeros_like(normal_residual), current_normal,
                    reliability_4d, log_ratio,
                ), dim=1)
                raw = (
                    self.normal_bottleneck_refiner(refine_input)
                    - self.normal_bottleneck_refiner(baseline_input)
                )
                step = reliability_4d * step_limit * torch.tanh(raw / step_limit)
                log_depth = log_depth + step
                updates.append(step)
            total_limit = min(max(float(self.depth_update_scale), 1e-6), .999)
            ratio_log = (log_depth - log_coarse_flat).clamp(
                min=math.log(1.0 - total_limit),
                max=math.log(1.0 + total_limit),
            )
            final = torch.exp(log_coarse_flat + ratio_log)[:, 0].reshape(
                b, v, height, width
            )
            ratio = final / coarse.clamp_min(1e-6) - 1.0
            dx = ratio[..., :, 1:] - ratio[..., :, :-1]
            dy = ratio[..., 1:, :] - ratio[..., :-1, :]
            regularizer = .5 * (dx.abs().mean() + dy.abs().mean())
            self._write_depth(views, output, coarse, final, ratio, regularizer)
            iteration_updates = torch.cat(updates, dim=1).reshape(
                b, v, self.normal_refine_iterations, height, width
            )

        fused_bv = fused_normal.reshape(b, v, 3, height, width).movedim(2, -1)
        for index, item in enumerate(output.ress):
            item["event_normal"] = full_normal[:, index]
            item["event_normal_full"] = full_normal[:, index]
            item["event_normal_geo"] = (
                geo_normal[:, index] if geo_normal is not None else full_normal[:, index].detach()
            )
            item["event_normal_support"] = full_local[:, index]
            item["geo_event_support"] = geo_local[:, index]
            item["event_contribution"] = reliability[:, index]
            item["event_contribution_spatial"] = reliability[:, index]
            item["alignment_reliability"] = reliability[:, index]
            item["alignment_reliability_target"] = reliability_target[:, index]
            item["alignment_feature_error"] = feature_error[:, index]
            item["alignment_feature_correction"] = correction[:, index]
            item["normal_refine_target"] = fused_bv[:, index]
            item["normal_bottleneck_active"] = item["depth"].new_tensor(
                float(bottleneck_active)
            )
            if iteration_updates is not None:
                item["normal_refine_iteration_updates"] = iteration_updates[:, index]
        return GeometryAdapterOutput(ress=output.ress, views=output.views)


__all__ = ["FullGeoAlignmentLinearVoxelModel", "FullToGeoAlignment"]
