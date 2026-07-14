"""Dual alignment: full->geo event tokens, then LDR+event->HDR tokens."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_calibrated_model import (
    CalibratedLinearVoxelMultiscalePixelModel,
)
from paired_token_reliability.linear_voxel_full_geo_alignment_model import (
    FullToGeoAlignment,
)
from paired_token_reliability.signed_multiscale_model import signed_support
from stage2_geometry_adapter.model import GeometryAdapterOutput, depth_to_normals


class LdrEventToHdrTokenAligner(nn.Module):
    """Residual token adapter; zero initialization preserves pretrained RGB."""

    def __init__(self, token_dim, bottleneck=256):
        super().__init__()
        token_dim, bottleneck = int(token_dim), int(bottleneck)
        self.rgb_norm = nn.LayerNorm(token_dim)
        self.event_norm = nn.LayerNorm(token_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * token_dim, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, token_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, ldr_token, event_token):
        residual = self.fusion(torch.cat((
            self.rgb_norm(ldr_token), self.event_norm(event_token),
        ), dim=-1))
        return ldr_token + residual, residual


class DualAlignmentHDRLinearVoxelModel(CalibratedLinearVoxelMultiscalePixelModel):
    checkpoint_schema = "linear_time_voxel_dual_alignment_hdr_predicted_c_v7"

    def __init__(self, *args, pixel_hidden=32, hdr_token_bottleneck=256,
                 alignment_confidence_tau=.10, hdr_warmup_steps=1000,
                 normal_refine_iterations=3, normal_refine_step_limit=.05,
                 point_update_scale=.10,
                 **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        hidden = int(pixel_hidden)
        # VGGT's aggregator exposes concatenated local/global tokens.  Their
        # channel width is therefore 2 * embed_dim (2048 for embed_dim=1024),
        # not the transformer's internal embed_dim itself.
        token_dim = 2 * int(kwargs.get("embed_dim"))
        self.full_geo_aligner = FullToGeoAlignment(hidden)
        self.event_token_projection = nn.Conv2d(hidden, token_dim, 1)
        self.ldr_event_hdr_aligner = LdrEventToHdrTokenAligner(
            token_dim, hdr_token_bottleneck
        )
        # aligned event feature + current/base log depth + current/target
        # normal + learned event confidence.  This head predicts an explicit
        # dense log-depth residual after token alignment.
        self.normal_depth_refiner = nn.Sequential(
            nn.Conv2d(hidden + 9, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.normal_depth_refiner[-1].weight)
        nn.init.zeros_(self.normal_depth_refiner[-1].bias)
        # event feature + coarse point direction/log-radius + token-induced
        # point displacement + final normal + C.
        # This operates after the frozen point decoder and predicts a dense
        # explicit point-map residual.
        self.point_refiner = nn.Sequential(
            nn.Conv2d(hidden + 11, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.point_refiner[-1].weight)
        nn.init.zeros_(self.point_refiner[-1].bias)
        nn.init.xavier_uniform_(self.event_token_projection.weight)
        nn.init.zeros_(self.event_token_projection.bias)
        self.alignment_confidence_tau = max(float(alignment_confidence_tau), 1e-4)
        self.hdr_warmup_steps = max(0, int(hdr_warmup_steps))
        self.normal_refine_iterations = max(1, int(normal_refine_iterations))
        self.normal_refine_step_limit = max(float(normal_refine_step_limit), 1e-6)
        self.point_update_scale = max(float(point_update_scale), 1e-6)
        self._dual_alignment_step = 0

    def _decode_event_normal(self, feature):
        b, v, channels, height, width = feature.shape
        normal = self.event_normal_decoder(
            feature.reshape(b * v, channels, height, width)
        )
        return F.normalize(normal.float(), dim=1, eps=1e-6).reshape(
            b, v, 3, height, width
        ).movedim(2, -1)

    def _encode_event(self, views, voxel):
        representation, decay = self._decayed_signed(views, voxel)
        feature = self.event_encoder(representation)
        support = signed_support(representation, self.support_dilation_kernel)[:, :, 0] > 0
        return representation, decay, feature, support

    def _event_patch_tokens(self, feature, reliability, patch_count, image_hw):
        b, v, channels, height, width = feature.shape
        image_h, image_w = image_hw
        grid_h = max(1, image_h // int(self.patch_size))
        grid_w = max(1, image_w // int(self.patch_size))
        if grid_h * grid_w != patch_count:
            # Some preprocessing paths pad rather than floor-divide. Infer the
            # second dimension only when the exact standard grid is unavailable.
            grid_h = max(1, round(patch_count ** .5))
            while grid_h > 1 and patch_count % grid_h:
                grid_h -= 1
            grid_w = patch_count // grid_h
        flat = feature.reshape(b * v, channels, height, width)
        pooled = F.adaptive_avg_pool2d(flat, (grid_h, grid_w))
        projected = self.event_token_projection(pooled)
        token = projected.flatten(2).transpose(1, 2).reshape(
            b, v, patch_count, projected.shape[1]
        )
        reliability_patch = F.adaptive_avg_pool2d(
            reliability.reshape(b * v, 1, height, width), (grid_h, grid_w)
        ).flatten(2).transpose(1, 2).reshape(b, v, patch_count, 1)
        return token * reliability_patch, reliability_patch

    def _fuse_tokens(self, tokens_list, patch_start, event_token):
        fused, residuals = [], []
        for tokens in tokens_list:
            special = tokens[:, :, :patch_start]
            patch = tokens[:, :, patch_start:]
            aligned, residual = self.ldr_event_hdr_aligner(patch, event_token)
            fused.append(torch.cat((special, aligned), dim=2))
            residuals.append(residual)
        return fused, residuals

    def forward(self, views, query_points=None, **_kwargs):
        images, full_voxel, intrinsics = self._stack_inputs(views)
        full_voxel = full_voxel.to(images.device)
        intrinsics = intrinsics.to(images.device).float()
        b, v, _, image_h, image_w = images.shape

        full_repr, decay, full_feature, full_support = self._encode_event(
            views, full_voxel
        )
        geo_fields = [view.get("geometry_event_voxel") for view in views]
        have_geo = all(torch.is_tensor(value) for value in geo_fields)
        if self.training and not have_geo:
            raise RuntimeError("dual alignment training requires geometry_event_voxel")
        if have_geo:
            geo_voxel = torch.stack(geo_fields, dim=1).to(full_voxel)
            _, _, geo_feature, geo_support = self._encode_event(views, geo_voxel)
        else:
            geo_feature, geo_support = None, full_support

        flat_full = full_feature.reshape(b * v, full_feature.shape[2], image_h, image_w)
        aligned_flat, correction_flat, _ = self.full_geo_aligner(flat_full)
        aligned_feature = aligned_flat.reshape_as(full_feature)
        correction = correction_flat.reshape_as(full_feature)
        full_normal = self._decode_event_normal(aligned_feature)
        geo_normal = self._decode_event_normal(geo_feature) if geo_feature is not None else None

        # C is always predicted from inference-available inputs.  E_geo is
        # never an input to this predictor: it is used below only to construct
        # a training-time reliability target.
        ldr_tokens, patch_start = self.aggregator(images)
        with torch.no_grad():
            raw_depth, raw_conf = self.depth_head(
                ldr_tokens, images=images, patch_start_idx=patch_start,
                frames_chunk_size=self.head_frames_chunk_size,
            )
        scale = self.metric_depth_scale
        coarse = raw_depth[..., 0] * scale
        coarse_normal = depth_to_normals(coarse.float(), intrinsics)
        reliability = self.contribution_net(
            full_repr,
            images,
            coarse.detach(),
            coarse_normal.detach(),
        )

        if geo_feature is not None:
            event_feature_error = F.smooth_l1_loss(
                aligned_feature, geo_feature.detach(), beta=.05, reduction="none"
            ).mean(dim=2)
            reliability_target = torch.exp(
                -event_feature_error.detach() / self.alignment_confidence_tau
            ).clamp(0, 1) * (full_support | geo_support).float()
            reliability_target_available = True
        else:
            event_feature_error = reliability.new_zeros(reliability.shape)
            # Inference has no E_geo, hence no alignment-derived target.  Keep
            # only a shape-compatible diagnostic tensor; it is never used to
            # produce C or gate an inference feature.
            reliability_target = reliability.new_zeros(reliability.shape)
            reliability_target_available = False

        # Student LDR tokens, gated only by predicted C (never by E_geo).
        patch_count = ldr_tokens[-1].shape[2] - patch_start
        event_token, reliability_patch = self._event_patch_tokens(
            aligned_feature, reliability, patch_count, (image_h, image_w)
        )
        hdr_pred_tokens, hdr_residuals = self._fuse_tokens(
            ldr_tokens, patch_start, event_token
        )

        # Training-only teacher: the less saturated paired image is attached as
        # hdr_img by the trainer. If a real HDR field is added later it can be
        # placed in the same key without changing this model.
        hdr_fields = [view.get("hdr_img") for view in views]
        have_hdr = all(torch.is_tensor(value) for value in hdr_fields)
        if self.training and not have_hdr:
            raise RuntimeError("dual alignment training requires hdr_img teacher")
        if have_hdr:
            hdr_images = torch.stack(hdr_fields, dim=1).to(images)
            with torch.no_grad():
                hdr_teacher_tokens, teacher_patch_start = self.aggregator(hdr_images)
            if teacher_patch_start != patch_start:
                raise RuntimeError("LDR/HDR patch token layouts differ")
            hdr_token_error = F.smooth_l1_loss(
                hdr_pred_tokens[-1][:, :, patch_start:],
                hdr_teacher_tokens[-1][:, :, patch_start:].detach(),
                beta=.05, reduction="none",
            ).mean(dim=-1)
        else:
            hdr_images = images
            hdr_teacher_tokens = None
            hdr_token_error = reliability_patch[..., 0] * 0.0

        # Raw LDR depth is the coarse baseline; HDR-aligned tokens drive the
        # final geometry heads. All pretrained heads/backbone remain frozen.
        hdr_depth, hdr_conf = self.depth_head(
            hdr_pred_tokens, images=images, patch_start_idx=patch_start,
            frames_chunk_size=self.head_frames_chunk_size,
        )
        pose = self.camera_head(ldr_tokens)[-1]
        with torch.no_grad():
            coarse_points, coarse_point_conf = self.point_head(
                ldr_tokens, images=images, patch_start_idx=patch_start,
                frames_chunk_size=self.head_frames_chunk_size,
            )
        hdr_points, point_conf = self.point_head(
            hdr_pred_tokens, images=images, patch_start_idx=patch_start,
            frames_chunk_size=self.head_frames_chunk_size,
        )
        hdr_map = hdr_depth[..., 0] * scale
        hdr_base_normal = depth_to_normals(hdr_map.float(), intrinsics)

        # The event branch predicts an absolute normal.  Express it as a
        # residual around the HDR-token base normal, then apply the learned
        # alignment confidence softly.  Reliability is trained toward zero in
        # empty-event regions; there is no hard support multiplication here.
        event_normal_delta = full_normal - hdr_base_normal
        normal_refine_target = F.normalize(
            hdr_base_normal + reliability.unsqueeze(-1) * event_normal_delta,
            dim=-1, eps=1e-6,
        )
        if self.training:
            self._dual_alignment_step += 1
        warmup = self.training and self._dual_alignment_step <= self.hdr_warmup_steps
        if warmup:
            # Keep every student module in DDP while geometry is scale-only
            # and both alignment objectives learn a stable representation.
            zero = sum((value.sum() * 0.0 for value in hdr_residuals), coarse.new_zeros(()))
            for parameter in self.normal_depth_refiner.parameters():
                zero = zero + parameter.sum() * 0.0
            final = coarse + zero
            geometry_ratio = torch.zeros_like(coarse) + zero
            geometry_update = torch.zeros_like(coarse) + zero
            iteration_updates = None
        else:
            bv = b * v
            log_base = torch.log(hdr_map.clamp_min(1e-6)).reshape(
                bv, 1, image_h, image_w
            )
            log_depth = log_base.clone()
            feature_flat = aligned_feature.reshape(
                bv, aligned_feature.shape[2], image_h, image_w
            ).float()
            target_flat = normal_refine_target.movedim(-1, 2).reshape(
                bv, 3, image_h, image_w
            )
            confidence_flat = reliability.reshape(bv, 1, image_h, image_w)
            intrinsics_flat = intrinsics.reshape(bv, 3, 3)
            steps = []
            for _ in range(self.normal_refine_iterations):
                current_depth = torch.exp(log_depth[:, 0])
                current_normal = depth_to_normals(
                    current_depth.unsqueeze(1), intrinsics_flat.unsqueeze(1)
                )[:, 0].movedim(-1, 1)
                refine_input = torch.cat((
                    feature_flat, log_depth, log_base, current_normal,
                    target_flat, confidence_flat,
                ), dim=1)
                # Subtract the no-normal-residual response.  Consequently the
                # geometry branch cannot invent a depth update from a constant
                # bias when the target and current normals already agree.
                baseline_input = torch.cat((
                    feature_flat, log_depth, log_base, current_normal,
                    current_normal, confidence_flat,
                ), dim=1)
                raw_step = (
                    self.normal_depth_refiner(refine_input)
                    - self.normal_depth_refiner(baseline_input)
                )
                step = self.normal_refine_step_limit * torch.tanh(
                    raw_step / self.normal_refine_step_limit
                )
                log_depth = log_depth + step
                steps.append(step)

            total_limit = min(max(float(self.depth_update_scale), 1e-6), .999)
            log_ratio = (log_depth - log_base).clamp(
                min=math.log(1.0 - total_limit),
                max=math.log(1.0 + total_limit),
            )
            final = torch.exp(log_base + log_ratio)[:, 0].reshape(
                b, v, image_h, image_w
            )
            geometry_ratio = final / hdr_map.clamp_min(1e-6) - 1.0
            geometry_update = final - hdr_map
            iteration_updates = torch.cat(steps, dim=1).reshape(
                b, v, self.normal_refine_iterations, image_h, image_w
            )
        final_normal = depth_to_normals(final.float(), intrinsics)

        # Decoder-after-token alignment gives the HDR point base.  A separate
        # dense point refiner then learns the residual under direct point-map
        # supervision.  It receives confidence as a feature, never as a hard
        # output mask, so corrections are not restricted to event pixels.
        coarse_point_radius = coarse_points.float().norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-4)
        coarse_point_direction = coarse_points.float() / coarse_point_radius
        point_token_update = hdr_points - coarse_points
        point_token_delta_ratio = point_token_update.float() / coarse_point_radius
        point_radius = hdr_points.float().norm(dim=-1, keepdim=True).clamp_min(1e-4)
        point_input = torch.cat((
            aligned_feature.float(),
            coarse_point_direction.movedim(-1, 2),
            torch.log(coarse_point_radius).movedim(-1, 2),
            point_token_delta_ratio.movedim(-1, 2),
            final_normal.movedim(-1, 2),
            reliability.unsqueeze(2),
        ), dim=2)
        raw_point_ratio = self.point_refiner(
            point_input.reshape(b * v, point_input.shape[2], image_h, image_w)
        ).reshape(b, v, 3, image_h, image_w).movedim(2, -1)
        bounded_point_ratio = self.point_update_scale * torch.tanh(
            raw_point_ratio / self.point_update_scale
        )
        predicted_point_update = point_radius.detach() * bounded_point_ratio
        if warmup:
            point_zero = coarse_points.new_zeros(())
            for parameter in self.point_refiner.parameters():
                point_zero = point_zero + parameter.sum() * 0.0
            points = coarse_points + point_zero
            point_update = torch.zeros_like(coarse_points) + point_zero
            applied_point_ratio = torch.zeros_like(raw_point_ratio) + point_zero
        else:
            points = hdr_points + predicted_point_update
            point_update = predicted_point_update
            applied_point_ratio = bounded_point_ratio
        point_total_update = points - coarse_points

        token_update = hdr_map - coarse
        total_update = final - coarse
        tv = .5 * (
            (geometry_ratio[..., :, 1:] - geometry_ratio[..., :, :-1]).abs().mean()
            + (geometry_ratio[..., 1:, :] - geometry_ratio[..., :-1, :]).abs().mean()
        )
        point_dx = applied_point_ratio[..., :, 1:, :] - applied_point_ratio[..., :, :-1, :]
        point_dy = applied_point_ratio[..., 1:, :, :] - applied_point_ratio[..., :-1, :, :]
        point_tv = .5 * (point_dx.abs().mean() + point_dy.abs().mean())
        geometry_regularizer = tv + point_tv
        results = []
        for index in range(v):
            results.append(dict(
                pts3d_in_other_view=points[:, index], conf=point_conf[:, index],
                pts3d_coarse=coarse_points[:, index],
                pts3d_coarse_conf=coarse_point_conf[:, index],
                pts3d_hdr_base=hdr_points[:, index],
                point_token_map_update=point_token_update[:, index],
                point_token_delta_ratio=point_token_delta_ratio[:, index],
                point_pixel_update=point_update[:, index],
                point_total_update=point_total_update[:, index],
                point_delta_ratio=applied_point_ratio[:, index],
                point_update_tv=point_tv,
                depth=final[:, index].unsqueeze(-1), normal=final_normal[:, index],
                depth_conf=hdr_conf[:, index], depth_coarse=coarse[:, index].unsqueeze(-1),
                depth_coarse_raw=raw_depth[:, index], depth_coarse_conf=raw_conf[:, index],
                camera_pose=pose[:, index], metric_depth_scale=scale,
                event_contribution=reliability[:, index],
                event_contribution_spatial=reliability[:, index],
                event_normal=full_normal[:, index],
                event_normal_full=full_normal[:, index],
                event_normal_geo=(geo_normal[:, index] if geo_normal is not None else full_normal[:, index].detach()),
                event_normal_reliability=reliability[:, index],
                event_normal_support=full_support[:, index],
                geo_event_support=geo_support[:, index],
                signed_event=full_repr[:, index], temporal_decay_weights=decay[:, index],
                alignment_feature_error=event_feature_error[:, index],
                alignment_reliability_target=reliability_target[:, index],
                alignment_reliability_target_available=final.new_tensor(
                    float(reliability_target_available)
                ),
                alignment_feature_correction=correction[:, index],
                hdr_token_alignment_error=hdr_token_error[:, index],
                hdr_token_reliability=reliability_patch[:, index, :, 0],
                hdr_token_update=hdr_residuals[-1][:, index],
                hdr_warmup_active=final.new_tensor(float(warmup)),
                depth_hdr_base=hdr_map[:, index].unsqueeze(-1),
                depth_token_map_update=token_update[:, index],
                depth_geometry_update=geometry_update[:, index],
                depth_delta_ratio=geometry_ratio[:, index],
                depth_pixel_update=geometry_update[:, index],
                depth_total_update=total_update[:, index],
                depth_update_final_absolute=geometry_update[:, index],
                depth_update_centered_ratio=geometry_ratio[:, index],
                depth_update_detail_ratio=geometry_ratio[:, index],
                depth_update_tv=tv, adapter_update_loss=geometry_regularizer,
                adapter_alpha_depth=final.new_zeros(4), adapter_alpha_point=final.new_zeros(4),
                adapter_depth_update_magnitudes=final.new_zeros(4),
                adapter_point_update_magnitudes=final.new_zeros(4),
                selected_event_mass=full_repr[:, index].abs().sum(1),
                coarse_normal=coarse_normal[:, index],
                event_normal_delta=event_normal_delta[:, index],
                normal_confidence=reliability[:, index],
                learned_normal_confidence=reliability[:, index],
                normal_refine_target=normal_refine_target[:, index],
                normal_refine_active=final.new_tensor(float(not warmup)),
                **({"normal_refine_iteration_updates": iteration_updates[:, index]}
                   if iteration_updates is not None else {}),
            ))
        return GeometryAdapterOutput(ress=results, views=views)


__all__ = ["DualAlignmentHDRLinearVoxelModel", "LdrEventToHdrTokenAligner"]
