"""Staged direct pixel geometry: learn on E_geo, then align E_full to it."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_token_reliability.linear_voxel_multiscale_model import (
    LinearVoxelMultiscalePixelModel,
)
from paired_token_reliability.linear_voxel_full_geo_alignment_model import (
    FullToGeoAlignment,
)
from stage2_geometry_adapter.model import depth_to_normals


class SwitchableFullGeoEncoder(nn.Module):
    def __init__(self, backbone, channels):
        super().__init__()
        self.backbone = backbone
        self.aligner = FullToGeoAlignment(channels)
        self.use_alignment = False
        self.last_raw = None
        self.last_output = None
        self.last_correction = None

    def forward(self, voxel):
        raw = self.backbone(voxel)
        self.last_raw = raw
        if not self.use_alignment:
            self.last_output = raw
            self.last_correction = torch.zeros_like(raw)
            return raw
        b, v, c, h, w = raw.shape
        aligned, correction, _ = self.aligner(raw.reshape(b * v, c, h, w))
        self.last_output = aligned.reshape(b, v, c, h, w)
        self.last_correction = correction.reshape(b, v, c, h, w)
        return self.last_output


class StagedGeoDirect50Model(LinearVoxelMultiscalePixelModel):
    checkpoint_schema = "linear_voxel_staged_geo_direct50_full_alignment_v1"

    def __init__(self, *args, pixel_hidden=32, **kwargs):
        super().__init__(*args, pixel_hidden=pixel_hidden, **kwargs)
        self.event_encoder = SwitchableFullGeoEncoder(
            self.event_encoder, int(pixel_hidden)
        )
        self.stage_mode = "geo"

    def set_stage(self, mode):
        if mode not in {"geo", "full", "joint"}:
            raise ValueError(f"unknown staged mode {mode}")
        self.stage_mode = mode
        self.event_encoder.use_alignment = mode != "geo"

    def forward(self, views, *args, **kwargs):
        geo_fields = [view.get("geometry_event_voxel") for view in views]
        geo_teacher = None
        if self.stage_mode != "geo":
            if not all(torch.is_tensor(value) for value in geo_fields):
                raise RuntimeError("full alignment stage requires geometry_event_voxel teacher")
            geo_voxel = torch.stack(geo_fields, dim=1)
            representation, _ = self._decayed_signed(views, geo_voxel)
            with torch.no_grad():
                geo_teacher = self.event_encoder.backbone(representation)
            geo_support = representation.ne(0).any(2)
        else:
            geo_support = None

        output = super().forward(views, *args, **kwargs)
        raw_coarse = torch.stack(
            [item["depth_coarse"][..., 0] for item in output.ress], 1
        ).float()
        raw_final = torch.stack(
            [item["depth"][..., 0] for item in output.ress], 1
        ).float()
        gt_fields = [view.get("depthmap") for view in views]
        if not all(torch.is_tensor(value) for value in gt_fields):
            raise RuntimeError(
                "staged direct50 uses GT scene-scale protocol and requires depthmap at train/val/test"
            )
        gt = torch.stack(gt_fields, 1).to(raw_coarse).float()
        valid = (
            torch.isfinite(raw_coarse) & torch.isfinite(gt)
            & (raw_coarse > 1.0e-6) & (gt > 1.0e-6)
        ).float()
        dims = tuple(range(1, raw_coarse.ndim))
        scene_scale = (
            (valid * raw_coarse * gt).sum(dims)
            / (valid * raw_coarse.square()).sum(dims).clamp_min(1.0e-6)
        ).detach()
        if bool((valid.sum(dims) <= 0).any()):
            raise RuntimeError("no valid pixels for GT scene-scale alignment")
        scale = scene_scale.view(-1, *([1] * (raw_coarse.ndim - 1)))
        coarse = raw_coarse * scale
        final = raw_final * scale
        intrinsics = torch.stack(
            [view["camera_intrinsics"].to(final) for view in views], 1
        ).float()
        final_normal = depth_to_normals(final, intrinsics)
        aligned = self.event_encoder.last_output
        correction = self.event_encoder.last_correction
        if aligned is None:
            raise RuntimeError("event encoder did not capture its pixel feature")
        if geo_teacher is None:
            geo_teacher = aligned.detach()
        full_representation = torch.stack(
            [item["signed_event"] for item in output.ress], 1
        )
        full_support = full_representation.ne(0).any(2)
        if geo_support is None:
            geo_support = full_support
        alignment_support = full_support | geo_support
        feature_error = F.smooth_l1_loss(
            aligned, geo_teacher.detach(), beta=.02, reduction="none"
        ).mean(2)
        for index, item in enumerate(output.ress):
            point_scale = scene_scale.view(-1, 1, 1, 1)
            item["pts3d_in_other_view"] = item["pts3d_in_other_view"] * point_scale
            item["depth_coarse_raw"] = item["depth_coarse"]
            item["depth_coarse"] = coarse[:, index].unsqueeze(-1)
            item["depth"] = final[:, index].unsqueeze(-1)
            item["normal"] = final_normal[:, index]
            item["depth_pixel_update"] = final[:, index] - coarse[:, index]
            item["depth_total_update"] = final[:, index] - coarse[:, index]
            item["gt_scene_scale"] = scene_scale
            item["full_geo_feature_error"] = feature_error[:, index]
            item["full_geo_feature_correction"] = correction[:, index]
            item["full_geo_alignment_support"] = alignment_support[:, index]
            item["staged_geo_teacher_available"] = item["depth"].new_tensor(
                float(self.stage_mode != "geo")
            )
        return output


__all__ = ["StagedGeoDirect50Model", "SwitchableFullGeoEncoder"]
