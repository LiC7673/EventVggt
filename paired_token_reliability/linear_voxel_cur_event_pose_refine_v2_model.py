"""Spatial event-conditioned SE(3) refinement after the RGB camera head."""
from __future__ import annotations

import torch
from torch import nn

import finetune_event as fe
from paired_token_reliability.linear_voxel_cur_event_hf_residual_model import (
    CurEventHFResidualModel,
)
from paired_token_reliability.linear_voxel_cur_event_pose_refine_model import (
    _axis_angle_to_matrix,
)


class SpatialEventPoseHead(nn.Module):
    """Preserve event layout and predict anchor-relative pose corrections."""

    def __init__(self, event_channels=10, feature_dim=96, hidden=192):
        super().__init__()
        # raw event bins + normal-derivative magnitude + C + normalized depth
        in_channels = int(event_channels) + 3
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2),
            nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, feature_dim, 3, 2, 1),
            nn.GroupNorm(8, feature_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # current, anchor and current-anchor spatial features + base pose.
        self.regressor = nn.Sequential(
            nn.Linear(3 * feature_dim + 9, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 6),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, image_features, base_pose):
        b, v, c, h, w = image_features.shape
        encoded = self.spatial_encoder(
            image_features.reshape(b * v, c, h, w)
        ).flatten(1).reshape(b, v, -1)
        anchor = encoded[:, :1].expand_as(encoded)
        pair = torch.cat((encoded, anchor, encoded - anchor, base_pose), -1)
        return self.regressor(pair), encoded


class CurEventPoseRefineV2Model(CurEventHFResidualModel):
    """Final pose = frozen RGB pose composed with spatial event SE(3) residual."""

    checkpoint_schema = "cur_event_hf_pose_refine_spatial_v2"

    def __init__(
        self, *args, voxel_bins=5, pose_feature_dim=96,
        pose_refiner_hidden=192, pose_refiner_delay=0,
        pose_refiner_transition=500, pose_translation_limit=0.25,
        pose_rotation_limit_deg=10.0, **kwargs,
    ):
        super().__init__(*args, voxel_bins=voxel_bins, **kwargs)
        self.pose_spatial_refiner = SpatialEventPoseHead(
            2 * int(voxel_bins), int(pose_feature_dim),
            int(pose_refiner_hidden),
        )
        self.pose_refiner_delay = int(pose_refiner_delay)
        self.pose_refiner_transition = max(int(pose_refiner_transition), 1)
        self.pose_translation_limit = float(pose_translation_limit)
        self.pose_rotation_limit = (
            float(pose_rotation_limit_deg) * torch.pi / 180.0
        )

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        base_pose = torch.stack(
            [item["camera_pose"] for item in output.ress], 1
        ).float()
        b, v = base_pose.shape[:2]
        height, width = output.ress[0]["depth"].shape[-3:-1]
        event = torch.stack(
            [view["event_voxel"] for view in views], 1
        ).to(base_pose).float()
        derivative = torch.stack(
            [item["event_normal_derivative_full"] for item in output.ress], 1
        ).float().reshape(b, v, height, width, -1)
        derivative = derivative.norm(dim=-1).unsqueeze(2)
        confidence = torch.stack(
            [item["event_contribution"] for item in output.ress], 1
        ).float().unsqueeze(2)
        depth = torch.stack(
            [item["depth_hdr_base"][..., 0] for item in output.ress], 1
        ).float()
        log_depth = torch.log(depth.clamp_min(1e-6))
        center = log_depth.mean((-2, -1), keepdim=True)
        spread = log_depth.std((-2, -1), keepdim=True, unbiased=False)
        normalized_depth = ((log_depth - center) / spread.clamp_min(1e-4)).unsqueeze(2)
        spatial_input = torch.cat(
            (event, derivative, confidence, normalized_depth), dim=2
        )

        raw, encoded = self.pose_spatial_refiner(
            spatial_input, base_pose.detach()
        )
        raw = raw - raw[:, :1]  # first-view gauge
        step = int(getattr(self, "_dual_alignment_step", 0))
        coupling = max(
            0.0,
            min(1.0, (step - self.pose_refiner_delay)
                / self.pose_refiner_transition),
        )
        delta_t = self.pose_translation_limit * torch.tanh(raw[..., :3])
        delta_r = self.pose_rotation_limit * torch.tanh(raw[..., 3:])
        base_c2w, intrinsics = fe.pose_encoding_to_c2w(
            base_pose, image_size_hw=(height, width)
        )
        correction = torch.eye(
            4, device=base_pose.device, dtype=base_pose.dtype
        ).view(1, 1, 4, 4).repeat(b, v, 1, 1)
        correction[..., :3, :3] = _axis_angle_to_matrix(coupling * delta_r)
        correction[..., :3, 3] = coupling * delta_t
        refined_c2w = base_c2w @ correction
        final_pose = fe.c2w_to_pose_encoding(
            refined_c2w, intrinsics, image_size_hw=(height, width)
        )
        for index, item in enumerate(output.ress):
            item["camera_pose_base"] = base_pose[:, index]
            item["camera_pose"] = final_pose[:, index]
            item["pose_residual_translation"] = delta_t[:, index]
            item["pose_residual_rotation"] = delta_r[:, index]
            item["pose_spatial_feature"] = encoded[:, index]
            item["pose_refiner_coupling"] = base_pose.new_tensor(coupling)
        return output
