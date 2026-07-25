"""Cur-event geometry model with a late event-conditioned SE(3) pose refiner."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import finetune_event as fe
from paired_token_reliability.linear_voxel_cur_event_hf_residual_model import (
    CurEventHFResidualModel,
)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def _axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues map with a second-order small-angle approximation."""
    angle = vector.norm(dim=-1, keepdim=True)
    axis = vector / angle.clamp_min(1.0e-8)
    k = _skew(axis)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    eye = eye.view(*([1] * (vector.ndim - 1)), 3, 3)
    sine = torch.sin(angle)[..., None]
    cosine = torch.cos(angle)[..., None]
    regular = eye + sine * k + (1.0 - cosine) * (k @ k)
    small_k = _skew(vector)
    small = eye + small_k + 0.5 * (small_k @ small_k)
    return torch.where((angle < 1.0e-4)[..., None], small, regular)


class EventPoseResidualHead(nn.Module):
    """Predict a conservative pose correction from event and coarse geometry."""

    def __init__(self, event_channels=10, hidden=128):
        super().__init__()
        # pose(9), per-channel event mean/std(2C), depth mean/std(2),
        # dNormal magnitude mean/std(2), and relevance mean/std(2).
        input_dim = 9 + 2 * int(event_channels) + 6
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)


class CurEventPoseRefineModel(CurEventHFResidualModel):
    """Adds a delayed pose-residual path after the frozen camera head."""

    checkpoint_schema = "cur_event_hf_pose_refine_v1"

    def __init__(
        self,
        *args,
        voxel_bins=5,
        pose_refiner_hidden=128,
        pose_refiner_delay=1000,
        pose_refiner_transition=1000,
        pose_translation_limit=0.10,
        pose_rotation_limit_deg=5.0,
        **kwargs,
    ):
        super().__init__(*args, voxel_bins=voxel_bins, **kwargs)
        self.pose_residual_head = EventPoseResidualHead(
            event_channels=2 * int(voxel_bins), hidden=int(pose_refiner_hidden)
        )
        self.pose_refiner_delay = int(pose_refiner_delay)
        self.pose_refiner_transition = max(int(pose_refiner_transition), 1)
        self.pose_translation_limit = float(pose_translation_limit)
        self.pose_rotation_limit = float(pose_rotation_limit_deg) * torch.pi / 180.0

    @staticmethod
    def _mean_std(value: torch.Tensor, dimensions) -> torch.Tensor:
        mean = value.mean(dim=dimensions)
        std = value.std(dim=dimensions, unbiased=False)
        return torch.cat((mean, std), dim=-1)

    def forward(self, views, *args, **kwargs):
        output = super().forward(views, *args, **kwargs)
        pose = torch.stack([item["camera_pose"] for item in output.ress], 1).float()
        b, v = pose.shape[:2]
        height, width = output.ress[0]["depth"].shape[-3:-1]

        event = torch.stack([view["event_voxel"] for view in views], 1).to(pose).float()
        event_stats = torch.cat(
            (event.mean((-2, -1)), event.std((-2, -1), unbiased=False)), dim=-1
        )
        depth = torch.stack([item["depth_hdr_base"][..., 0] for item in output.ress], 1).float()
        log_depth = torch.log(depth.clamp_min(1.0e-6)).unsqueeze(-1)
        depth_stats = self._mean_std(log_depth, (2, 3))
        derivative = torch.stack(
            [item["event_normal_derivative_full"] for item in output.ress], 1
        ).float().reshape(b, v, height, width, -1)
        derivative_stats = self._mean_std(derivative.norm(dim=-1, keepdim=True), (2, 3))
        confidence = torch.stack(
            [item["event_contribution"] for item in output.ress], 1
        ).float().unsqueeze(-1)
        confidence_stats = self._mean_std(confidence, (2, 3))
        features = torch.cat(
            (pose.detach(), event_stats, depth_stats.detach(),
             derivative_stats, confidence_stats), dim=-1
        )

        step = int(getattr(self, "_dual_alignment_step", 0))
        coupling = max(
            0.0,
            min(1.0, (step - self.pose_refiner_delay) / self.pose_refiner_transition),
        )
        raw = self.pose_residual_head(features)
        # The first camera fixes the clip gauge. Corrections are relative to it.
        raw = raw - raw[:, :1]
        delta_t = self.pose_translation_limit * torch.tanh(raw[..., :3])
        delta_r = self.pose_rotation_limit * torch.tanh(raw[..., 3:])

        base_c2w, intrinsics = fe.pose_encoding_to_c2w(
            pose, image_size_hw=(height, width)
        )
        correction = torch.eye(4, device=pose.device, dtype=pose.dtype)
        correction = correction.view(1, 1, 4, 4).repeat(b, v, 1, 1)
        correction[..., :3, :3] = _axis_angle_to_matrix(coupling * delta_r)
        correction[..., :3, 3] = coupling * delta_t
        refined_c2w = base_c2w @ correction
        refined_pose = fe.c2w_to_pose_encoding(
            refined_c2w, intrinsics, image_size_hw=(height, width)
        )

        for index, item in enumerate(output.ress):
            item["camera_pose_base"] = pose[:, index]
            item["camera_pose"] = refined_pose[:, index]
            item["pose_residual_translation"] = delta_t[:, index]
            item["pose_residual_rotation"] = delta_r[:, index]
            item["pose_refiner_coupling"] = pose.new_tensor(coupling)
        return output
