"""Stage-2 contribution-guided adapters inside the VGGT DPT geometry heads.

There is deliberately no depth-residual module in this file.  The frozen
Stage-1 ContributionNet selects events, a polarity/temporal encoder builds a
feature pyramid, and zero-initialized residual *feature adapters* refine the
four DPT inputs.  The original depth and point heads then decode geometry.
The camera head remains on the unmodified RGB token path.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import ModelOutput

from paired_token_reliability.common import infer_patch_grid, torch_load
from paired_token_reliability.contribution_stage1 import (
    MultiLdrEventContributionModel,
    build_model_from_checkpoint,
    normalize_event_voxel,
)
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead, custom_interpolate
from streamvggt.heads.head_act import activate_head
from streamvggt.heads.track_head import TrackHead
from streamvggt.models.aggregator import Aggregator


@dataclass
class GeometryAdapterOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[List[dict]] = None


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def depth_to_normals(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Differentiable perspective depth-to-normal conversion."""
    *batch_dims, height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    shape = [1] * len(batch_dims) + [height, width]
    xs = xs.view(*shape)
    ys = ys.view(*shape)
    fx = intrinsics[..., 0, 0].view(*batch_dims, 1, 1).clamp_min(1.0e-6)
    fy = intrinsics[..., 1, 1].view(*batch_dims, 1, 1).clamp_min(1.0e-6)
    cx = intrinsics[..., 0, 2].view(*batch_dims, 1, 1)
    cy = intrinsics[..., 1, 2].view(*batch_dims, 1, 1)
    points = torch.stack(((xs - cx) * depth / fx, (ys - cy) * depth / fy, depth), dim=-1)
    dx = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
    dy = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
    core = F.normalize(torch.cross(dy, dx, dim=-1), dim=-1, eps=1.0e-6)
    normals = torch.zeros_like(points)
    normals[..., 1:-1, 1:-1, :] = core
    return normals


class PolarityTemporalEventPyramid(nn.Module):
    """Encode only selected events while retaining polarity and bin order."""

    def __init__(
        self,
        num_bins: int,
        hidden_channels: int = 48,
        pyramid_channels: int = 64,
        count_cmax: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.count_cmax = float(count_cmax)
        polarity_channels = max(hidden_channels // 2, 8)
        self.positive = nn.Sequential(
            nn.Conv3d(2, polarity_channels, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.GroupNorm(_group_count(polarity_channels), polarity_channels),
            nn.GELU(),
        )
        self.negative = nn.Sequential(
            nn.Conv3d(2, polarity_channels, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.GroupNorm(_group_count(polarity_channels), polarity_channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(2 * polarity_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.temporal_score = nn.Conv3d(hidden_channels, 1, 1, bias=False)
        self.scale_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_channels, pyramid_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(pyramid_channels), pyramid_channels),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )

    def forward(
        self,
        selected_event: torch.Tensor,
        contribution: torch.Tensor,
        target_shapes: Sequence[Tuple[int, int]],
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch, views, channels, height, width = selected_event.shape
        if channels != 2 * self.num_bins:
            raise ValueError(f"Expected {2 * self.num_bins} event channels, got {channels}")
        if len(target_shapes) != 4:
            raise ValueError(f"DPT requires four target shapes, got {target_shapes}")
        normalized = normalize_event_voxel(selected_event, self.count_cmax)
        positive = normalized[:, :, : self.num_bins]
        negative = normalized[:, :, self.num_bins :]
        time = torch.linspace(
            -1.0, 1.0, self.num_bins, device=selected_event.device, dtype=normalized.dtype
        ).view(1, 1, self.num_bins, 1, 1)
        positive = torch.stack((positive, positive * time), dim=2).reshape(
            batch * views, 2, self.num_bins, height, width
        )
        negative = torch.stack((negative, negative * time), dim=2).reshape(
            batch * views, 2, self.num_bins, height, width
        )
        feature = self.temporal(torch.cat((self.positive(positive), self.negative(negative)), dim=1))
        attention = torch.softmax(self.temporal_score(feature), dim=2)
        feature_2d = (feature * attention).sum(dim=2)

        # A spatial support factor makes zero-event and C=0 forwards exactly
        # equal to the RGB path even if downstream adapters contain biases.
        support = (selected_event.abs().sum(dim=2) > 0.0).float()
        effective_contribution = contribution.float().clamp(0.0, 1.0) * support
        event_pyramid: List[torch.Tensor] = []
        contribution_pyramid: List[torch.Tensor] = []
        for projection, shape in zip(self.scale_projections, target_shapes):
            event_feature = projection(feature_2d)
            event_feature = F.interpolate(event_feature, size=shape, mode="bilinear", align_corners=False)
            event_pyramid.append(
                event_feature.reshape(batch, views, event_feature.shape[1], shape[0], shape[1])
            )
            gate = F.interpolate(
                effective_contribution.reshape(batch * views, 1, height, width),
                size=shape,
                mode="bilinear",
                align_corners=False,
            )
            contribution_pyramid.append(gate.reshape(batch, views, 1, shape[0], shape[1]))
        return event_pyramid, contribution_pyramid


class GeometryFeatureAdapter(nn.Module):
    """A small convolutional adapter with an exactly-zero initial feature update."""

    def __init__(self, rgb_channels: int, event_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden = max(min(int(hidden_channels), int(rgb_channels)), 32)
        self.adapter = nn.Sequential(
            nn.Conv2d(rgb_channels + event_channels, hidden, 1),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=_group_count(hidden)),
            nn.GELU(),
            nn.Conv2d(hidden, rgb_channels, 1),
        )
        nn.init.normal_(self.adapter[-1].weight, std=1.0e-2)
        nn.init.zeros_(self.adapter[-1].bias)
        self.alpha_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        rgb_feature: torch.Tensor,
        event_feature: torch.Tensor,
        contribution: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_update = self.adapter(torch.cat((rgb_feature, event_feature), dim=1))
        alpha = torch.tanh(self.alpha_logit)
        applied_update = alpha * contribution * raw_update
        refined = rgb_feature + applied_update
        low_contribution_penalty = ((1.0 - contribution) * applied_update).abs().mean()
        return refined, applied_update, low_contribution_penalty


class GeometryAdapterDPTHead(DPTHead):
    """Original DPT head with adapters at its four projected dense feature scales."""

    def __init__(
        self,
        *args,
        event_feature_channels: int = 64,
        adapter_hidden_channels: int = 128,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        **kwargs,
    ) -> None:
        out_channels = list(out_channels)
        super().__init__(*args, out_channels=out_channels, **kwargs)
        self.geometry_adapters = nn.ModuleList(
            [
                GeometryFeatureAdapter(channels, event_feature_channels, adapter_hidden_channels)
                for channels in out_channels
            ]
        )
        self.last_update_loss: Optional[torch.Tensor] = None
        self.last_update_magnitudes: List[torch.Tensor] = []

    def forward(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx,
        frames_chunk_size=8,
        *,
        event_pyramid: Optional[Sequence[torch.Tensor]] = None,
        contribution_pyramid: Optional[Sequence[torch.Tensor]] = None,
    ):
        if event_pyramid is None or contribution_pyramid is None:
            self.last_update_loss = images.new_zeros(())
            self.last_update_magnitudes = [images.new_zeros(()) for _ in range(4)]
            return super().forward(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=frames_chunk_size,
            )
        batch, views = images.shape[:2]
        chunk_size = views if frames_chunk_size is None else int(frames_chunk_size)
        all_predictions = []
        all_confidences = []
        update_losses = []
        magnitude_sums = [images.new_zeros(()) for _ in range(4)]
        chunk_count = 0
        for start in range(0, views, max(chunk_size, 1)):
            end = min(start + max(chunk_size, 1), views)
            prediction, confidence, update_loss, magnitudes = self._forward_adapter_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                event_pyramid,
                contribution_pyramid,
                start,
                end,
            )
            all_predictions.append(prediction)
            all_confidences.append(confidence)
            update_losses.append(update_loss)
            magnitude_sums = [total + value for total, value in zip(magnitude_sums, magnitudes)]
            chunk_count += 1
        self.last_update_loss = torch.stack(update_losses).mean()
        self.last_update_magnitudes = [value / max(chunk_count, 1) for value in magnitude_sums]
        return torch.cat(all_predictions, dim=1), torch.cat(all_confidences, dim=1)

    def _forward_adapter_impl(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx,
        event_pyramid,
        contribution_pyramid,
        frames_start_idx,
        frames_end_idx,
    ):
        images_chunk = images[:, frames_start_idx:frames_end_idx].contiguous()
        batch, views, _, height, width = images_chunk.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        features = []
        penalties = []
        magnitudes = []
        for dpt_index, layer_index in enumerate(self.intermediate_layer_idx):
            tokens = aggregated_tokens_list[layer_index][
                :, frames_start_idx:frames_end_idx, patch_start_idx:
            ]
            tokens = tokens.reshape(batch * views, -1, tokens.shape[-1])
            tokens = self.norm(tokens)
            tokens = tokens.permute(0, 2, 1).reshape(
                tokens.shape[0], tokens.shape[-1], patch_h, patch_w
            )
            rgb_feature = self.projects[dpt_index](tokens)
            if self.pos_embed:
                rgb_feature = self._apply_pos_embed(rgb_feature, width, height)
            rgb_feature = self.resize_layers[dpt_index](rgb_feature)
            event_feature = event_pyramid[dpt_index][
                :, frames_start_idx:frames_end_idx
            ].reshape(batch * views, event_pyramid[dpt_index].shape[2], *rgb_feature.shape[-2:])
            contribution = contribution_pyramid[dpt_index][
                :, frames_start_idx:frames_end_idx
            ].reshape(batch * views, 1, *rgb_feature.shape[-2:])
            refined, update, penalty = self.geometry_adapters[dpt_index](
                rgb_feature, event_feature.to(rgb_feature.dtype), contribution.to(rgb_feature.dtype)
            )
            features.append(refined)
            penalties.append(penalty)
            magnitudes.append(update.detach().abs().mean())

        fused = self.scratch_forward(features)
        fused = custom_interpolate(
            fused,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, width, height)
        output = self.scratch.output_conv2(fused)
        predictions, confidence = activate_head(
            output, activation=self.activation, conf_activation=self.conf_activation
        )
        predictions = predictions.reshape(batch, views, *predictions.shape[1:])
        confidence = confidence.reshape(batch, views, *confidence.shape[1:])
        return predictions, confidence, torch.stack(penalties).mean(), magnitudes


def dpt_feature_shapes(height: int, width: int, patch_size: int) -> List[Tuple[int, int]]:
    patch_h, patch_w = height // patch_size, width // patch_size
    return [
        (patch_h * 4, patch_w * 4),
        (patch_h * 2, patch_w * 2),
        (patch_h, patch_w),
        ((patch_h + 1) // 2, (patch_w + 1) // 2),
    ]


class StreamVGGT(nn.Module):
    """VGGT with Stage-1 contribution and DPT-only multi-scale event adapters."""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        event_hidden_dim: int = 48,
        head_frames_chunk_size: int = 2,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        stage1_checkpoint: Optional[str] = None,
        event_pyramid_channels: int = 64,
        adapter_hidden_channels: int = 128,
        # Alias used by the historical held-out evaluator CLI.
        reliability_checkpoint: Optional[str] = None,
        **_legacy_unused,
    ) -> None:
        super().__init__()
        stage1_checkpoint = stage1_checkpoint or reliability_checkpoint
        if not stage1_checkpoint:
            raise ValueError("A Stage-1 event-contribution checkpoint is required.")
        self.stage1_checkpoint = str(Path(stage1_checkpoint).expanduser().resolve())
        raw_stage1 = torch_load(self.stage1_checkpoint)
        if raw_stage1.get("supervision_region") != "bridge":
            raise ValueError(
                "Stage 2 requires the full Stage-1 bridge checkpoint, not the event-support-only ablation."
            )
        if raw_stage1.get("training_phase") not in {"contribution", "joint"}:
            raise ValueError(
                "Stage 2 requires Stage 1-B checkpoint-best.pth or an explicitly selected "
                "short Stage 1-C joint checkpoint. A proxy-only checkpoint has no learned contribution map."
            )
        stage1_model = build_model_from_checkpoint(raw_stage1)
        if int(stage1_model.architecture["num_bins"]) != int(event_num_bins):
            raise ValueError(
                f"Stage-1 bins={stage1_model.architecture['num_bins']} but Stage-2 bins={event_num_bins}"
            )
        expected_feature_dim = int(stage1_model.architecture["coarse_feature_dim"])
        if expected_feature_dim not in (0, 2 * int(embed_dim)):
            raise ValueError(
                f"Stage-1 coarse feature dim={expected_feature_dim}, expected 0 or {2 * int(embed_dim)}"
            )
        self.contribution_net = stage1_model.contribution_net
        self.contribution_net.requires_grad_(False).eval()
        self.stage1_metadata = {
            "schema": raw_stage1.get("schema"),
            "frozen_rgb_checkpoint": raw_stage1.get("frozen_rgb_checkpoint"),
            "supervision_region": raw_stage1.get("supervision_region"),
            "training_phase": raw_stage1.get("training_phase"),
        }

        self.patch_size = int(patch_size)
        self.head_frames_chunk_size = int(head_frames_chunk_size)
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        common_head = dict(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            event_feature_channels=event_pyramid_channels,
            adapter_hidden_channels=adapter_hidden_channels,
        )
        self.depth_head = GeometryAdapterDPTHead(
            **common_head, output_dim=2, activation="exp", conf_activation="expp1"
        )
        self.point_head = GeometryAdapterDPTHead(
            **common_head, output_dim=4, activation="inv_log", conf_activation="expp1"
        )
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.event_encoder = PolarityTemporalEventPyramid(
            num_bins=event_num_bins,
            hidden_channels=event_hidden_dim,
            pyramid_channels=event_pyramid_channels,
            count_cmax=event_count_cmax,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.contribution_net.parameters()):
            self.contribution_net.eval()
        return self

    @staticmethod
    def _stack_inputs(views) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batched = views[0]["img"].ndim == 4
        if batched:
            images = torch.stack([view["img"] for view in views], dim=1)
            events = torch.stack([view["event_voxel"] for view in views], dim=1)
            intrinsics = torch.stack([view["camera_intrinsics"] for view in views], dim=1)
        else:
            images = torch.stack([view["img"] for view in views], dim=0).unsqueeze(0)
            events = torch.stack([view["event_voxel"] for view in views], dim=0).unsqueeze(0)
            intrinsics = torch.stack([view["camera_intrinsics"] for view in views], dim=0).unsqueeze(0)
        return images, events, intrinsics

    def _coarse_patch_features(
        self, tokens_list, patch_start_idx: int, height: int, width: int
    ) -> torch.Tensor:
        tokens = tokens_list[-1][:, :, patch_start_idx:]
        grid_h, grid_w = infer_patch_grid(tokens.shape[2], height, width)
        return tokens.reshape(
            tokens.shape[0], tokens.shape[1], grid_h, grid_w, tokens.shape[-1]
        ).permute(0, 1, 4, 2, 3)

    def forward(
        self,
        views,
        query_points: Optional[torch.Tensor] = None,
        contribution_override: Optional[torch.Tensor] = None,
        **_kwargs,
    ):
        images, event_voxel, intrinsics = self._stack_inputs(views)
        if query_points is not None and query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)
        event_voxel = event_voxel.to(images.device)
        intrinsics = intrinsics.to(images.device)
        tokens_list, patch_start_idx = self.aggregator(images)

        # Camera prediction is strictly RGB-only.
        with nullcontext():
            pose_encoding = self.camera_head(tokens_list)[-1]
            coarse_depth, coarse_depth_conf = self.depth_head(
                tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=self.head_frames_chunk_size,
            )
        coarse_depth_map = coarse_depth[..., 0] if coarse_depth.shape[-1] == 1 else coarse_depth.squeeze(2)
        coarse_normals = depth_to_normals(coarse_depth_map.float(), intrinsics.float())
        coarse_features = self._coarse_patch_features(
            tokens_list, patch_start_idx, images.shape[-2], images.shape[-1]
        )
        contribution = self.contribution_net(
            event_voxel,
            images,
            coarse_depth_map,
            coarse_normals,
            coarse_features if self.contribution_net.coarse_feature_dim > 0 else None,
        )
        if contribution_override is not None:
            if contribution_override.shape != contribution.shape:
                raise ValueError(
                    f"Contribution override {contribution_override.shape} != {contribution.shape}"
                )
            contribution = contribution_override.to(contribution).clamp(0.0, 1.0)
        selected_event = contribution.unsqueeze(2) * event_voxel
        shapes = dpt_feature_shapes(images.shape[-2], images.shape[-1], self.patch_size)
        event_pyramid, contribution_pyramid = self.event_encoder(
            selected_event, contribution, shapes
        )

        with nullcontext():
            depth, depth_confidence = self.depth_head(
                tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=self.head_frames_chunk_size,
                event_pyramid=event_pyramid,
                contribution_pyramid=contribution_pyramid,
            )
            world_points, point_confidence = self.point_head(
                tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=self.head_frames_chunk_size,
                event_pyramid=event_pyramid,
                contribution_pyramid=contribution_pyramid,
            )
            track = visibility = track_confidence = None
            if query_points is not None:
                track_list, visibility, track_confidence = self.track_head(
                    tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    query_points=query_points,
                )
                track = track_list[-1]

        update_loss = 0.5 * (
            self.depth_head.last_update_loss + self.point_head.last_update_loss
        )
        alpha_depth = torch.stack(
            [torch.tanh(adapter.alpha_logit) for adapter in self.depth_head.geometry_adapters]
        )
        alpha_point = torch.stack(
            [torch.tanh(adapter.alpha_logit) for adapter in self.point_head.geometry_adapters]
        )
        depth_update_magnitudes = torch.stack(self.depth_head.last_update_magnitudes)
        point_update_magnitudes = torch.stack(self.point_head.last_update_magnitudes)
        results = []
        for frame_index in range(images.shape[1]):
            item = {
                "pts3d_in_other_view": world_points[:, frame_index],
                "conf": point_confidence[:, frame_index],
                "depth": depth[:, frame_index],
                "depth_conf": depth_confidence[:, frame_index],
                "depth_coarse": coarse_depth[:, frame_index],
                "depth_coarse_conf": coarse_depth_conf[:, frame_index],
                "camera_pose": pose_encoding[:, frame_index],
                "event_contribution": contribution[:, frame_index],
                "selected_event_mass": selected_event[:, frame_index].abs().sum(dim=1),
                "adapter_update_loss": update_loss,
                "adapter_alpha_depth": alpha_depth,
                "adapter_alpha_point": alpha_point,
                "adapter_depth_update_magnitudes": depth_update_magnitudes,
                "adapter_point_update_magnitudes": point_update_magnitudes,
                **({"valid_mask": views[frame_index]["valid_mask"]} if "valid_mask" in views[frame_index] else {}),
            }
            if track is not None:
                item.update(
                    {
                        "track": track[:, frame_index],
                        "vis": visibility[:, frame_index],
                        "track_conf": track_confidence[:, frame_index],
                    }
                )
            results.append(item)
        return GeometryAdapterOutput(ress=results, views=views)


__all__ = [
    "GeometryAdapterDPTHead",
    "GeometryAdapterOutput",
    "GeometryFeatureAdapter",
    "PolarityTemporalEventPyramid",
    "StreamVGGT",
    "dpt_feature_shapes",
]
