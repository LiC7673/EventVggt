"""StreamVGGT with a dense temporal-event detail residual branch.

Unlike token-level event fusion, this branch keeps event localization in image
space and applies a bounded log-depth residual after the RGB depth prediction.
It is intended for fine geometry without introducing patch-grid shortcuts.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint
from transformers.file_utils import ModelOutput

from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from streamvggt.models.aggregator import Aggregator


@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[List[dict]] = None
    depth_coarse: Optional[torch.Tensor] = None
    depth_residual: Optional[torch.Tensor] = None


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class TemporalVoxelDetailRefiner(nn.Module):
    """Predict a smooth, bounded dense depth correction from temporal voxels."""

    def __init__(
        self,
        *,
        num_bins: int = 10,
        hidden_dim: int = 16,
        count_cmax: float = 3.0,
        residual_scale: float = 0.03,
        refine_points: bool = True,
        use_checkpoint: bool = True,
        min_depth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_bins = max(1, int(num_bins))
        self.count_cmax = max(1.0, float(count_cmax))
        self.residual_scale = float(residual_scale)
        self.refine_points = bool(refine_points)
        self.use_checkpoint = bool(use_checkpoint)
        self.min_depth = float(min_depth)
        groups = _group_count(hidden_dim)

        # Work primarily at half resolution to keep memory modest. No patch
        # striding is used, so there is no fixed token grid to imprint.
        self.event_half = nn.Sequential(
            nn.Conv2d(2 * self.num_bins, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            _ResidualBlock(hidden_dim, dilation=2),
        )
        self.context_half = nn.Sequential(
            nn.Conv2d(hidden_dim + 4, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
        )

        # Four summary maps retain full-resolution event localization while
        # the expensive temporal reasoning remains at half resolution.
        self.fine_head = nn.Sequential(
            nn.Conv2d(hidden_dim + 8, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            _ResidualBlock(hidden_dim, dilation=1),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.fine_head[-1].weight)
        nn.init.zeros_(self.fine_head[-1].bias)

    def forward(
        self,
        *,
        event_voxel: torch.Tensor,
        images: torch.Tensor,
        depth: torch.Tensor,
        points: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch, seq_len, _, height, width = event_voxel.shape
        branch_dtype = images.dtype
        voxels, event_summary = self._prepare_voxels(event_voxel.to(dtype=branch_dtype))

        image_flat = images.reshape(batch * seq_len, 3, height, width).to(dtype=branch_dtype)
        depth_flat = depth.permute(0, 1, 4, 2, 3).reshape(batch * seq_len, 1, height, width)
        log_depth = torch.log(depth_flat.clamp_min(self.min_depth)).to(dtype=branch_dtype)
        mean = log_depth.mean(dim=(-2, -1), keepdim=True)
        std = log_depth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        depth_feature = (log_depth - mean) / std

        event_half = self.event_half(voxels)
        half_size = event_half.shape[-2:]
        image_half = F.interpolate(image_flat, size=half_size, mode="bilinear", align_corners=False)
        depth_half = F.interpolate(depth_feature, size=half_size, mode="bilinear", align_corners=False)
        context = self.context_half(torch.cat([event_half, image_half, depth_half], dim=1))
        context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)

        fine_input = torch.cat([context, event_summary, image_flat, depth_feature], dim=1)
        if self.use_checkpoint and self.training and fine_input.requires_grad:
            raw_residual = checkpoint(self.fine_head, fine_input, use_reentrant=False)
        else:
            raw_residual = self.fine_head(fine_input)

        delta_log = torch.tanh(raw_residual) * self.residual_scale
        refined_depth_flat = depth_flat.to(dtype=delta_log.dtype) * torch.exp(delta_log)
        refined_depth = refined_depth_flat.permute(0, 2, 3, 1).reshape(batch, seq_len, height, width, 1)
        refined_depth = refined_depth.to(dtype=depth.dtype).clamp_min(self.min_depth)
        depth_residual = refined_depth - depth

        refined_points = points
        if self.refine_points and points is not None:
            ratio = refined_depth / depth.clamp_min(self.min_depth)
            refined_points = points * ratio.to(dtype=points.dtype)
        return refined_depth, refined_points, depth_residual

    def _prepare_voxels(self, voxel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, channels, height, width = voxel.shape
        source_bins = channels // 2
        if source_bins <= 0:
            raise ValueError("Temporal detail refiner requires polarity-separated event voxel channels.")
        pos = voxel[:, :, :source_bins].clamp_min(0.0)
        neg = voxel[:, :, source_bins : 2 * source_bins].clamp_min(0.0)
        pos = self._resample_time(pos, self.num_bins)
        neg = self._resample_time(neg, self.num_bins)
        norm = torch.log1p(voxel.new_tensor(self.count_cmax))
        pos = torch.log1p(pos.clamp_max(self.count_cmax)) / norm
        neg = torch.log1p(neg.clamp_max(self.count_cmax)) / norm

        pos_flat = pos.reshape(batch * seq_len, self.num_bins, height, width)
        neg_flat = neg.reshape(batch * seq_len, self.num_bins, height, width)
        voxels = torch.cat([pos_flat, neg_flat], dim=1)
        activity = pos_flat + neg_flat
        time = torch.linspace(-1.0, 1.0, self.num_bins, device=voxel.device, dtype=voxel.dtype).view(
            1, self.num_bins, 1, 1
        )
        summary = torch.cat(
            [
                pos_flat.mean(dim=1, keepdim=True),
                neg_flat.mean(dim=1, keepdim=True),
                activity.amax(dim=1, keepdim=True),
                ((pos_flat - neg_flat) * time).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return voxels, summary

    @staticmethod
    def _resample_time(voxel: torch.Tensor, target_bins: int) -> torch.Tensor:
        if voxel.shape[2] == target_bins:
            return voxel
        batch, seq_len, source_bins, height, width = voxel.shape
        resized = F.interpolate(
            voxel.reshape(batch * seq_len, 1, source_bins, height, width),
            size=(target_bins, height, width),
            mode="trilinear",
            align_corners=False,
        )
        resized = resized * (float(source_bins) / float(target_bins))
        return resized.reshape(batch, seq_len, target_bins, height, width)


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    """RGB VGGT coarse geometry followed by dense temporal-event refinement."""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        event_hidden_dim: int = 16,
        head_frames_chunk_size: int = 8,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        residual_scale: float = 0.03,
        refine_points: bool = True,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.head_frames_chunk_size = int(head_frames_chunk_size)
        self.event_detail_refiner = TemporalVoxelDetailRefiner(
            num_bins=event_num_bins,
            hidden_dim=event_hidden_dim,
            count_cmax=event_count_cmax,
            residual_scale=residual_scale,
            refine_points=refine_points,
            use_checkpoint=use_checkpoint,
        )
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(self, views, query_points: Optional[torch.Tensor] = None, **_kwargs):
        images = torch.stack([view["img"] for view in views], dim=0).permute(1, 0, 2, 3, 4)
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if query_points is not None and query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}
        with torch.amp.autocast(device_type="cuda", enabled=False):
            predictions["pose_enc"] = self.camera_head(aggregated_tokens_list)[-1]
            predictions["depth"], predictions["depth_conf"] = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=self.head_frames_chunk_size,
            )
            predictions["world_points"], predictions["world_points_conf"] = self.point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                frames_chunk_size=self.head_frames_chunk_size,
            )
            if query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    query_points=query_points,
                )
                predictions["track"] = track_list[-1]
                predictions["vis"] = vis
                predictions["conf"] = conf

        depth_coarse = predictions["depth"]
        points_coarse = predictions["world_points"]
        depth_final = depth_coarse
        points_final = points_coarse
        depth_residual = torch.zeros_like(depth_coarse)
        if all("event_voxel" in view for view in views):
            event_voxel = torch.stack([view["event_voxel"] for view in views], dim=1).to(device=images.device)
            if event_voxel.numel() > 0 and event_voxel.shape[2] > 0:
                depth_final, points_final, depth_residual = self.event_detail_refiner(
                    event_voxel=event_voxel,
                    images=images,
                    depth=depth_coarse,
                    points=points_coarse,
                )

        ress = []
        for frame_idx in range(images.shape[1]):
            result = {
                "pts3d_in_other_view": points_final[:, frame_idx],
                "conf": predictions["world_points_conf"][:, frame_idx],
                "depth": depth_final[:, frame_idx],
                "depth_coarse": depth_coarse[:, frame_idx],
                "depth_residual": depth_residual[:, frame_idx],
                "depth_conf": predictions["depth_conf"][:, frame_idx],
                "camera_pose": predictions["pose_enc"][:, frame_idx, :],
                **({"valid_mask": views[frame_idx]["valid_mask"]} if "valid_mask" in views[frame_idx] else {}),
            }
            if "track" in predictions:
                result.update(
                    {
                        "track": predictions["track"][:, frame_idx],
                        "vis": predictions["vis"][:, frame_idx],
                        "track_conf": predictions["conf"][:, frame_idx],
                    }
                )
            ress.append(result)
        return StreamVGGTOutput(
            ress=ress,
            views=views,
            depth_coarse=depth_coarse,
            depth_residual=depth_residual,
        )

    def inference(self, frames, query_points: Optional[torch.Tensor] = None, **kwargs):
        return self.forward(frames, query_points=query_points, **kwargs)


__all__ = ["StreamVGGT", "StreamVGGTOutput", "TemporalVoxelDetailRefiner"]
