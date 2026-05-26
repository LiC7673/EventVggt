"""StreamVGGT variant with correctly fused, polarity-preserving temporal event bins."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers.file_utils import ModelOutput

from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from streamvggt.models.aggregator import Aggregator


@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[List[dict]] = None


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class TemporalBinTokenEncoder(nn.Module):
    """Encode signed voxel bins at patch resolution while retaining time order."""

    def __init__(
        self,
        *,
        patch_size: int,
        token_dim: int,
        hidden_dim: int = 32,
        num_bins: int = 10,
        count_cmax: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_bins = max(1, int(num_bins))
        self.count_cmax = max(1.0, float(count_cmax))
        self.spatial_patch = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_mix = nn.Sequential(
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                groups=hidden_dim,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.temporal_score = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_dim, token_dim, kernel_size=1)

        # Start from the pretrained RGB path and let events enter gradually.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, event_voxel: torch.Tensor) -> torch.Tensor:
        if event_voxel.ndim != 5:
            raise ValueError(f"event_voxel must be [B,S,2*T,H,W], got {tuple(event_voxel.shape)}")

        batch, seq_len, channels, height, width = event_voxel.shape
        source_bins = channels // 2
        if source_bins <= 0:
            raise ValueError("TemporalBinTokenEncoder requires polarity-separated event voxel channels.")

        pos = event_voxel[:, :, :source_bins].clamp_min(0.0)
        neg = event_voxel[:, :, source_bins : 2 * source_bins].clamp_min(0.0)
        pos = self._resample_time(pos, self.num_bins)
        neg = self._resample_time(neg, self.num_bins)

        norm = torch.log1p(event_voxel.new_tensor(self.count_cmax))
        pos = torch.log1p(pos.clamp_max(self.count_cmax)) / norm
        neg = torch.log1p(neg.clamp_max(self.count_cmax)) / norm
        signed_bins = torch.stack([pos, neg], dim=3)  # [B,S,T,2,H,W]

        x = signed_bins.reshape(batch * seq_len * self.num_bins, 2, height, width)
        x = self.spatial_patch(x)
        patch_height, patch_width = x.shape[-2:]
        x = x.view(batch * seq_len, self.num_bins, -1, patch_height, patch_width).permute(0, 2, 1, 3, 4)

        time = torch.linspace(-1.0, 1.0, self.num_bins, device=x.device, dtype=x.dtype).unsqueeze(-1)
        time_embedding = self.time_embed(time).transpose(0, 1).view(1, x.shape[1], self.num_bins, 1, 1)
        x = x + time_embedding
        x = x + self.temporal_mix(x)

        weights = torch.softmax(self.temporal_score(x), dim=2)
        x = (x * weights).sum(dim=2)
        tokens = self.out_proj(x).flatten(2).transpose(1, 2)
        return tokens.view(batch, seq_len, tokens.shape[1], tokens.shape[2])

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
    """VGGT with temporal-bin event tokens fused into the patch-token stream."""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        event_hidden_dim: int = 32,
        head_frames_chunk_size: int = 8,
        event_num_bins: int = 10,
        event_count_cmax: float = 3.0,
        event_fusion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_frames_chunk_size = int(head_frames_chunk_size)
        self.event_fusion_scale = float(event_fusion_scale)
        self.event_encoder = TemporalBinTokenEncoder(
            patch_size=patch_size,
            token_dim=2 * embed_dim,
            hidden_dim=event_hidden_dim,
            num_bins=event_num_bins,
            count_cmax=event_count_cmax,
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
        event_tokens = self._encode_event_tokens(views, images)
        if event_tokens is not None:
            aggregated_tokens_list = self._fuse_event_tokens(
                aggregated_tokens_list,
                event_tokens,
                patch_start_idx,
            )

        predictions = {}
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(aggregated_tokens_list)[-1]
            if self.depth_head is not None:
                predictions["depth"], predictions["depth_conf"] = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.head_frames_chunk_size,
                )
            if self.point_head is not None:
                predictions["world_points"], predictions["world_points_conf"] = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.head_frames_chunk_size,
                )
            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    query_points=query_points,
                )
                predictions["track"] = track_list[-1]
                predictions["vis"] = vis
                predictions["conf"] = conf

        ress = []
        for frame_idx in range(images.shape[1]):
            result = {
                "pts3d_in_other_view": predictions["world_points"][:, frame_idx],
                "conf": predictions["world_points_conf"][:, frame_idx],
                "depth": predictions["depth"][:, frame_idx],
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
        return StreamVGGTOutput(ress=ress, views=views)

    def inference(self, frames, query_points: Optional[torch.Tensor] = None, **kwargs):
        return self.forward(frames, query_points=query_points, **kwargs)

    def _encode_event_tokens(self, views, images: torch.Tensor) -> Optional[torch.Tensor]:
        if not all("event_voxel" in view for view in views):
            return None
        voxel = torch.stack([view["event_voxel"] for view in views], dim=1)
        if voxel.numel() == 0 or voxel.shape[2] == 0:
            return None
        voxel = voxel.to(device=images.device, dtype=images.dtype)
        return self.event_encoder(voxel)

    def _fuse_event_tokens(self, tokens_list, event_tokens: torch.Tensor, patch_start_idx: int):
        fused_tokens = []
        injected = False
        for tokens in tokens_list:
            if (
                tokens.ndim == 4
                and tokens.shape[2] - patch_start_idx == event_tokens.shape[2]
                and tokens.shape[-1] == event_tokens.shape[-1]
            ):
                fused = tokens.clone()
                fused[:, :, patch_start_idx:, :] = (
                    fused[:, :, patch_start_idx:, :] + self.event_fusion_scale * event_tokens
                )
                fused_tokens.append(fused)
                injected = True
            else:
                fused_tokens.append(tokens)
        if not injected:
            raise RuntimeError(
                "Temporal event tokens were not fused: "
                f"event={tuple(event_tokens.shape)}, token_layers={[tuple(t.shape) for t in tokens_list]}, "
                f"patch_start_idx={patch_start_idx}"
            )
        return fused_tokens


__all__ = ["StreamVGGT", "StreamVGGTOutput", "TemporalBinTokenEncoder"]
