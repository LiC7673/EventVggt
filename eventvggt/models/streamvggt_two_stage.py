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
    depth_coarse: Optional[torch.Tensor] = None
    depth_residual: Optional[torch.Tensor] = None
    event_motion_density: Optional[torch.Tensor] = None
    residual_input_mode: Optional[str] = None


class BinnedEventEncoder(nn.Module):
    """Encode each inter-frame event packet as polarity-separated temporal bins."""

    def __init__(self, num_bins: int = 8, count_cmax: float = 3.0) -> None:
        super().__init__()
        self.num_bins = max(1, int(num_bins))
        self.count_cmax = max(1.0, float(count_cmax))
        self.out_channels = 2 * self.num_bins

    def forward(
        self,
        event_xy: List[List[torch.Tensor]],
        event_t: List[List[torch.Tensor]],
        event_p: List[List[torch.Tensor]],
        event_time_range: Optional[torch.Tensor],
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = len(event_xy)
        seq_len = len(event_xy[0]) if batch_size > 0 else 0
        encoded_frames = []

        for batch_idx in range(batch_size):
            frame_maps = []
            for frame_idx in range(seq_len):
                time_range = (
                    event_time_range[batch_idx, frame_idx]
                    if event_time_range is not None
                    else None
                )
                frame_maps.append(
                    self._encode_single_frame(
                        event_xy=event_xy[batch_idx][frame_idx],
                        event_t=event_t[batch_idx][frame_idx],
                        event_p=event_p[batch_idx][frame_idx],
                        event_time_range=time_range,
                        height=height,
                        width=width,
                        device=device,
                        dtype=dtype,
                    )
                )
            encoded_frames.append(torch.stack(frame_maps, dim=0))

        return torch.stack(encoded_frames, dim=0)

    def _encode_single_frame(
        self,
        event_xy: torch.Tensor,
        event_t: torch.Tensor,
        event_p: torch.Tensor,
        event_time_range: Optional[torch.Tensor],
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rep = torch.zeros(self.out_channels, height, width, device=device, dtype=dtype)
        if event_xy.numel() == 0 or event_t.numel() == 0 or event_p.numel() == 0:
            return rep

        event_xy = event_xy.to(device=device)
        event_t = event_t.to(device=device, dtype=dtype)
        event_p = event_p.to(device=device)

        x = event_xy[:, 0].long().clamp_(0, width - 1)
        y = event_xy[:, 1].long().clamp_(0, height - 1)
        pixel_idx = y * width + x

        if event_time_range is not None:
            event_time_range = event_time_range.to(device=device, dtype=dtype)
            start_t = event_time_range[0]
            end_t = event_time_range[1]
        else:
            start_t = event_t.min()
            end_t = event_t.max()

        duration = (end_t - start_t).clamp_min(1.0)
        norm_t = ((event_t - start_t) / duration).clamp_(0.0, 1.0 - 1e-6)
        bin_idx = torch.floor(norm_t * self.num_bins).long().clamp_(0, self.num_bins - 1)
        polarity_offset = torch.where(
            event_p.to(device=device).float() > 0,
            torch.zeros_like(bin_idx),
            torch.full_like(bin_idx, self.num_bins),
        )
        channel_idx = polarity_offset + bin_idx
        flat_idx = channel_idx * (height * width) + pixel_idx

        flat = rep.reshape(-1)
        event_weight = event_p.to(device=device, dtype=dtype).abs()
        flat.index_add_(0, flat_idx, event_weight)
        rep = torch.clamp(rep, max=self.count_cmax)
        rep = torch.log1p(rep) / torch.log1p(rep.new_tensor(self.count_cmax))
        return rep


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in (8, 4, 2):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(channels, channels, dilation=dilation),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.GELU()
        self.res_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.res_scale * self.net(x))


class DeepEventCNNBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        blocks = [
            ConvGNAct(in_channels, hidden_dim),
            ResidualConvBlock(hidden_dim, dilation=1),
            ResidualConvBlock(hidden_dim, dilation=1),
            ResidualConvBlock(hidden_dim, dilation=2),
            ResidualConvBlock(hidden_dim, dilation=2),
            ResidualConvBlock(hidden_dim, dilation=4),
            ResidualConvBlock(hidden_dim, dilation=1),
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EventUNetBackbone(nn.Module):
    """U-Net style feature extractor for dense event tensors."""

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        c1 = hidden_dim
        c2 = hidden_dim * 2
        c3 = hidden_dim * 3

        self.enc0 = nn.Sequential(
            ConvGNAct(in_channels, c1),
            ResidualConvBlock(c1),
            ResidualConvBlock(c1, dilation=2),
        )
        self.enc1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvGNAct(c1, c2),
            ResidualConvBlock(c2),
            ResidualConvBlock(c2, dilation=2),
        )
        self.enc2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            ConvGNAct(c2, c3),
            ResidualConvBlock(c3),
            ResidualConvBlock(c3, dilation=2),
        )
        self.context = nn.Sequential(
            ResidualConvBlock(c3, dilation=2),
            ResidualConvBlock(c3, dilation=4),
            ResidualConvBlock(c3, dilation=1),
        )
        self.dec1 = nn.Sequential(
            ConvGNAct(c3 + c2, c2),
            ResidualConvBlock(c2),
            ResidualConvBlock(c2),
        )
        self.dec0 = nn.Sequential(
            ConvGNAct(c2 + c1, c1),
            ResidualConvBlock(c1),
            ResidualConvBlock(c1),
        )
        self.out = nn.Conv2d(c1, hidden_dim, kernel_size=1)

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.enc0(x)
        skip1 = self.enc1(skip0)
        bottleneck = self.context(self.enc2(skip1))

        up1 = self._upsample_to(bottleneck, skip1)
        up1 = self.dec1(torch.cat([up1, skip1], dim=1))
        up0 = self._upsample_to(up1, skip0)
        up0 = self.dec0(torch.cat([up0, skip0], dim=1))
        return self.out(up0)


class TwoStageEventResidualRefiner(nn.Module):
    """Second-stage event/RGB branch that predicts a residual over RGB StreamVGGT depth."""

    VALID_MODES = {
        "current_event",
        "event_current_rgb",
        "event_global_rgb_current_rgb",
        "global_rgb_current_event",
        "single_frame_event",
    }

    def __init__(
        self,
        event_channels: int = 256,
        rgb_token_dim: int = 2048,
        hidden_dim: int = 96,
        event_downsample: int = 1,
        residual_scale: float = 0.1,
        residual_activation: str = "tanh",
        event_backbone: str = "unet",
        input_mode: str = "current_event",
    ) -> None:
        super().__init__()
        self.event_downsample = max(1, int(event_downsample))
        self.residual_scale = float(residual_scale)
        self.residual_activation = str(residual_activation).lower()
        self.event_backbone = str(event_backbone).lower()
        self.input_mode = str(input_mode)
        if self.input_mode not in self.VALID_MODES:
            raise ValueError(f"Unknown residual input mode {input_mode}. Expected one of {sorted(self.VALID_MODES)}")
        if self.residual_activation not in {"tanh", "linear"}:
            raise ValueError("residual_activation must be 'tanh' or 'linear'")
        if self.event_backbone not in {"shallow", "cnn", "unet"}:
            raise ValueError("event_backbone must be 'shallow', 'cnn', or 'unet'")

        if self.event_backbone == "unet":
            self.event_branch = EventUNetBackbone(event_channels, hidden_dim)
        elif self.event_backbone == "cnn":
            self.event_branch = DeepEventCNNBackbone(event_channels, hidden_dim)
        else:
            self.event_branch = self._make_branch(event_channels, hidden_dim)
        self.coarse_depth_branch = self._make_branch(1, hidden_dim)
        self.rgb_token_modulator = nn.Sequential(
            nn.LayerNorm(rgb_token_dim),
            nn.Linear(rgb_token_dim, 2 * hidden_dim),
        )
        self.motion_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    @staticmethod
    def _make_branch(in_channels: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def _detail_size(self, height: int, width: int) -> tuple:
        return (
            max(1, height // self.event_downsample),
            max(1, width // self.event_downsample),
        )

    def forward(
        self,
        event_features: torch.Tensor,
        global_rgb_token: torch.Tensor,
        frame_rgb_token: Optional[torch.Tensor],
        depth_coarse: torch.Tensor,
    ) -> dict:
        batch_size, seq_len, event_channels, event_h, event_w = event_features.shape
        target_h, target_w = depth_coarse.shape[2:4]
        detail_size = self._detail_size(target_h, target_w)
        dtype = event_features.dtype

        event_flat = event_features.reshape(batch_size * seq_len, event_channels, event_h, event_w).to(dtype=dtype)
        if (event_h, event_w) != detail_size:
            event_flat = F.interpolate(event_flat, size=detail_size, mode="bilinear", align_corners=False)
        event_feat = self.event_branch(event_flat)
        motion_density = self.motion_gate(event_feat)
        fused = event_feat * motion_density

        depth_flat = depth_coarse.permute(0, 1, 4, 2, 3).reshape(batch_size * seq_len, 1, target_h, target_w)
        depth_flat = depth_flat.to(dtype=dtype)
        depth_detail = depth_flat
        if (target_h, target_w) != detail_size:
            depth_detail = F.interpolate(depth_flat, size=detail_size, mode="bilinear", align_corners=False)
        fused = fused + self.coarse_depth_branch(depth_detail)

        condition_token = self._select_condition_token(global_rgb_token, frame_rgb_token)
        if condition_token is not None:
            if condition_token.ndim == 2:
                condition_token = condition_token[:, None, :].expand(-1, seq_len, -1)
            token_flat = condition_token.reshape(batch_size * seq_len, -1).to(dtype=dtype)
            scale, shift = self.rgb_token_modulator(token_flat).chunk(2, dim=-1)
            scale = 0.1 * torch.tanh(scale).view(batch_size * seq_len, -1, 1, 1)
            shift = 0.1 * torch.tanh(shift).view(batch_size * seq_len, -1, 1, 1)
            fused = fused * (1.0 + scale) + shift

        residual = self.refine(fused)
        if self.residual_activation == "tanh":
            residual = torch.tanh(residual)
        residual = residual * self.residual_scale
        if detail_size != (target_h, target_w):
            residual = F.interpolate(residual, size=(target_h, target_w), mode="bilinear", align_corners=False)
            motion_density = F.interpolate(motion_density, size=(target_h, target_w), mode="bilinear", align_corners=False)

        delta_depth = residual.reshape(batch_size, seq_len, 1, target_h, target_w).permute(0, 1, 3, 4, 2)
        motion_density = motion_density.reshape(batch_size, seq_len, target_h, target_w)
        return {
            "delta_depth": delta_depth,
            "motion_density": motion_density,
        }

    def _select_condition_token(
        self,
        global_rgb_token: Optional[torch.Tensor],
        frame_rgb_token: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.input_mode == "current_event":
            return None
        if self.input_mode in {"event_current_rgb", "single_frame_event"}:
            return frame_rgb_token
        if self.input_mode == "global_rgb_current_event":
            return global_rgb_token
        if self.input_mode == "event_global_rgb_current_rgb":
            if global_rgb_token is None:
                return frame_rgb_token
            if frame_rgb_token is None:
                return global_rgb_token
            return global_rgb_token + frame_rgb_token
        return global_rgb_token


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        event_hidden_dim: int = 32,
        event_num_bins: int = 8,
        event_count_cmax: float = 3.0,
        event_encode_downsample: Optional[int] = None,
        head_frames_chunk_size: int = 8,
        residual_hidden_dim: int = 96,
        event_downsample: int = 1,
        residual_scale: float = 0.1,
        residual_activation: str = "tanh",
        event_backbone: str = "unet",
        residual_input_mode: str = "current_event",
        disable_second_stage: bool = False,
    ) -> None:
        super().__init__()
        self.head_frames_chunk_size = head_frames_chunk_size
        self.residual_input_mode = str(residual_input_mode)
        self.disable_second_stage = bool(disable_second_stage)
        self.single_frame_rgb = self.residual_input_mode == "single_frame_event"
        self.event_encode_downsample = max(
            1,
            int(event_downsample if event_encode_downsample is None else event_encode_downsample),
        )

        self.event_encoder = BinnedEventEncoder(num_bins=event_num_bins, count_cmax=event_count_cmax)
        self.event_residual_refiner = TwoStageEventResidualRefiner(
            event_channels=self.event_encoder.out_channels,
            rgb_token_dim=2 * embed_dim,
            hidden_dim=residual_hidden_dim,
            event_downsample=event_downsample,
            residual_scale=residual_scale,
            residual_activation=residual_activation,
            event_backbone=event_backbone,
            input_mode=self.residual_input_mode,
        )
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache: bool = False,
        past_frame_idx: int = 0,
    ):
        images, event_features = self._build_model_inputs(views)

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        if history_info is None:
            history_info = {"token": None}

        predictions = self._run_rgb_streamvggt(images, query_points)
        depth_coarse = predictions["depth"]
        depth_residual = torch.zeros_like(depth_coarse)
        event_motion_density = None

        if event_features is not None and not self.disable_second_stage:
            residuals = self.event_residual_refiner(
                event_features=event_features,
                global_rgb_token=predictions.get("global_rgb_token"),
                frame_rgb_token=predictions.get("frame_rgb_token"),
                depth_coarse=depth_coarse,
            )
            depth_residual = residuals["delta_depth"]
            event_motion_density = residuals["motion_density"]

        depth_final = (depth_coarse + depth_residual).clamp_min(1e-6)

        batch_size, seq_len = images.shape[:2]
        ress = []
        for frame_idx in range(seq_len):
            res = {
                "pts3d_in_other_view": predictions["world_points"][:, frame_idx],
                "conf": predictions["world_points_conf"][:, frame_idx],
                "depth": depth_final[:, frame_idx],
                "depth_coarse": depth_coarse[:, frame_idx],
                "depth_residual": depth_residual[:, frame_idx],
                "depth_conf": predictions["depth_conf"][:, frame_idx],
                "camera_pose": predictions["pose_enc"][:, frame_idx, :],
                **({"valid_mask": views[frame_idx]["valid_mask"]} if "valid_mask" in views[frame_idx] else {}),
                **(
                    {
                        "track": predictions["track"][:, frame_idx],
                        "vis": predictions["vis"][:, frame_idx],
                        "track_conf": predictions["conf"][:, frame_idx],
                    }
                    if "track" in predictions
                    else {}
                ),
            }
            ress.append(res)

        return StreamVGGTOutput(
            ress=ress,
            views=views,
            depth_coarse=depth_coarse,
            depth_residual=depth_residual,
            event_motion_density=event_motion_density,
            residual_input_mode=self.residual_input_mode,
        )

    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):
        return self.forward(frames, query_points=query_points, past_key_values=past_key_values)

    def _run_rgb_streamvggt(self, images: torch.Tensor, query_points: Optional[torch.Tensor]) -> dict:
        if self.single_frame_rgb:
            per_frame_outputs = []
            for frame_idx in range(images.shape[1]):
                frame_query = query_points[:, frame_idx] if query_points is not None and query_points.ndim == 4 else query_points
                per_frame_outputs.append(self._run_rgb_streamvggt_multi(images[:, frame_idx : frame_idx + 1], frame_query))
            return self._cat_frame_predictions(per_frame_outputs)
        return self._run_rgb_streamvggt_multi(images, query_points)

    def _run_rgb_streamvggt_multi(self, images: torch.Tensor, query_points: Optional[torch.Tensor]) -> dict:
        aggregated_tokens_list, patch_start_idx = self._run_aggregator(images)
        predictions = {}

        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.head_frames_chunk_size,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=self.head_frames_chunk_size,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

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
            predictions["images"] = images
            patch_tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]
            frame_tokens = patch_tokens.mean(dim=2)
            global_token = frame_tokens.mean(dim=1, keepdim=True).expand(-1, images.shape[1], -1)
            predictions["frame_rgb_token"] = frame_tokens
            predictions["global_rgb_token"] = global_token
        return predictions

    @staticmethod
    def _cat_frame_predictions(per_frame_outputs: List[dict]) -> dict:
        keys = set(per_frame_outputs[0].keys())
        out = {}
        for key in keys:
            if key == "images":
                out[key] = torch.cat([item[key] for item in per_frame_outputs], dim=1)
            elif key in {
                "pose_enc",
                "depth",
                "depth_conf",
                "world_points",
                "world_points_conf",
                "track",
                "vis",
                "conf",
                "frame_rgb_token",
                "global_rgb_token",
            }:
                out[key] = torch.cat([item[key] for item in per_frame_outputs], dim=1)
        return out

    def _run_aggregator(self, images: torch.Tensor):
        aggregator_trainable = any(param.requires_grad for param in self.aggregator.parameters())
        if aggregator_trainable:
            return self.aggregator(images)

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        return [tokens.detach() for tokens in aggregated_tokens_list], patch_start_idx

    def _build_model_inputs(self, views):
        if not views:
            raise ValueError("views must not be empty")

        first_view = views[0]
        images = torch.stack([view["img"] for view in views], dim=0).permute(1, 0, 2, 3, 4)

        event_features = None
        if all(key in first_view for key in ("event_xy", "event_t", "event_p")):
            event_features = self._encode_events_as_images(views)

        return images, event_features

    def _encode_events_as_images(self, views):
        seq_len = len(views)
        event_xy = []
        event_t = []
        event_p = []
        event_time_range = []

        for view in views:
            event_xy.append(view["event_xy"])
            event_t.append(view["event_t"])
            event_p.append(view["event_p"])
            event_time_range.append(view["event_time_range"])

        batch_size = len(event_xy[0])
        batched_event_xy = [[event_xy[s][b] for s in range(seq_len)] for b in range(batch_size)]
        batched_event_t = [[event_t[s][b] for s in range(seq_len)] for b in range(batch_size)]
        batched_event_p = [[event_p[s][b] for s in range(seq_len)] for b in range(batch_size)]

        reference_img = views[0]["img"]
        if reference_img.ndim == 4:
            _, _, height, width = reference_img.shape
            device = reference_img.device
            dtype = reference_img.dtype
        else:
            _, height, width = reference_img.shape
            device = reference_img.device
            dtype = reference_img.dtype

        encode_h = max(1, height // self.event_encode_downsample)
        encode_w = max(1, width // self.event_encode_downsample)
        if encode_h != height or encode_w != width:
            scale_x = float(encode_w) / max(float(width), 1.0)
            scale_y = float(encode_h) / max(float(height), 1.0)

            def scale_event_xy(xy: torch.Tensor) -> torch.Tensor:
                if xy.numel() == 0:
                    return xy
                scaled = xy.clone()
                scaled_x = torch.floor(scaled[:, 0].float() * scale_x).clamp_(0, encode_w - 1)
                scaled_y = torch.floor(scaled[:, 1].float() * scale_y).clamp_(0, encode_h - 1)
                scaled[:, 0] = scaled_x.to(dtype=scaled.dtype)
                scaled[:, 1] = scaled_y.to(dtype=scaled.dtype)
                return scaled

            batched_event_xy = [
                [scale_event_xy(batched_event_xy[b][s]) for s in range(seq_len)]
                for b in range(batch_size)
            ]

        time_range = torch.stack(event_time_range, dim=1).to(device=device, dtype=dtype)
        return self.event_encoder(
            event_xy=batched_event_xy,
            event_t=batched_event_t,
            event_p=batched_event_p,
            event_time_range=time_range,
            height=encode_h,
            width=encode_w,
            device=device,
            dtype=dtype,
        )
