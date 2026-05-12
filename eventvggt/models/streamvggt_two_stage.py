from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers.file_utils import ModelOutput

from eventvggt.models.event_encoder import SimpleEventEncoder
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
        hidden_dim: int = 96,
        event_downsample: int = 4,
        residual_scale: float = 0.1,
        input_mode: str = "current_event",
    ) -> None:
        super().__init__()
        self.event_downsample = max(1, int(event_downsample))
        self.residual_scale = float(residual_scale)
        self.input_mode = str(input_mode)
        if self.input_mode not in self.VALID_MODES:
            raise ValueError(f"Unknown residual input mode {input_mode}. Expected one of {sorted(self.VALID_MODES)}")

        self.event_branch = self._make_branch(event_channels, hidden_dim)
        self.rgb_branch = self._make_branch(3, hidden_dim)
        self.global_rgb_branch = self._make_branch(3, hidden_dim)
        self.coarse_depth_branch = self._make_branch(1, hidden_dim)
        self.global_rgb_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
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
        images: torch.Tensor,
        depth_coarse: torch.Tensor,
    ) -> dict:
        batch_size, seq_len, event_channels, event_h, event_w = event_features.shape
        target_h, target_w = depth_coarse.shape[2:4]
        detail_size = self._detail_size(target_h, target_w)
        dtype = event_features.dtype

        event_flat = event_features.reshape(batch_size * seq_len, event_channels, event_h, event_w).to(dtype=dtype)
        event_flat = F.interpolate(event_flat, size=detail_size, mode="bilinear", align_corners=False)
        event_feat = self.event_branch(event_flat)
        motion_density = self.motion_gate(event_feat)
        fused = event_feat * motion_density

        depth_flat = depth_coarse.permute(0, 1, 4, 2, 3).reshape(batch_size * seq_len, 1, target_h, target_w)
        depth_flat = depth_flat.to(dtype=dtype)
        depth_detail = F.interpolate(depth_flat, size=detail_size, mode="bilinear", align_corners=False)
        fused = fused + self.coarse_depth_branch(depth_detail)

        image_flat = images.reshape(batch_size * seq_len, 3, images.shape[-2], images.shape[-1]).to(dtype=dtype)
        image_detail = F.interpolate(image_flat, size=detail_size, mode="bilinear", align_corners=False)
        if self.input_mode in {"event_current_rgb", "event_global_rgb_current_rgb", "single_frame_event"}:
            fused = fused + self.rgb_branch(image_detail)

        if self.input_mode in {"event_global_rgb_current_rgb", "global_rgb_current_event"}:
            global_rgb = images.mean(dim=1, keepdim=True).expand(-1, seq_len, -1, -1, -1)
            global_rgb = global_rgb.reshape(batch_size * seq_len, 3, images.shape[-2], images.shape[-1]).to(dtype=dtype)
            global_detail = F.interpolate(global_rgb, size=detail_size, mode="bilinear", align_corners=False)
            global_feat = self.global_rgb_branch(global_detail)
            fused = fused + global_feat
            fused = fused * (1.0 + self.global_rgb_gate(global_feat))

        residual = self.refine(fused) * self.residual_scale
        if detail_size != (target_h, target_w):
            residual = F.interpolate(residual, size=(target_h, target_w), mode="bilinear", align_corners=False)
            motion_density = F.interpolate(motion_density, size=(target_h, target_w), mode="bilinear", align_corners=False)

        delta_depth = residual.reshape(batch_size, seq_len, 1, target_h, target_w).permute(0, 1, 3, 4, 2)
        motion_density = motion_density.reshape(batch_size, seq_len, target_h, target_w)
        return {
            "delta_depth": delta_depth,
            "motion_density": motion_density,
        }


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        event_hidden_dim: int = 32,
        event_feature_dim: int = 64,
        event_encode_downsample: Optional[int] = None,
        head_frames_chunk_size: int = 8,
        residual_hidden_dim: int = 96,
        event_downsample: int = 4,
        residual_scale: float = 0.1,
        residual_input_mode: str = "current_event",
    ) -> None:
        super().__init__()
        self.head_frames_chunk_size = head_frames_chunk_size
        self.residual_input_mode = str(residual_input_mode)
        self.single_frame_rgb = self.residual_input_mode == "single_frame_event"
        self.event_encode_downsample = max(
            1,
            int(event_downsample if event_encode_downsample is None else event_encode_downsample),
        )

        self.event_encoder = SimpleEventEncoder(hidden_dim=event_hidden_dim, out_chans=event_feature_dim)
        self.event_residual_refiner = TwoStageEventResidualRefiner(
            event_channels=event_feature_dim,
            hidden_dim=residual_hidden_dim,
            event_downsample=event_downsample,
            residual_scale=residual_scale,
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

        if event_features is not None:
            residuals = self.event_residual_refiner(
                event_features=event_features,
                images=images,
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
        return predictions

    @staticmethod
    def _cat_frame_predictions(per_frame_outputs: List[dict]) -> dict:
        keys = set(per_frame_outputs[0].keys())
        out = {}
        for key in keys:
            if key == "images":
                out[key] = torch.cat([item[key] for item in per_frame_outputs], dim=1)
            elif key in {"pose_enc", "depth", "depth_conf", "world_points", "world_points_conf", "track", "vis", "conf"}:
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
