from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers.file_utils import ModelOutput

from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from streamvggt.models.aggregator import Aggregator
from eventvggt.models.event_encoder import SimpleEventEncoder


@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[List[dict]] = None
    depth_coarse: Optional[torch.Tensor] = None
    depth_residual: Optional[torch.Tensor] = None
    points_coarse: Optional[torch.Tensor] = None
    points_residual: Optional[torch.Tensor] = None
    event_motion_density: Optional[torch.Tensor] = None
    global_memory: Optional[torch.Tensor] = None


def _as_int_list(values: Optional[Sequence[int]], fallback: Sequence[int]) -> List[int]:
    if values is None:
        return list(fallback)
    return [int(v) for v in values]


class GlobalMemoryFusion(nn.Module):
    """Perceiver-style scene memory.

    Scene tokens attend to all RGB patch tokens first. Patch tokens then attend
    back to the compact scene memory. This keeps token count fixed for the DPT
    heads while avoiding a plain mean-pooling global token.
    """

    def __init__(
        self,
        token_dim: int,
        num_global_tokens: int = 16,
        num_heads: int = 8,
        inject_layer_indices: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.num_global_tokens = int(num_global_tokens)
        self.inject_layer_indices = _as_int_list(inject_layer_indices, fallback=[-1])

        self.scene_queries = nn.Parameter(torch.zeros(1, self.num_global_tokens, token_dim))
        self.query_norm = nn.LayerNorm(token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.memory_norm = nn.LayerNorm(token_dim)
        self.patch_norm = nn.LayerNorm(token_dim)
        self.scene_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.patch_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.patch_out = nn.Linear(token_dim, token_dim)
        self.special_out = nn.Linear(token_dim, token_dim)

        nn.init.normal_(self.scene_queries, std=1e-6)
        nn.init.zeros_(self.patch_out.weight)
        nn.init.zeros_(self.patch_out.bias)
        nn.init.zeros_(self.special_out.weight)
        nn.init.zeros_(self.special_out.bias)

    def _target_layers(self, num_layers: int) -> set:
        targets = set()
        for idx in self.inject_layer_indices:
            if idx < 0:
                idx = num_layers + idx
            if 0 <= idx < num_layers:
                targets.add(idx)
        targets.add(num_layers - 1)
        return targets

    def forward(self, aggregated_tokens_list: List[torch.Tensor], patch_start_idx: int) -> dict:
        last_tokens = aggregated_tokens_list[-1]
        batch_size, seq_len, _, token_dim = last_tokens.shape

        rgb_patch_tokens = last_tokens[:, :, patch_start_idx:, :].reshape(batch_size, -1, token_dim)
        queries = self.scene_queries.to(dtype=rgb_patch_tokens.dtype).expand(batch_size, -1, -1)
        memory_update, _ = self.scene_attn(
            self.query_norm(queries),
            self.token_norm(rgb_patch_tokens),
            self.token_norm(rgb_patch_tokens),
            need_weights=False,
        )
        memory = queries + memory_update

        special_bias = self.special_out(self.memory_norm(memory).mean(dim=1)).view(batch_size, 1, 1, token_dim)
        fused_tokens = []
        target_layers = self._target_layers(len(aggregated_tokens_list))

        for layer_idx, tokens in enumerate(aggregated_tokens_list):
            if layer_idx not in target_layers:
                fused_tokens.append(tokens)
                continue

            patch_tokens = tokens[:, :, patch_start_idx:, :]
            num_patches = patch_tokens.shape[2]
            patch_flat = patch_tokens.reshape(batch_size, seq_len * num_patches, token_dim)
            patch_context, _ = self.patch_attn(
                self.patch_norm(patch_flat),
                self.memory_norm(memory),
                self.memory_norm(memory),
                need_weights=False,
            )
            patch_context = self.patch_out(patch_context).reshape(batch_size, seq_len, num_patches, token_dim)

            fused = tokens.clone()
            fused[:, :, patch_start_idx:, :] = patch_tokens + patch_context
            fused[:, :, :patch_start_idx, :] = fused[:, :, :patch_start_idx, :] + special_bias
            fused_tokens.append(fused)

        return {"tokens": fused_tokens, "memory": memory}


class EventDetailRefiner(nn.Module):
    """High-resolution event branch that predicts local geometry residuals."""

    def __init__(
        self,
        event_channels: int = 256,
        hidden_dim: int = 128,
        event_downsample: int = 4,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.event_downsample = max(1, int(event_downsample))
        self.residual_scale = float(residual_scale)

        self.event_stem = nn.Sequential(
            nn.Conv2d(event_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.motion_gate = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim + 7, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

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
        points_coarse: torch.Tensor,
    ) -> dict:
        batch_size, seq_len, event_channels, event_h, event_w = event_features.shape
        target_h, target_w = depth_coarse.shape[2:4]
        detail_size = self._detail_size(target_h, target_w)
        dtype = depth_coarse.dtype

        event_flat = event_features.reshape(batch_size * seq_len, event_channels, event_h, event_w).to(dtype=dtype)
        event_flat = F.interpolate(event_flat, size=detail_size, mode="bilinear", align_corners=False)
        event_feat = self.event_stem(event_flat)
        motion_density = torch.sigmoid(self.motion_gate(event_feat))
        event_feat = event_feat * motion_density

        depth_flat = depth_coarse.permute(0, 1, 4, 2, 3).reshape(batch_size * seq_len, 1, target_h, target_w)
        points_flat = points_coarse.permute(0, 1, 4, 2, 3).reshape(batch_size * seq_len, 3, target_h, target_w)
        image_flat = images.reshape(batch_size * seq_len, 3, images.shape[-2], images.shape[-1]).to(dtype=dtype)

        depth_detail = F.interpolate(depth_flat, size=detail_size, mode="bilinear", align_corners=False)
        points_detail = F.interpolate(points_flat, size=detail_size, mode="bilinear", align_corners=False)
        image_detail = F.interpolate(image_flat, size=detail_size, mode="bilinear", align_corners=False)

        residual = self.refine(torch.cat([event_feat, depth_detail, points_detail, image_detail], dim=1))
        if detail_size != (target_h, target_w):
            residual = F.interpolate(residual, size=(target_h, target_w), mode="bilinear", align_corners=False)
            motion_density = F.interpolate(motion_density, size=(target_h, target_w), mode="bilinear", align_corners=False)

        residual = residual * self.residual_scale
        delta_depth = residual[:, :1].reshape(batch_size, seq_len, 1, target_h, target_w).permute(0, 1, 3, 4, 2)
        delta_points = residual[:, 1:4].reshape(batch_size, seq_len, 3, target_h, target_w).permute(0, 1, 3, 4, 2)
        motion_density = motion_density.reshape(batch_size, seq_len, target_h, target_w)

        return {
            "delta_depth": delta_depth,
            "delta_points": delta_points,
            "motion_density": motion_density,
        }


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        event_hidden_dim: int = 32,
        head_frames_chunk_size: int = 8,
        branch_mode: str = "global_local",
        num_global_tokens: int = 16,
        event_downsample: int = 4,
        global_num_heads: int = 8,
        global_inject_layers: Optional[Sequence[int]] = None,
        detail_hidden_dim: int = 128,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.head_frames_chunk_size = head_frames_chunk_size
        self.branch_mode = branch_mode.lower()
        if self.branch_mode not in {"global_local", "global_only", "local_only", "rgb_coarse"}:
            raise ValueError(
                "branch_mode must be one of global_local, global_only, local_only, rgb_coarse; "
                f"got {branch_mode}"
            )
        self.use_global_branch = self.branch_mode in {"global_local", "global_only"}
        self.use_local_branch = self.branch_mode in {"global_local", "local_only"}

        self.event_encoder = SimpleEventEncoder(hidden_dim=event_hidden_dim)
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

        self.global_memory = GlobalMemoryFusion(
            token_dim=2 * embed_dim,
            num_global_tokens=num_global_tokens,
            num_heads=global_num_heads,
            inject_layer_indices=global_inject_layers,
        )
        self.event_detail_refiner = EventDetailRefiner(
            event_channels=256,
            hidden_dim=detail_hidden_dim,
            event_downsample=event_downsample,
            residual_scale=residual_scale,
        )

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

        aggregated_tokens_list, patch_start_idx = self._run_aggregator(images)

        global_memory = None
        head_tokens_list = aggregated_tokens_list
        if self.use_global_branch:
            global_fusion = self.global_memory(aggregated_tokens_list, patch_start_idx)
            head_tokens_list = global_fusion["tokens"]
            global_memory = global_fusion["memory"]

        predictions = self._run_heads(head_tokens_list, images, patch_start_idx, query_points)

        depth_coarse = predictions["depth"]
        points_coarse = predictions["world_points"]
        depth_residual = torch.zeros_like(depth_coarse)
        points_residual = torch.zeros_like(points_coarse)
        event_motion_density = None

        if self.use_local_branch and event_features is not None:
            residuals = self.event_detail_refiner(
                event_features=event_features,
                images=images,
                depth_coarse=depth_coarse,
                points_coarse=points_coarse,
            )
            depth_residual = residuals["delta_depth"]
            points_residual = residuals["delta_points"]
            event_motion_density = residuals["motion_density"]

        depth_final = (depth_coarse + depth_residual).clamp_min(1e-6)
        points_final = points_coarse + points_residual
        predictions["depth_final"] = depth_final
        predictions["world_points_final"] = points_final

        batch_size, seq_len = images.shape[:2]
        ress = []
        for frame_idx in range(seq_len):
            res = {
                "pts3d_in_other_view": points_final[:, frame_idx],
                "pts3d_coarse": points_coarse[:, frame_idx],
                "pts3d_residual": points_residual[:, frame_idx],
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
            points_coarse=points_coarse,
            points_residual=points_residual,
            event_motion_density=event_motion_density,
            global_memory=global_memory,
        )

    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):
        return self.forward(frames, query_points=query_points, past_key_values=past_key_values)

    def _run_aggregator(self, images: torch.Tensor):
        aggregator_trainable = any(param.requires_grad for param in self.aggregator.parameters())
        if aggregator_trainable:
            return self.aggregator(images)

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        return [tokens.detach() for tokens in aggregated_tokens_list], patch_start_idx

    def _run_heads(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        query_points: Optional[torch.Tensor],
    ) -> dict:
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

    def _build_model_inputs(self, views):
        if not views:
            raise ValueError("views must not be empty")

        first_view = views[0]
        images = torch.stack([view["img"] for view in views], dim=0).permute(1, 0, 2, 3, 4)

        event_features = None
        if self.use_local_branch and all(key in first_view for key in ("event_xy", "event_t", "event_p")):
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

        time_range = torch.stack(event_time_range, dim=1).to(device=device, dtype=dtype)
        return self.event_encoder(
            event_xy=batched_event_xy,
            event_t=batched_event_t,
            event_p=batched_event_p,
            event_time_range=time_range,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
