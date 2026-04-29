import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from eventvggt.models.event_encoder import SimpleEventEncoder
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None
    global_token_rgb: Optional[torch.Tensor] = None
    global_token_event: Optional[torch.Tensor] = None
    global_token_refined: Optional[torch.Tensor] = None
    global_token_target: Optional[torch.Tensor] = None


class GlobalTokenFusion(nn.Module):
    """Build a sequence-level global token from RGB views and refine it with events.

    The DPT heads require the patch-token count to remain unchanged, so the
    global token is injected with lightweight FiLM modulation instead of being
    appended to the token sequence.
    """

    def __init__(self, token_dim: int, event_channels: int = 256):
        super().__init__()
        self.global_token = nn.Parameter(torch.zeros(1, token_dim))
        self.rgb_norm = nn.LayerNorm(token_dim)
        self.rgb_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.event_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.event_gate = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.Sigmoid(),
        )
        self.inject = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * token_dim),
        )
        self.special_token_bias = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
        )

        nn.init.normal_(self.global_token, std=1e-6)

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
        event_tokens: Optional[torch.Tensor] = None,
    ):
        last_tokens = aggregated_tokens_list[-1]
        rgb_summary = last_tokens[:, :, patch_start_idx:, :].mean(dim=(1, 2))
        global_token_rgb = self.rgb_mlp(self.rgb_norm(rgb_summary)) + self.global_token.to(rgb_summary.dtype)

        if event_tokens is None:
            global_token_event = torch.zeros_like(global_token_rgb)
            refined_global_token = global_token_rgb
        else:
            event_summary = event_tokens.mean(dim=(1, 2))
            global_token_event = self.event_mlp(event_summary)
            gate = self.event_gate(torch.cat([global_token_rgb, global_token_event], dim=-1))
            refined_global_token = global_token_rgb + gate * global_token_event

        scale, shift = self.inject(refined_global_token).chunk(2, dim=-1)
        scale = 0.1 * torch.tanh(scale).unsqueeze(1).unsqueeze(1)
        shift = 0.1 * torch.tanh(shift).unsqueeze(1).unsqueeze(1)
        special_bias = 0.1 * torch.tanh(self.special_token_bias(refined_global_token)).unsqueeze(1).unsqueeze(1)

        fused_tokens = []
        for tokens in aggregated_tokens_list:
            fused = tokens * (1.0 + scale) + shift
            fused = fused.clone()
            fused[:, :, :patch_start_idx, :] = fused[:, :, :patch_start_idx, :] + special_bias
            fused_tokens.append(fused)

        return {
            "tokens": fused_tokens,
            "global_token_rgb": global_token_rgb,
            "global_token_event": global_token_event,
            "global_token_refined": refined_global_token,
            "global_token_target": rgb_summary.detach(),
        }

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, event_hidden_dim=32, head_frames_chunk_size=8):
        super().__init__()

        self.head_frames_chunk_size = head_frames_chunk_size
        self.event_encoder = SimpleEventEncoder(hidden_dim=event_hidden_dim)
        self.event_patch_embed = nn.Conv2d(
            in_channels=256, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.event_token_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.global_fusion = GlobalTokenFusion(token_dim=2 * embed_dim, event_channels=256)
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images, event_features = self._build_model_inputs(views)

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        # 2. RGB 走原本的 Aggregator，完美保留预训练权重
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        # 3. 事件流走专属的 PatchEmbed，并与 RGB Tokens 融合
        event_tokens_for_global = None
        if event_features is not None:
            # event_features 形状为 [B, S, C, H, W]
            B, S, C, H, W = event_features.shape
            
            # 折叠 B 和 S 维度送入 Conv2d
            event_flat = event_features.view(B * S, C, H, W)
            event_tokens = self.event_patch_embed(event_flat) # -> [B*S, embed_dim, H/P, W/P]
            
            # 展平为 Token 序列 -> [B*S, N, embed_dim]
            event_tokens = self.event_token_proj(event_tokens.flatten(2).transpose(1, 2))
            
            # ViT 的输出通常带一个 [CLS] Token，所以 N 比 event_tokens 多 1
            # 我们给 event_tokens 头部补一个全零的 CLS Token 以对齐维度
            event_tokens = event_tokens.view(B, S, event_tokens.shape[1], event_tokens.shape[-1])
            event_tokens_for_global = event_tokens
            # 将事件特征加到多层/多尺度的 aggregated_tokens 中
            if isinstance(aggregated_tokens_list, list):
                fused_tokens_list = []
                for tokens in aggregated_tokens_list:
                    # 检查维度是否对齐 (有些层可能不带 CLS 或者分辨率不同)
                    if (
                        tokens.shape[2] - patch_start_idx == event_tokens.shape[2]
                        and tokens.shape[-1] == event_tokens.shape[-1]
                    ):
                        fused = tokens.clone()
                        fused[:, :, patch_start_idx:, :] = fused[:, :, patch_start_idx:, :] + event_tokens
                        fused_tokens_list.append(fused)
                    else:
                        fused_tokens_list.append(tokens)
                aggregated_tokens_list = fused_tokens_list
            else:
                if (
                    aggregated_tokens_list.shape[2] - patch_start_idx == event_tokens.shape[2]
                    and aggregated_tokens_list.shape[-1] == event_tokens.shape[-1]
                ):
                    fused = aggregated_tokens_list.clone()
                    fused[:, :, patch_start_idx:, :] = fused[:, :, patch_start_idx:, :] + event_tokens
                    aggregated_tokens_list = fused

        # 下面的预测逻辑 (camera_head, depth_head等) 保持不变！
        global_fusion = self.global_fusion(aggregated_tokens_list, patch_start_idx, event_tokens_for_global)
        aggregated_tokens_list = global_fusion["tokens"]

        predictions = {}

        with torch.amp.autocast(device_type='cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

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
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(
                ress=ress,
                views=views,
                global_token_rgb=global_fusion["global_token_rgb"],
                global_token_event=global_fusion["global_token_event"],
                global_token_refined=global_fusion["global_token_refined"],
                global_token_target=global_fusion["global_token_target"],
            )  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            # 1. 同样获取两者
            images, event_features = self._build_model_inputs([frame])
            
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            # 2. 融合事件特征
            if event_features is not None:
                B, S, C, H, W = event_features.shape
                event_flat = event_features.view(B * S, C, H, W)
                event_tokens = self.event_patch_embed(event_flat).flatten(2).transpose(1, 2)
                cls_pad = torch.zeros(B * S, 1, event_tokens.shape[-1], device=event_tokens.device, dtype=event_tokens.dtype)
                event_tokens_padded = torch.cat([cls_pad, event_tokens], dim=1)

                if aggregated_tokens.shape[1] == event_tokens_padded.shape[1]:
                    aggregated_tokens = aggregated_tokens + event_tokens_padded
                elif aggregated_tokens.shape[1] == event_tokens.shape[1]:
                    aggregated_tokens = aggregated_tokens + event_tokens
            
            with torch.amp.autocast(device_type='cuda', enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        frames_chunk_size=self.head_frames_chunk_size,
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        frames_chunk_size=self.head_frames_chunk_size,
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            processed_frames.append(frame)
        
        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output

    def _build_model_inputs(self, views):
        if not views:
            raise ValueError("views must not be empty")

        first_view = views[0]
        
        # 1. 永远提取 RGB 图像 (保持 3 通道)
        images = torch.stack([view["img"] for view in views], dim=0).permute(1, 0, 2, 3, 4)
        
        # 2. 如果存在事件数据，同时也提取事件特征 (你设定的是 256 通道)
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
