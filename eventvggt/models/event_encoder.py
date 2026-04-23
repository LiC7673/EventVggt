from typing import List, Optional

import torch
import torch.nn as nn
import os
from datetime import datetime
import matplotlib.pyplot as plt
def save_energy_map_viz(tensor: torch.Tensor, save_dir: str = "event_viz_debug"):
    """
    可视化事件流的能量图
    tensor shape expected: [B, S, C, H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 为了不影响反向传播，必须 detach() 并转到 CPU
    # 提取 batch 中第一个样本 (b=0) 的特征
    tensor_b0 = tensor[0].detach().cpu().float()  # shape: [S, C, H, W]
    
    seq_len = tensor_b0.shape[0]
    
    for s in range(seq_len):
        frame_feat = tensor_b0[s]  # shape: [C, H, W]
        
        # 计算“能量”：沿着通道维度(C)求均方根或 L2 范数
        # 这能代表该空间位置上所有通道的激活强度总和
        energy_map = torch.norm(frame_feat, p=2, dim=0)  # shape: [H, W]
        
        # 归一化到 0~1，方便用 matplotlib 上色
        e_min, e_max = energy_map.min(), energy_map.max()
        if e_max > e_min:
            energy_map = (energy_map - e_min) / (e_max - e_min)
            
        # 生成时间戳文件名 (精确到毫秒)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
   
        filename = f"event_energy_seq{s}_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        
        # 使用 magma 或 jet 热力图保存 (能量越高的地方越亮/红)
        plt.imsave(filepath, energy_map.numpy(), cmap='magma')

class SimpleEventEncoder(nn.Module):
    """Encode per-frame events into a 3-channel pseudo-image for StreamVGGT."""

    # def __init__(self, in_chans: int = 5, hidden_dim: int = 32, out_chans: int = 3):
    #     super().__init__()
    #     self.proj = nn.Sequential(
    #         nn.Conv2d(in_chans, hidden_dim, kernel_size=3, padding=1),
    #         nn.GroupNorm(4, hidden_dim),
    #         nn.GELU(),
    #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
    #         nn.GroupNorm(4, hidden_dim),
    #         nn.GELU(),
    #         nn.Conv2d(hidden_dim, out_chans, kernel_size=1),
    #         nn.Sigmoid(),
    #     )
    def __init__(self, in_chans: int = 5, hidden_dim: int = 64, out_chans: int = 256): # out_chans 设为和你 Backbone 特征匹配的维度
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim), # 注意调整 Group 数
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            # 最后一层不要 Sigmoid，直接输出高维特征
            nn.Conv2d(hidden_dim, out_chans, kernel_size=3, padding=1), 
            # nn.Sigmoid(), # 致命的
        )
    
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
        """Build [B, S, 3, H, W] pseudo-images from event lists."""
        batch_size = len(event_xy)
        seq_len = len(event_xy[0]) if batch_size > 0 else 0
        encoded_frames = []

        for batch_idx in range(batch_size):
            frame_maps = []
            for frame_idx in range(seq_len):
                frame_maps.append(
                    self._encode_single_frame(
                        event_xy=batch_idx_select(event_xy, batch_idx, frame_idx),
                        event_t=batch_idx_select(event_t, batch_idx, frame_idx),
                        event_p=batch_idx_select(event_p, batch_idx, frame_idx),
                        event_time_range=event_time_range[batch_idx, frame_idx]
                        if event_time_range is not None
                        else None,
                        height=height,
                        width=width,
                        device=device,
                        dtype=dtype,
                    )
                )
            encoded_frames.append(torch.stack(frame_maps, dim=0))

        event_tensor = torch.stack(encoded_frames, dim=0)  # [B, S, 5, H, W]
        bs, seq, channels, h, w = event_tensor.shape
        pseudo_images = self.proj(event_tensor.reshape(bs * seq, channels, h, w))
        out_features = pseudo_images.reshape(bs, seq, -1, h, w)
        
        # ==========================================
        # [插入 Debug 可视化代码]
        # 我们在这里拦截输出的特征图进行可视化
        # 建议加一个条件判断，避免训练时每一步都疯狂写硬盘
        # ==========================================
        # if getattr(self, "debug_viz", False): # 可以通过给实例设置 encoder.debug_viz = True 来随时开启
        # save_energy_map_viz(out_features, save_dir="debug_event_energy")
        return pseudo_images.reshape(bs, seq, -1, h, w)

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
        rep = torch.zeros(5, height, width, device=device, dtype=dtype)

        if event_xy.numel() == 0 or event_t.numel() == 0 or event_p.numel() == 0:
            return rep

        event_xy = event_xy.to(device=device)
        event_t = event_t.to(device=device, dtype=dtype)
        event_p = event_p.to(device=device, dtype=dtype)

        x = event_xy[:, 0].long().clamp_(0, width - 1)
        y = event_xy[:, 1].long().clamp_(0, height - 1)
        linear_idx = y * width + x

        if event_time_range is not None:
            event_time_range = event_time_range.to(device=device, dtype=dtype)
            start_t = event_time_range[0]
            end_t = event_time_range[1]
        else:
            start_t = event_t.min()
            end_t = event_t.max()

        duration = (end_t - start_t).clamp_min(1.0)
        norm_t = ((event_t - start_t) / duration).clamp_(0.0, 1.0)

        pos_mask = event_p > 0
        neg_mask = ~pos_mask

        rep[0] = self._build_count_map(linear_idx, pos_mask, height, width, device, dtype)
        rep[1] = self._build_count_map(linear_idx, neg_mask, height, width, device, dtype)
        rep[2] = self._build_latest_time_map(linear_idx, norm_t, pos_mask, height, width, device, dtype)
        rep[3] = self._build_latest_time_map(linear_idx, norm_t, neg_mask, height, width, device, dtype)
        rep[4].view(-1).index_fill_(0, linear_idx.unique(), 1.0)
        return rep

    @staticmethod
    def _build_count_map(
        linear_idx: torch.Tensor,
        mask: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        counts = torch.zeros(height * width, device=device, dtype=dtype)
        if mask.any():
            counts.index_add_(0, linear_idx[mask], torch.ones(int(mask.sum().item()), device=device, dtype=dtype))
            counts = torch.log1p(counts)
            counts = counts / counts.max().clamp_min(1.0)
        return counts.view(height, width)

    @staticmethod
    def _build_latest_time_map(
        linear_idx: torch.Tensor,
        norm_t: torch.Tensor,
        mask: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        latest = torch.zeros(height * width, device=device, dtype=dtype)
        if not mask.any():
            return latest.view(height, width)

        idx = linear_idx[mask]
        values = norm_t[mask]
        try:
            latest.scatter_reduce_(0, idx, values, reduce="amax", include_self=False)
        except (AttributeError, RuntimeError, TypeError):
            for event_idx, event_value in zip(idx.tolist(), values.tolist()):
                latest[event_idx] = max(float(latest[event_idx].item()), float(event_value))
        return latest.view(height, width)


def batch_idx_select(items: List[List[torch.Tensor]], batch_idx: int, frame_idx: int) -> torch.Tensor:
    return items[batch_idx][frame_idx]
