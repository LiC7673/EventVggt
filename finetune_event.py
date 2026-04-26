import datetime
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from eventvggt.datasets.my_event_dataset import (
    event_multiview_collate,
    get_combined_dataset,
)
from eventvggt.models.streamvggt import StreamVGGT as EventStreamVGGT
from eventvggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

torch.backends.cuda.matmul.allow_tf32 = True

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

printer = get_logger(__name__, log_level="INFO")


def save_current_code(outdir: str):
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d-%H-%M-%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", date_time)
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def maybe_denormalize_views(views: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    for view in views:
        if "img" in view:
            view["img"] = (view["img"] + 1.0) / 2.0
        # 法向图归一化到[-1,1]（如有）
        if "normal" in view:
            n = view["normal"]
            if torch.is_tensor(n):
                if n.max() > 2.0:
                    n = n / 127.5 - 1.0
                view["normal"] = n
            elif isinstance(n, np.ndarray):
                n = torch.from_numpy(n).float() / 127.5 - 1.0
                view["normal"] = n
    return views


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def build_event_loader(cfg, split="train"):
    dataset = get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=cfg.data.scene_names if cfg.data.scene_names else None,
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
    )
    if len(dataset) <= 0:
        scene_stats = []
        for scene_name, meta in getattr(dataset, "active_scene_data", {}).items():
            scene_stats.append(
                {
                    "scene": scene_name,
                    "frame_count": int(meta["frame_count"]),
                    "num_start_ids": int(len(meta["start_ids"])),
                }
            )
        raise ValueError(
            f"Dataset has no valid samples under {cfg.data.root}. "
            f"num_views={cfg.data.num_views}, active_scenes={dataset.get_active_scenes()}, "
            f"scene_stats={scene_stats}"
        )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
        collate_fn=event_multiview_collate,
    )
    printer.info(
        "Event dataset loaded from %s with %d active scenes and %d samples",
        cfg.data.root,
        len(dataset.get_active_scenes()),
        len(dataset),
    )
    return data_loader


def configure_trainable_params(model: EventStreamVGGT, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Only train event_encoder (these modules actually exist)
    enabled_prefixes = ["event_encoder"]
    for name, param in model.named_parameters():
        if any(token in name for token in enabled_prefixes):
            param.requires_grad = True

    if cfg.train.unfreeze_heads:
        for module in (model.camera_head, model.depth_head, model.point_head, model.track_head):
            for param in module.parameters():
                param.requires_grad = True

    if cfg.train.unfreeze_aggregator_blocks:
        for param in model.aggregator.frame_blocks.parameters():
            param.requires_grad = True
        for param in model.aggregator.global_blocks.parameters():
            param.requires_grad = True


def log_trainable_params(model: nn.Module) -> None:
    total_params = 0
    trainable_params = 0
    trainable_names = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)

    printer.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / max(total_params, 1),
    )
    if trainable_names:
        printer.info(
            "Example trainable parameters: %s%s",
            ", ".join(trainable_names[:8]),
            "..." if len(trainable_names) > 8 else "",
        )


def save_checkpoint(accelerator, model, optimizer, loss_scaler, cfg, epoch, global_step, best_loss):
    ckpt = {
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "best_loss": best_loss,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    ckpt_path = Path(cfg.output_dir) / "checkpoint-last.pth"
    misc.save_on_master(accelerator, ckpt, ckpt_path)


def stack_view_field(views: List[Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.stack([view[key] for view in views], dim=1)


def build_valid_mask(
    views: List[Dict[str, torch.Tensor]],
    depth_gt: torch.Tensor,
    depth_min: float = 1e-6,
    depth_max: Optional[float] = None,
) -> torch.Tensor:
    if "valid_mask" in views[0]:
        mask = stack_view_field(views, "valid_mask")
    else:
        mask = depth_gt > 0
    mask = mask.bool()
    mask = mask & torch.isfinite(depth_gt) & (depth_gt > depth_min)
    if depth_max is not None and depth_max > 0:
        mask = mask & (depth_gt < depth_max)
    return mask


def depth_to_world_points(depth: torch.Tensor, intrinsics: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    _, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.view(1, 1, height, width)
    ys = ys.view(1, 1, height, width)

    fx = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

    x_cam = (xs - cx) * depth / fx.clamp_min(1e-6)
    y_cam = (ys - cy) * depth / fy.clamp_min(1e-6)
    z_cam = depth
    cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    rot = pose[..., :3, :3]
    trans = pose[..., :3, 3]
    world_points = torch.einsum("bsij,bshwj->bshwi", rot, cam_points) + trans.unsqueeze(-2).unsqueeze(-2)
    return world_points


def camera_pose_to_pose_encoding(
    camera_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: Tuple[int, int],
) -> torch.Tensor:
    """Convert dataset camera_pose into VGGT pose encoding.

    The dataset stores camera_pose as a 4x4 camera-to-world transform (c2w).
    VGGT pose encoding expects world-to-camera extrinsics in OpenCV coordinates.
    """
    if camera_pose.shape[-2:] == (3, 4):
        bottom_row = torch.tensor([0, 0, 0, 1], device=camera_pose.device, dtype=camera_pose.dtype)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(*camera_pose.shape[:-2], 1, 4)
        camera_pose = torch.cat([camera_pose, bottom_row], dim=-2)

    if camera_pose.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera_pose shape [...,4,4] or [...,3,4], got {camera_pose.shape}")

    w2c = torch.linalg.inv(camera_pose)
    return extri_intri_to_pose_encoding(
        w2c[..., :3, :],
        intrinsics,
        image_size_hw=image_size_hw,
    )


def ensure_homogeneous_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-2:] == (3, 4):
        bottom_row = torch.tensor([0, 0, 0, 1], device=pose.device, dtype=pose.dtype)
        bottom_row = bottom_row.view(*([1] * (pose.ndim - 2)), 1, 4).expand(*pose.shape[:-2], 1, 4)
        pose = torch.cat([pose, bottom_row], dim=-2)
    if pose.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose shape [...,4,4] or [...,3,4], got {pose.shape}")
    return pose


def pose_encoding_to_c2w(
    pose_encoding: torch.Tensor,
    image_size_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    w2c, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding,
        image_size_hw=image_size_hw,
    )
    w2c = ensure_homogeneous_pose(w2c)
    c2w = torch.linalg.inv(w2c)
    return c2w, intrinsics


def relative_c2w_to_first(c2w: torch.Tensor) -> torch.Tensor:
    c2w = ensure_homogeneous_pose(c2w)
    first_inv = torch.linalg.inv(c2w[:, 0:1])
    return torch.matmul(first_inv, c2w)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    transform = ensure_homogeneous_pose(transform)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    return torch.einsum("bsij,bshwj->bshwi", rot, points) + trans.unsqueeze(-2).unsqueeze(-2)


def depth_to_camera_points(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Convert depth map to camera coordinate points using only intrinsics."""
    _, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype
    
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.view(1, 1, height, width)
    ys = ys.view(1, 1, height, width)

    fx = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

    x_cam = (xs - cx) * depth / fx.clamp_min(1e-6)
    y_cam = (ys - cy) * depth / fy.clamp_min(1e-6)
    z_cam = depth
    
    cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    return cam_points


def align_depth_scale(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    aligned_depth_gt = depth_gt.clone()
    scales = []
    batch, seq = depth_gt.shape[:2]
    for b in range(batch):
        seq_scales = []
        for s in range(seq):
            mask = valid_mask[b, s] & torch.isfinite(depth_pred[b, s]) & (depth_pred[b, s] > eps)
            if mask.sum() < 16:
                scale = depth_gt.new_tensor(1.0)
            else:
                ratio = depth_pred[b, s][mask] / depth_gt[b, s][mask].clamp_min(eps)
                finite_ratio = ratio[torch.isfinite(ratio)]
                scale = finite_ratio.median() if finite_ratio.numel() > 0 else depth_gt.new_tensor(1.0)
                scale = scale.clamp(1e-3, 1e3)
            aligned_depth_gt[b, s] = depth_gt[b, s] * scale
            seq_scales.append(scale)
        scales.append(torch.stack(seq_scales))
    return aligned_depth_gt, torch.stack(scales)


def compute_normals_from_points(points: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    normals = torch.zeros_like(points)
    dx = points[..., 1:, :-1, :] - points[..., :-1, :-1, :]
    dy = points[..., :-1, 1:, :] - points[..., :-1, :-1, :]
    core_normals = torch.cross(dx, dy, dim=-1)
    core_normals = F.normalize(core_normals, dim=-1, eps=1e-6)
    normals[..., :-1, :-1, :] = core_normals
    if valid_mask is not None:
        normals = normals * valid_mask.unsqueeze(-1).to(normals.dtype)
    return normals
def masked_chamfer_distance(pred_pts: torch.Tensor, gt_pts: torch.Tensor, mask: torch.Tensor, num_samples: int = 4096) -> torch.Tensor:
    """
    计算掩码点云之间的 Chamfer Distance。
    为了防止显存爆炸 (OOM)，采用了每帧独立降采样的策略。
    
    参数:
        pred_pts: [B, S, H, W, 3] 或 [B, H, W, 3]
        gt_pts:   [B, S, H, W, 3] 或 [B, H, W, 3]
        mask:     [B, S, H, W] 或 [B, H, W]
        num_samples: 随机采样的点数上限
    """
    # 统一降维到 [Batch, N, 3] 结构来处理，避免 B 和 S 维度的复杂性
    pred_flat = pred_pts.reshape(-1, pred_pts.shape[-3], pred_pts.shape[-2], 3)
    gt_flat = gt_pts.reshape(-1, gt_pts.shape[-3], gt_pts.shape[-2], 3)
    mask_flat = mask.reshape(-1, mask.shape[-2], mask.shape[-1]).bool()
    
    B_total = pred_flat.shape[0]
    total_loss = 0.0
    valid_batches = 0
    
    for i in range(B_total):
        # 1. 提取当前帧有效点
        valid_mask = mask_flat[i]
        p_pred = pred_flat[i][valid_mask] # [N_pred, 3]
        p_gt = gt_flat[i][valid_mask]     # [N_gt, 3]
        
        N_pred, N_gt = p_pred.shape[0], p_gt.shape[0]
        
        # 过滤极少有效点的异常帧
        if N_pred < 10 or N_gt < 10:
            continue
            
        # 2. 随机降采样 (防显存爆炸的核心)
        if N_pred > num_samples:
            idx_pred = torch.randperm(N_pred, device=p_pred.device)[:num_samples]
            p_pred = p_pred[idx_pred]
        if N_gt > num_samples:
            idx_gt = torch.randperm(N_gt, device=p_gt.device)[:num_samples]
            p_gt = p_gt[idx_gt]
            
        # 3. 计算距离矩阵 (由于采过样，这里最大只会是 num_samples x num_samples)
        # torch.cdist 支持批量计算，需要扩展一个维度 -> [1, N, 3]
        dist_matrix = torch.cdist(p_pred.unsqueeze(0), p_gt.unsqueeze(0)).squeeze(0) # [N_pred, N_gt]
        
        # 4. Chamfer Distance 核心公式
        # 从 pred 找最近的 gt
        min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=1)
        # 从 gt 找最近的 pred
        min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=0)
        
        # 论文标准公式：两项求均值后再相加
        frame_cd_loss = min_dist_pred_to_gt.mean() + min_dist_gt_to_pred.mean()
        
        total_loss += frame_cd_loss
        valid_batches += 1
        
    if valid_batches == 0:
        return pred_pts.new_tensor(0.0, requires_grad=True)
        
    return total_loss / valid_batches

class EventSupervisedLoss(nn.Module):
    def __init__(
        self,
        pose_weight: float = 1.0,
        depth_weight: float = 1.0,
        points_weight: float = 1.0,
        normal_weight: float = 0.0,
        depth_min: float = 1e-6,
        depth_max: Optional[float] = None,
        align_depth_scale_enabled: bool = False,
        points_loss_type: str = "cd",  # "l1" or "cd" (Chamfer Distance)
    ):
        super().__init__()
        self.pose_weight = pose_weight
        self.depth_weight = depth_weight
        self.points_weight = points_weight
        self.normal_weight = normal_weight
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.align_depth_scale_enabled = align_depth_scale_enabled
        self.points_loss_type = points_loss_type.lower()
        
        if self.points_loss_type not in ["l1", "cd"]:
            raise ValueError(f"points_loss_type must be 'l1' or 'cd', got '{self.points_loss_type}'")

    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        pred = model_output.ress

        depth_pred = torch.stack([res["depth"] for res in pred], dim=1).squeeze(-1)
        points_pred = torch.stack([res["pts3d_in_other_view"] for res in pred], dim=1)
        pose_pred = torch.stack([res["camera_pose"] for res in pred], dim=1)

        depth_gt = stack_view_field(views, "depthmap").to(device=depth_pred.device, dtype=depth_pred.dtype)
        intrinsics_gt = stack_view_field(views, "camera_intrinsics").to(device=depth_pred.device, dtype=depth_pred.dtype)
        pose_matrix_gt = stack_view_field(views, "camera_pose").to(device=depth_pred.device, dtype=depth_pred.dtype)
        valid_mask = build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)

        if self.align_depth_scale_enabled:
            depth_gt_aligned, depth_scales = align_depth_scale(depth_pred.detach(), depth_gt, valid_mask)
        else:
            depth_gt_aligned = depth_gt
            depth_scales = depth_gt.new_ones(depth_gt.shape[:2])

        points_gt_aligned = depth_to_world_points(depth_gt_aligned, intrinsics_gt, pose_matrix_gt)
        points_gt = points_gt_aligned  # Use world coordinates for consistency with points_pred
        points_mask = valid_mask.unsqueeze(-1).expand_as(points_gt)
       

        # points_gt_aligned = depth_to_world_points(depth_gt_aligned, intrinsics_gt, pose_matrix_gt)
        # points_pred = torch.stack([res["pts3d_in_other_view"] for res in pred], dim=1)

        # 单独为可视化生成一个未被 scale 污染的纯净 GT 点云
        # points_gt_pure = depth_to_world_points(depth_gt, intrinsics_gt, pose_matrix_gt)

        # 计算 Loss 时依然使用对齐后的 points_gt_aligned
        # points_loss = masked_l1(points_pred, points_gt_aligned, points_mask)
        # points_loss =   masked_chamfer_distance(points_pred, points_gt, points_mask) 

        normal_mask = valid_mask.clone()
        normal_mask[..., 0, :] = False
        normal_mask[..., -1, :] = False
        normal_mask[..., :, 0] = False
        normal_mask[..., :, -1] = False

        # 优先用GT法向图
        if all('normal' in v for v in views):
            gt_normals = stack_view_field(views, 'normal')  # [B, S, 3, H, W] or [B, S, H, W, 3]
            # 调整到[B, S, H, W, 3]
            if gt_normals.ndim == 5 and gt_normals.shape[2] == 3:
                gt_normals = gt_normals.permute(0, 1, 3, 4, 2)
            elif gt_normals.ndim == 5 and gt_normals.shape[-1] != 3:
                gt_normals = gt_normals.permute(0, 1, 2, 3, 4)
        else:
            gt_normals = depth_to_normals(depth_gt, intrinsics_gt)
        pred_normals = depth_to_normals(depth_pred, intrinsics_gt)
        normal_loss = masked_cosine_loss(pred_normals, gt_normals, normal_mask)

        # ============ 计算 Pose 对齐 ============
        # 将数据集里的 camera_pose (c2w) 转成 VGGT 的 pose encoding 输入格式
        # Matrix-relative supervision: frame 0 is an anchor only.
        # Frames 1..S-1 are compared in their first-frame coordinate systems.
        height, width = depth_gt.shape[-2:]
        pred_c2w, _ = pose_encoding_to_c2w(pose_pred, image_size_hw=(height, width))
        gt_c2w = ensure_homogeneous_pose(pose_matrix_gt)
        
        # 计算第一帧的pose对齐矩阵 (used to align other frames)
        pose_pred_relative = relative_c2w_to_first(pred_c2w)
        pose_gt_relative = relative_c2w_to_first(gt_c2w)

        pred_world_to_first = torch.linalg.inv(pred_c2w[:, 0:1]).expand(-1, points_pred.shape[1], -1, -1)
        gt_world_to_first = torch.linalg.inv(gt_c2w[:, 0:1]).expand(-1, points_gt.shape[1], -1, -1)
        points_pred_aligned = transform_points(points_pred, pred_world_to_first)
        points_gt_aligned_for_loss = transform_points(points_gt, gt_world_to_first)

        # ============ Pose Loss 计算 ============
        # 将对齐矩阵应用到其他帧的预测 pose (frame 1 onwards)
        
        # Pose loss 只计算第一帧以外的帧 (skip first frame for pose supervision)
        if pose_pred_relative.shape[1] > 1:
            pose_loss = F.smooth_l1_loss(
                pose_pred_relative[:, 1:, :3, :],
                pose_gt_relative[:, 1:, :3, :],
            )
        else:
            # 如果只有一帧，则pose loss为0
            pose_loss = pose_pred.new_tensor(0.0, requires_grad=True)
        
        # depth_loss = masked_l1(depth_pred, depth_gt_aligned, valid_mask)
        depth_loss = masked_l1(depth_pred, depth_gt_aligned, valid_mask)
        
        # points_loss 根据 points_loss_type 选择 L1 或 Chamfer Distance
        if points_pred_aligned.shape[1] > 1:
            if self.points_loss_type == "l1":
                points_loss = masked_l1(
                    points_pred_aligned[:, 1:],
                    points_gt_aligned_for_loss[:, 1:],
                    points_mask[:, 1:],
                )
            else:  # "cd"
                points_loss = masked_chamfer_distance(
                    points_pred_aligned[:, 1:],
                    points_gt_aligned_for_loss[:, 1:],
                    valid_mask[:, 1:],
                )
        else:
            points_loss = points_pred.new_tensor(0.0, requires_grad=True)

        total_loss = (
            # self.depth_weight * depth_loss
            self.pose_weight * pose_loss
            + self.depth_weight * depth_loss
            + self.points_weight * points_loss
            # + self.normal_weight * normal_loss
        )

        details = {
            "pose_loss": float(pose_loss.detach()),
            "depth_loss": float(depth_loss.detach()),
            "points_loss": float(points_loss.detach()),
            "normal_loss": float(normal_loss.detach()),
            "depth_scale": float(depth_scales.mean().detach()),
        }
        aux = {
            "depth_pred": depth_pred.detach(),
            "depth_gt": depth_gt.detach(),
            "depth_gt_aligned": depth_gt_aligned.detach(),
            "points_pred": points_pred_aligned.detach(),
            "points_gt": points_gt_aligned_for_loss.detach(),
            "valid_mask": valid_mask.detach(),
            "pose_pred": pose_pred_relative.detach(),
            "pose_gt": pose_gt_relative.detach(),
        }
        return total_loss, details, aux


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device)
    if mask.dtype != pred.dtype:
        mask = mask.to(dtype=pred.dtype)
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1, eps=1e-6)
    target = F.normalize(target, dim=-1, eps=1e-6)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    loss = 1.0 - cos
    if mask is not None:
        mask = mask.to(device=loss.device)
        if mask.dtype != loss.dtype:
            mask = mask.to(dtype=loss.dtype)
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = loss.numel()
    return loss.sum() / denom


def tensor_rgb_to_uint8(img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
    img = img.detach().float().cpu().clamp(0.0, 1.0)
    img_np = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if mask is not None:
        mask_np = mask.detach().cpu().numpy().astype(bool)
        if mask_np.shape != img_np.shape[:2]:
            mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
            resized_mask = F.interpolate(mask_tensor, size=img_np.shape[:2], mode='nearest').squeeze().numpy() > 0.5
            img_np[~resized_mask] = 0
        else:
            img_np[~mask_np] = 0
    return img_np


def depth_to_uint8(depth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
    depth_np = depth.detach().float().cpu().numpy()
    if mask is not None:
        mask_np = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.isfinite(depth_np)

    # Always perform normalization for visualization
    if mask_np.any():
        valid = depth_np[mask_np]
        lo = np.percentile(valid, 2.0)
        hi = np.percentile(valid, 98.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(valid.min())
            hi = float(valid.max()) if valid.size > 0 else lo + 1.0
    else:
        # Fallback normalization even without valid mask
        valid = depth_np[np.isfinite(depth_np)]
        if valid.size > 0:
            lo = float(valid.min())
            hi = float(valid.max())
        else:
            lo, hi = 0.0, 1.0

    denom = max(hi - lo, 1e-6)
    norm = np.clip((depth_np - lo) / denom, 0.0, 1.0)
    gray = (norm * 255.0).round().astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    if mask is not None:
        rgb[~mask_np] = 0
    return rgb


def normal_to_uint8(normal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
    # Always normalize normals for visualization
    normal = F.normalize(normal.detach().float(), dim=-1, eps=1e-6).cpu()
    rgb = (((normal + 1.0) * 0.5).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.ndim == 3 and rgb.shape[2] != 3:
        if rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        elif rgb.shape[1] == 3:
            rgb = rgb.transpose(0, 2, 1)
        else:
            h = rgb.shape[0]
            w = rgb.shape[1]
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if mask is not None:
        mask_np = mask.detach().cpu().numpy().astype(bool)
        if mask_np.shape != rgb.shape[:2]:
            mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
            resized_mask = F.interpolate(mask_tensor, size=rgb.shape[:2], mode='nearest').squeeze().numpy() > 0.5
            rgb[~resized_mask] = 0
        else:
            rgb[~mask_np] = 0
    return rgb


def make_labeled_panel(label: str, image_array: np.ndarray) -> Image.Image:
    panel = Image.fromarray(image_array)
    label_bar = Image.new("RGB", (panel.width, 24), color=(20, 20, 20))
    ImageDraw.Draw(label_bar).text((6, 5), label, fill=(220, 220, 220))
    out = Image.new("RGB", (panel.width, panel.height + 24))
    out.paste(label_bar, (0, 0))
    out.paste(panel, (0, 24))
    return out


def depth_to_normals(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    # depth: [..., H, W], intrinsics: [..., 3, 3]
    *batch_dims, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    expand_dims = [1] * len(batch_dims) + [H, W]
    xs = xs.view(*expand_dims)
    ys = ys.view(*expand_dims)

    fx = intrinsics[..., 0, 0].view(*batch_dims, 1, 1)
    fy = intrinsics[..., 1, 1].view(*batch_dims, 1, 1)
    cx = intrinsics[..., 0, 2].view(*batch_dims, 1, 1)
    cy = intrinsics[..., 1, 2].view(*batch_dims, 1, 1)

    x = (xs - cx) / fx.clamp_min(1e-6) * depth
    y = (ys - cy) / fy.clamp_min(1e-6) * depth
    z = depth
    pts = torch.stack([x, y, z], dim=-1)

    fx_diff = pts[..., 1:-1, 2:, :] - pts[..., 1:-1, :-2, :]
    fy_diff = pts[..., 2:, 1:-1, :] - pts[..., :-2, 1:-1, :]
    core = torch.cross(fy_diff, fx_diff, dim=-1)
    core = F.normalize(core, dim=-1, eps=1e-6)

    normals = torch.zeros_like(pts)
    normals[..., 1:-1, 1:-1, :] = core
    return normals


def save_pointcloud_ply(pointcloud: torch.Tensor, rgb: np.ndarray, mask: torch.Tensor, path: Path) -> None:
    pointcloud = pointcloud.detach().float().cpu()
    if mask is not None:
        mask = mask.detach().bool().cpu()
    else:
        mask = torch.ones(pointcloud.shape[:-1], dtype=torch.bool)

    points = pointcloud[mask].reshape(-1, 3).cpu().numpy()
    colors = rgb[mask.cpu().numpy()].reshape(-1, 3).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def normalize_normal_tensor(normal: torch.Tensor) -> torch.Tensor:
    """Convert normal tensor to [H, W, 3] format."""
    if normal.ndim == 2:
        normal = normal.unsqueeze(-1).repeat(1, 1, 3)
    elif normal.ndim == 3:
        if normal.shape[2] == 3:
            return normal
        if normal.shape[0] == 3 and normal.shape[1] > 1 and normal.shape[2] > 1:
            return normal.permute(1, 2, 0)
        if normal.shape[1] == 3 and normal.shape[0] > 1 and normal.shape[2] > 1:
            return normal.permute(0, 2, 1)
        if normal.shape[1] == 1 and normal.shape[2] == 3:
            return normal.permute(0, 2, 1)
        if normal.shape[0] == 1 and normal.shape[2] == 3:
            return normal.squeeze(0).permute(1, 0, 2)
        # fallback: try to infer H,W from size if divisible by 3
        total = normal.numel()
        if total % 3 == 0:
            hw = total // 3
            h = int(math.sqrt(hw))
            if h * h == hw:
                return normal.view(h, h, 3)
    elif normal.ndim == 4 and normal.shape[0] == 1:
        if normal.shape[1] == 3:
            return normal.squeeze(0).permute(1, 2, 0)
        if normal.shape[3] == 3:
            return normal.squeeze(0)
    # unsupported shape, fallback to zero normal map with last known 3 channels if possible
    if normal.ndim >= 2:
        h = normal.shape[-2]
        w = normal.shape[-1]
        return torch.zeros((h, w, 3), dtype=normal.dtype, device=normal.device)
    return torch.zeros((1, 1, 3), dtype=normal.dtype, device=normal.device)


@torch.no_grad()
def evaluate_on_test_set(
    model: nn.Module,
    data_loader,
    criterion,
    accelerator: Accelerator,
    cfg,
    global_step: int,
    log_writer=None,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """Evaluate model on test set and return metrics."""
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    
    test_vis_dir = Path(cfg.output_dir) / "test_vis" / f"step_{global_step:07d}"
    test_vis_dir.mkdir(parents=True, exist_ok=True)
    
    aux_accum = {
        "depth_pred_all": [],
        "depth_gt_all": [],
        "valid_mask_all": [],
    }
    
    for batch_idx, views in enumerate(metric_logger.log_every(data_loader, cfg.print_freq, accelerator, "Test Batch")):
        views = maybe_denormalize_views(views)
        
        # Move views to device and convert to model's dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        for view in views:
            for key in view:
                if isinstance(view[key], torch.Tensor):
                    view[key] = view[key].to(device=device, dtype=dtype if key != 'events' else dtype)
        
        model_output = model(views)
        loss, loss_details, aux = criterion(model_output, views)
        
        loss_value = float(loss.detach())
        metric_logger.update(loss=loss_value)
        metric_logger.update(**loss_details)
        
        # Accumulate for visualization (limit to first few samples)
        if batch_idx < 2:  # Save visualization for first 2 batches
            batch_size = aux["depth_pred"].shape[0]
            num_views = aux["depth_pred"].shape[1]
            num_views_to_save = min(10, num_views)
            
            for i in range(min(batch_size, 1)):
                for frame_idx in range(num_views_to_save):
                    aux_accum["depth_pred_all"].append(aux["depth_pred"][i, frame_idx])
                    aux_accum["depth_gt_all"].append(aux["depth_gt"][i, frame_idx])
                    aux_accum["valid_mask_all"].append(aux["valid_mask"][i, frame_idx])
                
                # Save per-batch visualization for each frame
                for frame_idx in range(num_views_to_save):
                    save_training_visuals(
                        cfg,
                        views,
                        aux,
                        global_step * 1000 + batch_idx,
                        frame_idx,
                        i,
                        vis_dir=test_vis_dir,
                        force_save=True,
                    )
    
    metric_logger.synchronize_between_processes(accelerator)
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if log_writer is not None:
        for key, value in test_stats.items():
            log_writer.add_scalar(f"test/{key}", value, global_step)
    
    printer.info("Test stats: %s", test_stats)
    
    return test_stats, aux_accum


def save_test_summary(cfg, epoch: int, global_step: int, test_stats: Dict[str, float]) -> None:
    """Save test summary to file."""
    summary_file = Path(cfg.output_dir) / "test_summary.txt"
    
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Epoch {epoch} - Step {global_step} - Test Results\n")
        f.write(f"{'='*80}\n")
        
        for metric_name in sorted(test_stats.keys()):
            metric_value = test_stats[metric_name]
            if isinstance(metric_value, (int, float)):
                if isinstance(metric_value, float):
                    f.write(f"  {metric_name:30s}: {metric_value:12.6f}\n")
                else:
                    f.write(f"  {metric_name:30s}: {metric_value:12d}\n")
        
        f.write(f"\n{'='*80}\n")
    
    printer.info(f"Test summary saved to {summary_file}")


def save_metrics_json(cfg, epoch: int, global_step: int, train_stats: Dict[str, float], 
                     test_stats: Optional[Dict[str, float]] = None) -> None:
    """Save training and test metrics to JSON file."""
    metrics_file = Path(cfg.output_dir) / "metrics.json"
    
    metrics_entry = {
        "epoch": epoch,
        "step": global_step,
        "train": train_stats,
    }
    
    if test_stats is not None:
        metrics_entry["test"] = test_stats
    
    # Load existing metrics or create new list
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            try:
                all_metrics = json.load(f)
                if not isinstance(all_metrics, list):
                    all_metrics = [all_metrics]
            except:
                all_metrics = []
    else:
        all_metrics = []
    
    all_metrics.append(metrics_entry)
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


def generate_loss_plots(cfg) -> None:
    """Generate and save loss plots for pose, depth, and points losses."""
    metrics_file = Path(cfg.output_dir) / "metrics.json"
    if not metrics_file.exists():
        printer.warning("Metrics file not found, skipping loss plots")
        return
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)
    
    if not isinstance(all_metrics, list):
        all_metrics = [all_metrics]
    
    # Extract data
    steps = []
    train_pose_loss = []
    train_depth_loss = []
    train_points_loss = []
    test_pose_loss = []
    test_depth_loss = []
    test_points_loss = []
    
    for entry in all_metrics:
        step = entry.get("step", 0)
        steps.append(step)
        
        train = entry.get("train", {})
        train_pose_loss.append(train.get("pose_loss", None))
        train_depth_loss.append(train.get("depth_loss", None))
        train_points_loss.append(train.get("points_loss", None))
        
        test = entry.get("test", {})
        test_pose_loss.append(test.get("pose_loss", None))
        test_depth_loss.append(test.get("depth_loss", None))
        test_points_loss.append(test.get("points_loss", None))
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    loss_names = ["Pose Loss", "Depth Loss", "Points Loss"]
    train_losses = [train_pose_loss, train_depth_loss, train_points_loss]
    test_losses = [test_pose_loss, test_depth_loss, test_points_loss]
    
    for i, (name, train_loss, test_loss) in enumerate(zip(loss_names, train_losses, test_losses)):
        ax = axes[i]
        
        # Plot train loss
        valid_train = [(s, l) for s, l in zip(steps, train_loss) if l is not None]
        if valid_train:
            train_steps, train_vals = zip(*valid_train)
            ax.plot(train_steps, train_vals, label="Train", marker='o', markersize=2)
        
        # Plot test loss
        valid_test = [(s, l) for s, l in zip(steps, test_loss) if l is not None]
        if valid_test:
            test_steps, test_vals = zip(*valid_test)
            ax.plot(test_steps, test_vals, label="Test", marker='s', markersize=4, color='red')
        
        ax.set_title(f"{name} over Training Steps")
        ax.set_xlabel("Global Step")
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(cfg.output_dir) / "loss_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    printer.info(f"Loss plots saved to {plot_path}")


def save_pointcloud_ply(pointcloud: torch.Tensor, rgb: np.ndarray, mask: torch.Tensor, path: Path) -> None:
    # pointcloud: [H, W, 3], rgb: [H, W, 3], mask: [H, W]
    points = pointcloud[mask].reshape(-1, 3).cpu().numpy()
    colors = rgb[mask.cpu().numpy()].reshape(-1, 3).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_training_visuals(
    cfg,
    views: List[Dict[str, torch.Tensor]],
    aux: Dict[str, torch.Tensor],
    global_step: int,
    frame_idx: Optional[int] = None,
    sample_idx: Optional[int] = None,
    vis_dir: Optional[Path] = None,
    force_save: bool = False,
) -> None:
    vis_cfg = getattr(cfg, "vis", None)
    save_every = getattr(vis_cfg, "save_every_steps", 200)
    if not force_save and (save_every <= 0 or global_step % save_every != 0):
        return

    if sample_idx is None:
        sample_idx = int(getattr(vis_cfg, "sample_index", 0))
    
    # 获取所有视角数量
    num_views = len(views)
    
    # 如果指定了frame_idx，只保存该frame；否则保存所有frame（最多10张）
    if frame_idx is not None:
        frames_to_save = [frame_idx]
    else:
        num_views_to_save = min(10, num_views)
        frames_to_save = list(range(num_views_to_save))
    
    if vis_dir is None:
        vis_dir = Path(cfg.output_dir) / "train_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 为每个视角保存可视化面板和点云
    for frame_id in frames_to_save:
        rgb = views[frame_id]["img"][sample_idx]
        depth_gt = aux["depth_gt"][sample_idx, frame_id]
        depth_pred = aux["depth_pred"][sample_idx, frame_id]
        valid_mask = aux["valid_mask"][sample_idx, frame_id]

        intrinsics = views[frame_id]["camera_intrinsics"][sample_idx]
        pred_normals = depth_to_normals(depth_pred, intrinsics)
        gt_normals = depth_to_normals(depth_gt, intrinsics)

        panels = [
            make_labeled_panel("rgb", tensor_rgb_to_uint8(rgb, valid_mask)),
            make_labeled_panel("gt_depth", depth_to_uint8(depth_gt, valid_mask)),
            make_labeled_panel("pred_depth", depth_to_uint8(depth_pred, valid_mask)),
            make_labeled_panel("pred_normal", normal_to_uint8(pred_normals, valid_mask)),
            make_labeled_panel("gt_normal", normal_to_uint8(gt_normals, valid_mask)),
        ]

        total_width = sum(panel.width for panel in panels)
        max_height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
        x_offset = 0
        for panel in panels:
            canvas.paste(panel, (x_offset, 0))
            x_offset += panel.width

        # 保存可视化面板，frame编号为frame_id
        canvas_name = f"{timestamp}_step_{global_step:07d}_frame_{frame_id:02d}.png"
        canvas.save(vis_dir / canvas_name)

        # 保存预测点云
        pointcloud = aux["points_pred"][sample_idx, frame_id]  # [H, W, 3]
        if isinstance(rgb, torch.Tensor):
            rgb_np = tensor_rgb_to_uint8(rgb, valid_mask)
        else:
            rgb_np = np.array(rgb)

        pcl_name = f"{timestamp}_step_{global_step:07d}_frame_{frame_id:02d}.ply"
        save_pointcloud_ply(pointcloud, rgb_np, valid_mask, vis_dir / pcl_name)

        # 保存GT点云
        gt_pointcloud = aux["points_gt"][sample_idx, frame_id]  # [H, W, 3]
        gt_pcl_name = f"{timestamp}_step_{global_step:07d}_frame_{frame_id:02d}_gt.ply"
        save_pointcloud_ply(gt_pointcloud, rgb_np, valid_mask, vis_dir / gt_pcl_name)


def train(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accum_iter,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
        dst_dir = save_current_code(cfg.output_dir)
        printer.info("Saved current code snapshot to %s", dst_dir)

    seed = cfg.seed + accelerator.process_index
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = cfg.benchmark

    # Build train and test dataloaders
    data_loader_train = build_event_loader(cfg, split="train")
    data_loader_test = build_event_loader(cfg, split="test")
    
    test_samples_count = len(data_loader_test) if data_loader_test else 0
    train_samples_count = len(data_loader_train)
    
    printer.info("Train dataset: %d samples, Test dataset: %d samples", train_samples_count, test_samples_count)

    model = EventStreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=cfg.model.event_hidden_dim,
    )

    if cfg.pretrained:
        printer.info("Loading model init weights from %s", cfg.pretrained)
        ckpt = unwrap_state_dict(torch.load(cfg.pretrained, map_location="cpu"))
        msg = model.load_state_dict(ckpt, strict=False)
        printer.info("Checkpoint load result: %s", msg)

    configure_trainable_params(model, cfg)
    log_trainable_params(model)

    criterion = EventSupervisedLoss(
        pose_weight=cfg.loss.pose_weight,
        depth_weight=cfg.loss.depth_weight,
        points_weight=cfg.loss.points_weight,
        normal_weight=float(getattr(cfg.loss, "normal_weight", 0.0)),
        depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
        depth_max=(float(cfg.loss.depth_max) if getattr(cfg.loss, "depth_max", None) is not None else None),
        align_depth_scale_enabled=bool(getattr(cfg.loss, "align_depth_scale",True)),
        points_loss_type=str(getattr(cfg.loss, "points_loss_type", "cd")),
    )
    
    printer.info("Loss configuration: pose_weight=%.4f, depth_weight=%.4f, points_weight=%.4f, points_loss_type=%s",
                 cfg.loss.pose_weight, cfg.loss.depth_weight, cfg.loss.points_weight, criterion.points_loss_type)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were enabled. Check configure_trainable_params().")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    loss_scaler = NativeScaler(accelerator=accelerator)

    model, optimizer, data_loader_train = accelerator.prepare(model, optimizer, data_loader_train)
    # Note: test loader is not prepared for evaluation flexibility
    criterion = criterion.to(accelerator.device)

    log_writer = SummaryWriter(log_dir=cfg.logdir) if accelerator.is_main_process else None

    best_loss = float("inf")
    global_step = 0
    start_time = time.time()
    
    # Test evaluation interval
    eval_every_steps = getattr(cfg, "eval_every_steps", max(cfg.save_every_steps, 500))

    for epoch in range(cfg.start_epoch, cfg.epochs):
        model.train()
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"

        for data_iter_step, views in enumerate(metric_logger.log_every(data_loader_train, cfg.print_freq, accelerator, header)):
            with accelerator.accumulate(model):

                optimizer.zero_grad(set_to_none=True)
                views = maybe_denormalize_views(views)
             
                if global_step % cfg.accum_iter == 0:
                    epoch_f = epoch + data_iter_step / max(len(data_loader_train), 1)
                    misc.adjust_learning_rate(optimizer, epoch_f, cfg)

                model_output = model(views)
                loss, loss_details, aux = criterion(model_output, views)

                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Non-finite loss detected: {loss_value}")

                loss_scaler(
                    loss,
                    optimizer,
                    parameters=[p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad],
                    update_grad=True,
                    clip_grad=cfg.clip_grad,
                )

                metric_logger.update(loss=loss_value)
                metric_logger.update(**loss_details)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                if accelerator.is_main_process and log_writer is not None and global_step % cfg.log_freq == 0:
                    log_writer.add_scalar("train/loss", loss_value, global_step)
                    log_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    for key, value in loss_details.items():
                        log_writer.add_scalar(f"train/{key}", value, global_step)

                if accelerator.is_main_process:
                    save_training_visuals(cfg, views, aux, global_step)

                # Periodic test evaluation
                if (accelerator.is_main_process and 
                    test_samples_count > 0 and 
                    eval_every_steps > 0 and 
                    global_step % eval_every_steps == 0 and 
                    global_step > 0):
                    printer.info("Running test evaluation at step %d", global_step)
                    test_stats, _ = evaluate_on_test_set(
                        accelerator.unwrap_model(model),
                        data_loader_test,
                        criterion,
                        accelerator,
                        cfg,
                        global_step,
                        log_writer=log_writer,
                    )
                    save_test_summary(cfg, epoch, global_step, test_stats)
                    save_metrics_json(cfg, epoch, global_step, {}, test_stats)
                    model.train()  # Switch back to train mode

                if accelerator.is_main_process and global_step % cfg.save_every_steps == 0 and global_step > 0:
                    save_checkpoint(accelerator, model, optimizer, loss_scaler, cfg, epoch, global_step, best_loss)

                best_loss = min(best_loss, loss_value)
                global_step += 1

        metric_logger.synchronize_between_processes(accelerator)
        epoch_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        printer.info("Epoch %d stats: %s", epoch, epoch_stats)

        if accelerator.is_main_process:
            with open(Path(cfg.output_dir) / "log.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "step": global_step, **epoch_stats}) + "\n")
            
            # Save metrics to JSON
            save_metrics_json(cfg, epoch, global_step, epoch_stats, test_stats=None)

        save_checkpoint(accelerator, model, optimizer, loss_scaler, cfg, epoch, global_step, best_loss)

    total_time = time.time() - start_time
    printer.info("Training finished in %.2f minutes", total_time / 60.0)

    # Final test evaluation and plots
    if accelerator.is_main_process:
        if test_samples_count > 0:
            printer.info("Running final test evaluation")
            test_stats, _ = evaluate_on_test_set(
                accelerator.unwrap_model(model),
                data_loader_test,
                criterion,
                accelerator,
                cfg,
                global_step,
                log_writer=log_writer,
            )
            save_test_summary(cfg, epoch, global_step, test_stats)
            save_metrics_json(cfg, epoch, global_step, {}, test_stats)  # Save final test stats
        generate_loss_plots(cfg)

    if log_writer is not None:
        log_writer.close()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent/ "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    train(cfg)


if __name__ == "__main__":
    run()
