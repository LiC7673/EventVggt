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
from eventvggt.models.streamvggt_rgb import StreamVGGT as RGBStreamVGGT
from eventvggt.utils.pose_enc import extri_intri_to_pose_encoding

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
    return views


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def build_rgb_loader(cfg, split="train"):
    """Build dataloader with RGB images (still loads event data but won't use it)"""
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
        ldr_event_id=getattr(cfg.data, "ldr_event_id", "auto"),
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
        "RGB %s dataset loaded from %s with %d active scenes and %d samples (uses RGB only, ignores events)",
        split,
        cfg.data.root,
        len(dataset.get_active_scenes()),
        len(dataset),
    )
    return data_loader


def configure_trainable_params(model: RGBStreamVGGT, cfg) -> None:
    """Configure trainable parameters for RGB-only model (no event_encoder)"""
    for _, param in model.named_parameters():
        param.requires_grad = False

    # For RGB model, train aggregator blocks (event_encoder doesn't exist)
    if cfg.train.unfreeze_aggregator_blocks:
        for param in model.aggregator.frame_blocks.parameters():
            param.requires_grad = True
        for param in model.aggregator.global_blocks.parameters():
            param.requires_grad = True

    if cfg.train.unfreeze_heads:
        for module in (model.camera_head, model.depth_head, model.point_head, model.track_head):
            if module is not None:
                for param in module.parameters():
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


class SupervisedLoss(nn.Module):
    """Loss for RGB-only (no event) training"""
    def __init__(
        self,
        pose_weight: float = 1.0,
        depth_weight: float = 1.0,
        points_weight: float = 1.0,
        normal_weight: float = 0.0,
        depth_min: float = 1e-6,
        depth_max: Optional[float] = None,
        align_depth_scale_enabled: bool = True,
    ):
        super().__init__()
        self.pose_weight = pose_weight
        self.depth_weight = depth_weight
        self.points_weight = points_weight
        self.normal_weight = normal_weight
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.align_depth_scale_enabled = align_depth_scale_enabled

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

        normal_mask = valid_mask.clone()
        normal_mask[..., 0, :] = False
        normal_mask[..., -1, :] = False
        normal_mask[..., :, 0] = False
        normal_mask[..., :, -1] = False

        if self.normal_weight > 0.0:
            pred_normals = depth_to_normals(depth_pred, intrinsics_gt)
            gt_normals = depth_to_normals(depth_gt, intrinsics_gt)
            normal_loss = masked_cosine_loss(pred_normals, gt_normals, normal_mask)
        else:
            normal_loss = depth_pred.new_tensor(0.0)

        height, width = depth_gt.shape[-2:]
        pose_gt = extri_intri_to_pose_encoding(
            pose_matrix_gt[..., :3, :],
            intrinsics_gt,
            image_size_hw=(height, width),
        ).to(device=pose_pred.device, dtype=pose_pred.dtype)
        if self.align_depth_scale_enabled:
            pose_gt[..., :3] = pose_gt[..., :3] * depth_scales.unsqueeze(-1)

        depth_loss = masked_l1(depth_pred, depth_gt_aligned, valid_mask)
        points_loss = masked_l1(points_pred, points_gt, points_mask)
        pose_loss = F.smooth_l1_loss(pose_pred, pose_gt)

        total_loss = (
            self.pose_weight * pose_loss
            + self.depth_weight * depth_loss
            + self.points_weight * points_loss
            + self.normal_weight * normal_loss
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
            "points_pred": points_pred.detach(),
            "points_gt": points_gt.detach(),
            "valid_mask": valid_mask.detach(),
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


def depth_to_normals(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    # depth: [..., H, W], intrinsics: [..., 3, 3]
    *batch_dims, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
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
    points = pointcloud[mask].reshape(-1, 3).cpu().numpy()
    colors = rgb[mask.cpu().numpy()].reshape(-1, 3).astype(np.uint8)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n')


def make_labeled_panel(label: str, image_array: np.ndarray) -> Image.Image:
    panel = Image.fromarray(image_array)
    label_bar = Image.new('RGB', (panel.width, 24), color=(20, 20, 20))
    ImageDraw.Draw(label_bar).text((6, 5), label, fill=(220, 220, 220))
    out = Image.new('RGB', (panel.width, panel.height + 24))
    out.paste(label_bar, (0, 0))
    out.paste(panel, (0, 24))
    return out


def save_training_visuals(
    cfg,
    views: List[Dict[str, torch.Tensor]],
    aux: Dict[str, torch.Tensor],
    global_step: int,
) -> None:
    vis_cfg = getattr(cfg, 'vis', None)
    save_every = getattr(vis_cfg, 'save_every_steps', 200)
    if save_every <= 0 or global_step % save_every != 0:
        return

    frame_idx = int(getattr(vis_cfg, 'frame_index', 0))
    sample_idx = int(getattr(vis_cfg, 'sample_index', 0))

    rgb = views[frame_idx]['img'][sample_idx]
    depth_gt = aux['depth_gt'][sample_idx, frame_idx]
    depth_pred = aux['depth_pred'][sample_idx, frame_idx]
    valid_mask = aux['valid_mask'][sample_idx, frame_idx]

    intrinsics = views[frame_idx]['camera_intrinsics'][sample_idx]
    pred_normals = depth_to_normals(depth_pred, intrinsics)
    gt_normals = depth_to_normals(depth_gt, intrinsics)

    panels = [
        make_labeled_panel('rgb', tensor_rgb_to_uint8(rgb, valid_mask)),
        make_labeled_panel('gt_depth', depth_to_uint8(depth_gt, valid_mask)),
        make_labeled_panel('pred_depth', depth_to_uint8(depth_pred, valid_mask)),
        make_labeled_panel('pred_normal', normal_to_uint8(pred_normals, valid_mask)),
        make_labeled_panel('gt_normal', normal_to_uint8(gt_normals, valid_mask)),
    ]

    total_width = sum(panel.width for panel in panels)
    max_height = max(panel.height for panel in panels)
    canvas = Image.new('RGB', (total_width, max_height), color=(0, 0, 0))
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width

    vis_dir = Path(cfg.output_dir) / 'train_vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    canvas_name = f'{timestamp}_step_{global_step:07d}.png'
    canvas.save(vis_dir / canvas_name)

    pointcloud = aux['points_pred'][sample_idx, frame_idx]
    if isinstance(rgb, torch.Tensor):
        rgb_np = tensor_rgb_to_uint8(rgb, valid_mask)
    else:
        rgb_np = np.array(rgb)

    pcl_name = f'{timestamp}_step_{global_step:07d}.ply'
    save_pointcloud_ply(pointcloud, rgb_np, valid_mask, vis_dir / pcl_name)

    # Save GT point cloud for this frame
    gt_pointcloud = aux['points_gt'][sample_idx, frame_idx]  # [H, W, 3]
    gt_pcl_name = f'{timestamp}_step_{global_step:07d}_gt.ply'
    save_pointcloud_ply(gt_pointcloud, rgb_np, valid_mask, vis_dir / gt_pcl_name)


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
            for i in range(min(aux["depth_pred"].shape[0], 1)):
                aux_accum["depth_pred_all"].append(aux["depth_pred"][i])
                aux_accum["depth_gt_all"].append(aux["depth_gt"][i])
                aux_accum["valid_mask_all"].append(aux["valid_mask"][i])
                
                # Save per-batch visualization
                save_training_visuals(cfg, views, aux, global_step * 1000 + batch_idx)
    
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

    data_loader_train = build_rgb_loader(cfg, split="train")
    data_loader_test = build_rgb_loader(cfg, split="test")
    train_samples_count = len(data_loader_train)
    test_samples_count = len(data_loader_test)
    printer.info("RGB train dataset: %d batches, test dataset: %d batches", train_samples_count, test_samples_count)

    # Use RGB version of StreamVGGT (no event_encoder)
    model = RGBStreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
    )

    if cfg.pretrained:
        printer.info("Loading model init weights from %s", cfg.pretrained)
        ckpt = unwrap_state_dict(torch.load(cfg.pretrained, map_location="cpu"))
        msg = model.load_state_dict(ckpt, strict=False)
        printer.info("Checkpoint load result: %s", msg)

    configure_trainable_params(model, cfg)
    log_trainable_params(model)

    criterion = SupervisedLoss(
        pose_weight=cfg.loss.pose_weight,
        depth_weight=cfg.loss.depth_weight,
        points_weight=cfg.loss.points_weight,
        normal_weight=float(getattr(cfg.loss, "normal_weight", 0.0)),
        depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
        depth_max=(float(cfg.loss.depth_max) if getattr(cfg.loss, "depth_max", None) is not None else None),
        align_depth_scale_enabled=bool(getattr(cfg.loss, "align_depth_scale", True)),
    )

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
    criterion = criterion.to(accelerator.device)

    log_writer = SummaryWriter(log_dir=cfg.logdir) if accelerator.is_main_process else None

    best_loss = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(cfg.start_epoch, cfg.epochs):
        model.train()
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"

        for data_iter_step, views in enumerate(metric_logger.log_every(data_loader_train, cfg.print_freq, accelerator, header)):
            with accelerator.accumulate(model):
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
    config_name="finetune_no_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    train(cfg)


if __name__ == "__main__":
    run()
