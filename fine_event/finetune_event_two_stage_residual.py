from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_two_stage import StreamVGGT as TwoStageStreamVGGT

_BASE_SAVE_TRAINING_VISUALS = fe.save_training_visuals


def _set_exp_paths(cfg, exp_name: str) -> None:
    cfg.exp_name = exp_name
    cfg.logdir = f"{cfg.save_dir}/{exp_name}/logs"
    cfg.output_dir = f"{cfg.save_dir}/{exp_name}"


def _stack_res_field(model_output, key: str) -> Optional[torch.Tensor]:
    if not getattr(model_output, "ress", None):
        return None
    if not all(key in res for res in model_output.ress):
        return None
    value = torch.stack([res[key] for res in model_output.ress], dim=1)
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    return value


def _edge_aware_residual_smoothness(
    residual: torch.Tensor,
    views: List[Dict[str, torch.Tensor]],
    valid_mask: torch.Tensor,
    edge_alpha: float = 10.0,
) -> torch.Tensor:
    if residual.ndim == 5 and residual.shape[-1] == 1:
        residual = residual.squeeze(-1)

    mask = valid_mask.to(device=residual.device).bool()
    dx = (residual[..., :, 1:] - residual[..., :, :-1]).abs()
    dy = (residual[..., 1:, :] - residual[..., :-1, :]).abs()
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]

    weight_x = torch.ones_like(dx)
    weight_y = torch.ones_like(dy)
    if views and all("img" in view for view in views):
        rgb = fe.stack_view_field(views, "img").to(device=residual.device, dtype=residual.dtype)
        if rgb.ndim == 5 and rgb.shape[2] in (1, 3):
            rgb_dx = (rgb[..., :, 1:] - rgb[..., :, :-1]).abs().mean(dim=2)
            rgb_dy = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(dim=2)
            weight_x = torch.exp(-float(edge_alpha) * rgb_dx).detach()
            weight_y = torch.exp(-float(edge_alpha) * rgb_dy).detach()

    mask_x = mask_x.to(dtype=residual.dtype)
    mask_y = mask_y.to(dtype=residual.dtype)
    weight_x = weight_x * mask_x
    weight_y = weight_y * mask_y

    loss_x = (dx * weight_x).sum() / weight_x.sum().clamp_min(1.0)
    loss_y = (dy * weight_y).sum() / weight_y.sum().clamp_min(1.0)
    return loss_x + loss_y


def _residual_second_order_smoothness(
    residual: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if residual.ndim == 5 and residual.shape[-1] == 1:
        residual = residual.squeeze(-1)

    mask = valid_mask.to(device=residual.device).bool()
    dxx = (residual[..., :, 2:] - 2.0 * residual[..., :, 1:-1] + residual[..., :, :-2]).abs()
    dyy = (residual[..., 2:, :] - 2.0 * residual[..., 1:-1, :] + residual[..., :-2, :]).abs()
    mask_x = mask[..., :, 2:] & mask[..., :, 1:-1] & mask[..., :, :-2]
    mask_y = mask[..., 2:, :] & mask[..., 1:-1, :] & mask[..., :-2, :]

    mask_x = mask_x.to(dtype=residual.dtype)
    mask_y = mask_y.to(dtype=residual.dtype)
    loss_x = (dxx * mask_x).sum() / mask_x.sum().clamp_min(1.0)
    loss_y = (dyy * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    return loss_x + loss_y


def _depth_normal_loss(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    normal_mask = valid_mask.clone().to(device=depth_pred.device).bool()
    normal_mask[..., 0, :] = False
    normal_mask[..., -1, :] = False
    normal_mask[..., :, 0] = False
    normal_mask[..., :, -1] = False
    pred_normals = fe.depth_to_normals(depth_pred, intrinsics)
    gt_normals = fe.depth_to_normals(depth_gt, intrinsics)
    return fe.masked_cosine_loss(pred_normals, gt_normals, normal_mask)


class TwoStageResidualEventSupervisedLoss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        residual_depth_weight: float = 0.0,
        coarse_depth_weight: float = 0.0,
        residual_smooth_weight: float = 0.0,
        residual_second_order_weight: float = 0.0,
        residual_abs_weight: float = 0.0,
        residual_smooth_alpha: float = 10.0,
        train_mode: str = "stage2",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.residual_depth_weight = float(residual_depth_weight)
        self.coarse_depth_weight = float(coarse_depth_weight)
        self.residual_smooth_weight = float(residual_smooth_weight)
        self.residual_second_order_weight = float(residual_second_order_weight)
        self.residual_abs_weight = float(residual_abs_weight)
        self.residual_smooth_alpha = float(residual_smooth_alpha)
        self.train_mode = str(train_mode).lower()

    def _forward_stage1(
        self,
        model_output,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        coarse_ress = []
        for res in model_output.ress:
            coarse_res = dict(res)
            if "depth_coarse" in res:
                coarse_res["depth"] = res["depth_coarse"]
            coarse_ress.append(coarse_res)

        total_loss, details, aux = super().forward(SimpleNamespace(ress=coarse_ress), views)

        depth_coarse = _stack_res_field(model_output, "depth_coarse")
        depth_residual = _stack_res_field(model_output, "depth_residual")
        if depth_coarse is not None:
            aux["depth_coarse"] = depth_coarse.detach()
        if depth_residual is not None:
            aux["depth_residual"] = depth_residual.detach()

        event_motion_density = getattr(model_output, "event_motion_density", None)
        if event_motion_density is not None:
            aux["event_motion_density"] = event_motion_density.detach()

        details["coarse_depth_loss"] = details.get("depth_loss", 0.0)
        details["residual_depth_loss"] = 0.0
        details["residual_smooth_loss"] = 0.0
        details["residual_second_order_loss"] = 0.0
        details["depth_residual_abs"] = 0.0
        return total_loss, details, aux

    def forward(
        self,
        model_output,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        if self.train_mode == "stage1":
            return self._forward_stage1(model_output, views)

        pred = model_output.ress
        depth_pred = torch.stack([res["depth"] for res in pred], dim=1).squeeze(-1)
        pose_pred = torch.stack([res["camera_pose"] for res in pred], dim=1)
        points_pred_head = torch.stack([res["pts3d_in_other_view"] for res in pred], dim=1)

        depth_gt = fe.stack_view_field(views, "depthmap").to(device=depth_pred.device, dtype=depth_pred.dtype)
        intrinsics_gt = fe.stack_view_field(views, "camera_intrinsics").to(
            device=depth_pred.device,
            dtype=depth_pred.dtype,
        )
        pose_matrix_gt = fe.stack_view_field(views, "camera_pose").to(
            device=depth_pred.device,
            dtype=depth_pred.dtype,
        )
        valid_mask = fe.build_valid_mask(
            views,
            depth_gt,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
        ).to(device=depth_pred.device)

        if self.align_depth_scale_enabled:
            depth_gt_aligned, depth_scales = fe.align_depth_scale(depth_pred.detach(), depth_gt, valid_mask)
        else:
            depth_gt_aligned = depth_gt
            depth_scales = depth_gt.new_ones(depth_gt.shape[:2])

        depth_loss = fe.masked_l1(depth_pred, depth_gt_aligned, valid_mask)
        points_gt = fe.depth_to_world_points(depth_gt_aligned, intrinsics_gt, pose_matrix_gt)
        points_pred_from_depth = fe.depth_to_world_points(depth_pred, intrinsics_gt, pose_matrix_gt)
        points_mask = valid_mask.unsqueeze(-1).expand_as(points_gt)
        if self.points_loss_type == "l1":
            points_loss = fe.masked_l1(points_pred_from_depth, points_gt, points_mask)
        else:
            points_loss = fe.masked_chamfer_distance(points_pred_from_depth, points_gt, valid_mask)

        height, width = depth_gt.shape[-2:]
        pose_gt = fe.camera_pose_to_pose_encoding(
            pose_matrix_gt,
            intrinsics_gt,
            image_size_hw=(height, width),
        ).to(device=pose_pred.device, dtype=pose_pred.dtype)
        if self.align_depth_scale_enabled:
            pose_gt[..., :3] = pose_gt[..., :3] * depth_scales.unsqueeze(-1)

        pred_c2w, pred_intrinsics = fe.pose_encoding_to_c2w(pose_pred, image_size_hw=(height, width))
        pred_c2w_aligned, first_frame_alignment = fe.align_c2w_by_first_frame(pred_c2w, pose_matrix_gt)
        pose_pred_aligned = fe.c2w_to_pose_encoding(
            pred_c2w_aligned,
            pred_intrinsics,
            image_size_hw=(height, width),
        ).to(device=pose_pred.device, dtype=pose_pred.dtype)
        if pose_pred_aligned.shape[1] > 1:
            pose_loss = F.smooth_l1_loss(pose_pred_aligned[:, 1:, :], pose_gt[:, 1:, :])
        else:
            pose_loss = pose_pred.new_tensor(0.0)
        points_pred_head_aligned = fe.transform_world_points(points_pred_head, first_frame_alignment)

        depth_coarse = _stack_res_field(model_output, "depth_coarse")
        depth_residual = _stack_res_field(model_output, "depth_residual")

        if depth_coarse is None or depth_residual is None:
            zero = depth_pred.new_tensor(0.0)
            coarse_depth_loss = zero
            residual_depth_loss = zero
            residual_smooth_loss = zero
            residual_second_order_loss = zero
            residual_abs = zero
            residual_target = None
        else:
            depth_coarse = depth_coarse.to(device=depth_pred.device, dtype=depth_pred.dtype)
            depth_residual = depth_residual.to(device=depth_pred.device, dtype=depth_pred.dtype)
            residual_target = depth_gt_aligned - depth_coarse.detach()
            coarse_depth_loss = fe.masked_l1(depth_coarse, depth_gt_aligned, valid_mask)
            residual_depth_loss = fe.masked_l1(depth_residual, residual_target, valid_mask)
            residual_smooth_loss = _edge_aware_residual_smoothness(
                depth_residual,
                views,
                valid_mask,
                edge_alpha=self.residual_smooth_alpha,
            )
            residual_second_order_loss = _residual_second_order_smoothness(depth_residual, valid_mask)
            residual_abs = fe.masked_l1(depth_residual, torch.zeros_like(depth_residual), valid_mask)

        if self.normal_weight > 0:
            normal_loss = _depth_normal_loss(depth_pred, depth_gt_aligned, intrinsics_gt, valid_mask)
        else:
            normal_loss = depth_pred.new_tensor(0.0)

        total_loss = (
            self.depth_weight * depth_loss
            + self.pose_weight * pose_loss
            + self.points_weight * points_loss
            + self.coarse_depth_weight * coarse_depth_loss
            + self.residual_depth_weight * residual_depth_loss
            + self.residual_smooth_weight * residual_smooth_loss
            + self.residual_second_order_weight * residual_second_order_loss
            + self.residual_abs_weight * residual_abs
            + self.normal_weight * normal_loss
        )

        details = {
            "pose_loss": float(pose_loss.detach()),
            "depth_loss": float(depth_loss.detach()),
            "points_loss": float(points_loss.detach()),
            "normal_loss": float(normal_loss.detach()),
            "depth_scale": float(depth_scales.mean().detach()),
            "coarse_depth_loss": float(coarse_depth_loss.detach()),
            "residual_depth_loss": float(residual_depth_loss.detach()),
            "residual_smooth_loss": float(residual_smooth_loss.detach()),
            "residual_second_order_loss": float(residual_second_order_loss.detach()),
            "depth_residual_abs": float(residual_abs.detach()),
        }

        aux = {
            "depth_pred": depth_pred.detach(),
            "depth_gt": depth_gt.detach(),
            "depth_gt_aligned": depth_gt_aligned.detach(),
            "points_pred": points_pred_from_depth.detach(),
            "points_pred_head": points_pred_head_aligned.detach(),
            "points_gt": points_gt.detach(),
            "valid_mask": valid_mask.detach(),
            "pose_pred": pose_pred_aligned.detach(),
            "pose_gt": pose_gt.detach(),
        }

        if depth_coarse is not None and depth_residual is not None:
            aux["depth_coarse"] = depth_coarse.detach()
            aux["depth_residual"] = depth_residual.detach()
        if residual_target is not None:
            aux["depth_residual_target"] = residual_target.detach()

        event_motion_density = getattr(model_output, "event_motion_density", None)
        if event_motion_density is not None:
            aux["event_motion_density"] = event_motion_density.detach()

        return total_loss, details, aux


def _signed_residual_to_uint8(value: torch.Tensor, mask: torch.Tensor) -> "fe.np.ndarray":
    value = value.detach().float().cpu()
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    mask = mask.detach().bool().cpu()
    valid = mask & torch.isfinite(value)
    image = torch.zeros(*value.shape[-2:], 3, dtype=torch.float32)
    if not valid.any():
        return image.numpy().astype("uint8")

    abs_values = value[valid].abs()
    if abs_values.numel() > 16:
        limit = torch.quantile(abs_values, 0.995)
    else:
        limit = abs_values.max()
    limit = limit.clamp_min(1e-6)
    normalized = (value / limit).clamp(-1.0, 1.0)
    positive = normalized.clamp_min(0.0)
    negative = (-normalized).clamp_min(0.0)

    image[..., 0] = positive
    image[..., 1] = 0.65 * (positive + negative).clamp_max(1.0)
    image[..., 2] = negative
    image[~mask] = 0.0
    return (image.clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")


def _event_stream_to_uint8(view: Dict[str, torch.Tensor], sample_idx: int, height: int, width: int) -> "fe.np.ndarray":
    image = torch.zeros(height, width, 3, dtype=torch.float32)
    event_xy = view.get("event_xy")
    event_p = view.get("event_p")
    if not isinstance(event_xy, (list, tuple)) or not isinstance(event_p, (list, tuple)):
        return image.numpy().astype("uint8")
    if sample_idx >= len(event_xy) or sample_idx >= len(event_p):
        return image.numpy().astype("uint8")

    xy = event_xy[sample_idx]
    polarity = event_p[sample_idx]
    if xy is None or polarity is None or xy.numel() == 0:
        return image.numpy().astype("uint8")

    xy = xy.detach().long().cpu()
    polarity = polarity.detach().float().cpu()
    x = xy[:, 0]
    y = xy[:, 1]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not valid.any():
        return image.numpy().astype("uint8")

    x = x[valid]
    y = y[valid]
    polarity = polarity[valid]
    flat_idx = y * width + x
    pos = torch.zeros(height * width, dtype=torch.float32)
    neg = torch.zeros(height * width, dtype=torch.float32)
    pos_mask = polarity > 0
    neg_mask = ~pos_mask
    if pos_mask.any():
        pos.index_add_(0, flat_idx[pos_mask], torch.ones_like(flat_idx[pos_mask], dtype=torch.float32))
    if neg_mask.any():
        neg.index_add_(0, flat_idx[neg_mask], torch.ones_like(flat_idx[neg_mask], dtype=torch.float32))

    pos = pos.view(height, width)
    neg = neg.view(height, width)
    max_count = torch.stack([pos.max(), neg.max()]).max().clamp_min(1.0)
    denom = torch.log1p(max_count)
    pos = torch.log1p(pos) / denom
    neg = torch.log1p(neg) / denom
    image[..., 0] = pos
    image[..., 1] = 0.35 * (pos + neg).clamp_max(1.0)
    image[..., 2] = neg
    return (image.clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")


def _concat_horizontal(images):
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = fe.Image.new("RGB", (width, height), color=(0, 0, 0))
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def _concat_vertical(images):
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = fe.Image.new("RGB", (width, height), color=(0, 0, 0))
    y_offset = 0
    for image in images:
        canvas.paste(image, (0, y_offset))
        y_offset += image.height
    return canvas


def save_two_stage_training_visuals(
    cfg,
    views: List[Dict[str, torch.Tensor]],
    aux: Dict[str, torch.Tensor],
    global_step: int,
    frame_idx: Optional[int] = None,
    sample_idx: Optional[int] = None,
) -> None:
    _BASE_SAVE_TRAINING_VISUALS(cfg, views, aux, global_step, frame_idx, sample_idx)

    vis_cfg = getattr(cfg, "vis", None)
    save_every = getattr(vis_cfg, "save_every_steps", 200)
    if save_every <= 0 or global_step % save_every != 0:
        return
    if "depth_residual" not in aux or "depth_residual_target" not in aux:
        return

    if sample_idx is None:
        sample_idx = int(getattr(vis_cfg, "sample_index", 0))
    if sample_idx >= aux["depth_residual"].shape[0]:
        return

    if frame_idx is not None:
        frames_to_save = [int(frame_idx)]
    else:
        max_views = int(getattr(vis_cfg, "residual_num_views", len(views)))
        frames_to_save = list(range(min(len(views), max_views)))

    valid_mask = aux["valid_mask"]
    pred_residual = aux["depth_residual"]
    target_residual = aux["depth_residual_target"]
    rows = [[], [], []]

    for frame_id in frames_to_save:
        mask = valid_mask[sample_idx, frame_id]
        height, width = mask.shape[-2:]
        rows[0].append(
            fe.make_labeled_panel(
                f"pred_residual_f{frame_id:02d}",
                _signed_residual_to_uint8(pred_residual[sample_idx, frame_id], mask),
            )
        )
        rows[1].append(
            fe.make_labeled_panel(
                f"events_f{frame_id:02d}",
                _event_stream_to_uint8(views[frame_id], sample_idx, height, width),
            )
        )
        rows[2].append(
            fe.make_labeled_panel(
                f"gt_residual_f{frame_id:02d}",
                _signed_residual_to_uint8(target_residual[sample_idx, frame_id], mask),
            )
        )

    if not rows[0]:
        return

    grid = _concat_vertical([_concat_horizontal(row) for row in rows])
    vis_dir = Path(cfg.output_dir) / "train_vis_residual"
    vis_dir.mkdir(parents=True, exist_ok=True)
    timestamp = fe.datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_frame_{frame_idx:02d}" if frame_idx is not None else ""
    image_name = f"{timestamp}_step_{global_step:07d}_sample_{sample_idx}{suffix}_residual.png"
    grid.save(vis_dir / image_name)


def configure_two_stage_trainable_params(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    train_mode = str(getattr(cfg.train, "two_stage_train_mode", "stage2")).lower()
    if train_mode == "stage1":
        if cfg.train.unfreeze_heads:
            for module in (model.camera_head, model.depth_head, model.point_head, model.track_head):
                for param in module.parameters():
                    param.requires_grad = True

        if cfg.train.unfreeze_aggregator_blocks:
            for param in model.aggregator.frame_blocks.parameters():
                param.requires_grad = True
            for param in model.aggregator.global_blocks.parameters():
                param.requires_grad = True

        last_blocks = int(getattr(cfg.train, "unfreeze_aggregator_last_blocks", 0))
        if last_blocks > 0:
            for block in model.aggregator.frame_blocks[-last_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            for block in model.aggregator.global_blocks[-last_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        return

    if train_mode != "stage2":
        raise ValueError(f"Unknown two_stage_train_mode={train_mode!r}; expected 'stage1' or 'stage2'")

    for name, param in model.named_parameters():
        if name.startswith(("event_encoder", "event_residual_refiner")):
            param.requires_grad = True


def build_two_stage_optimizer_params(model, cfg):
    base_lr = float(cfg.lr)
    vggt_lr = float(getattr(cfg.train, "vggt_last_blocks_lr", base_lr))
    groups = {"event": [], "heads": [], "vggt": [], "other": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("event_encoder", "event_residual_refiner")):
            groups["event"].append(param)
        elif name.startswith("aggregator"):
            groups["vggt"].append(param)
        elif name.startswith(("camera_head", "depth_head", "point_head", "track_head")):
            groups["heads"].append(param)
        else:
            groups["other"].append(param)

    param_groups = []
    for key in ("event", "heads", "other"):
        if groups[key]:
            param_groups.append({"params": groups[key], "lr_scale": 1.0})
    if groups["vggt"]:
        param_groups.append({"params": groups["vggt"], "lr_scale": vggt_lr / max(base_lr, 1e-12)})
    return param_groups


def launch(
    cfg,
    *,
    exp_name: Optional[str] = None,
    residual_input_mode: Optional[str] = None,
    train_mode: Optional[str] = None,
) -> None:
    if residual_input_mode is not None:
        cfg.model.residual_input_mode = residual_input_mode
    if train_mode is not None:
        cfg.train.two_stage_train_mode = train_mode
    if exp_name is not None:
        _set_exp_paths(cfg, exp_name)

    OmegaConf.resolve(cfg)

    class ConfiguredTwoStageStreamVGGT(TwoStageStreamVGGT):
        def __init__(self, *args, **kwargs):
            kwargs.pop("event_hidden_dim", None)
            kwargs.pop("head_frames_chunk_size", None)
            super().__init__(
                *args,
                event_hidden_dim=int(getattr(cfg.model, "event_hidden_dim", 32)),
                event_num_bins=int(getattr(cfg.model, "event_num_bins", 8)),
                event_count_cmax=float(getattr(cfg.model, "event_count_cmax", 3.0)),
                event_encode_downsample=int(getattr(cfg.model, "event_encode_downsample", getattr(cfg.model, "event_downsample", 1))),
                head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 8)),
                residual_hidden_dim=int(getattr(cfg.model, "residual_hidden_dim", 96)),
                event_downsample=int(getattr(cfg.model, "event_downsample", 1)),
                residual_scale=float(getattr(cfg.model, "residual_scale", 0.1)),
                residual_activation=str(getattr(cfg.model, "residual_activation", "tanh")),
                event_backbone=str(getattr(cfg.model, "event_backbone", "unet")),
                residual_input_mode=str(getattr(cfg.model, "residual_input_mode", "current_event")),
                disable_second_stage=bool(getattr(cfg.model, "disable_second_stage", False)),
                **kwargs,
            )

    class ConfiguredTwoStageLoss(TwoStageResidualEventSupervisedLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                residual_depth_weight=float(getattr(cfg.loss, "residual_depth_weight", 0.0)),
                coarse_depth_weight=float(getattr(cfg.loss, "coarse_depth_weight", 0.0)),
                residual_smooth_weight=float(getattr(cfg.loss, "residual_smooth_weight", 0.0)),
                residual_second_order_weight=float(getattr(cfg.loss, "residual_second_order_weight", 0.0)),
                residual_abs_weight=float(getattr(cfg.loss, "residual_abs_weight", 0.0)),
                residual_smooth_alpha=float(getattr(cfg.loss, "residual_smooth_alpha", 10.0)),
                train_mode=str(getattr(cfg.train, "two_stage_train_mode", "stage2")),
                **kwargs,
            )

    fe.EventStreamVGGT = ConfiguredTwoStageStreamVGGT
    fe.EventSupervisedLoss = ConfiguredTwoStageLoss
    fe.configure_trainable_params = configure_two_stage_trainable_params
    fe.build_optimizer_params = build_two_stage_optimizer_params
    fe.save_training_visuals = save_two_stage_training_visuals
    fe.train(cfg)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event_two_stage_residual.yaml",
)
def run(cfg: OmegaConf):
    launch(cfg)


if __name__ == "__main__":
    run()
