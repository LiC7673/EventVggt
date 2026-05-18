from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import finetune_event as fe


def _as_weight_map(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return mask.to(dtype=dtype)


def _make_event_support(
    views: List[Dict[str, torch.Tensor]],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    blur_kernel: int = 5,
    dilate_kernel: int = 3,
    threshold: float = 0.02,
    power: float = 1.0,
) -> torch.Tensor:
    if "event_voxel" not in views[0]:
        batch = views[0]["img"].shape[0]
        return torch.zeros((batch, len(views), height, width), device=device, dtype=dtype)

    voxels = fe.stack_view_field(views, "event_voxel").to(device=device, dtype=dtype)
    if voxels.ndim != 5 or voxels.shape[2] == 0:
        batch = views[0]["img"].shape[0]
        return torch.zeros((batch, len(views), height, width), device=device, dtype=dtype)

    support = torch.log1p(voxels.abs().sum(dim=2))
    support = support.flatten(2)
    support = support / support.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    support = support.view(voxels.shape[0], voxels.shape[1], voxels.shape[-2], voxels.shape[-1])

    if support.shape[-2:] != (height, width):
        support = F.interpolate(
            support.flatten(0, 1).unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).view(voxels.shape[0], voxels.shape[1], height, width)

    if threshold > 0:
        support = (support - float(threshold)).clamp_min(0.0) / max(1.0 - float(threshold), 1e-6)
    if dilate_kernel and dilate_kernel > 1:
        pad = int(dilate_kernel) // 2
        support = F.max_pool2d(
            support.flatten(0, 1).unsqueeze(1),
            kernel_size=int(dilate_kernel),
            stride=1,
            padding=pad,
        ).squeeze(1).view_as(support)
    if blur_kernel and blur_kernel > 1:
        pad = int(blur_kernel) // 2
        support = F.avg_pool2d(
            support.flatten(0, 1).unsqueeze(1),
            kernel_size=int(blur_kernel),
            stride=1,
            padding=pad,
        ).squeeze(1).view_as(support)
    if power != 1.0:
        support = support.clamp_min(0.0).pow(float(power))
    return support.clamp(0.0, 1.0)


def _normal_gradient_magnitude(normals: torch.Tensor) -> torch.Tensor:
    # normals: [B, H, W, 3], return [B, 1, H, W]
    n = normals.permute(0, 3, 1, 2)
    dx = F.pad(n[..., :, 1:] - n[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(n[..., 1:, :] - n[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt((dx.square() + dy.square()).sum(dim=1, keepdim=True).clamp_min(1e-12))


def _high_frequency_normals(normals: torch.Tensor, kernel: int = 7) -> torch.Tensor:
    n = normals.permute(0, 3, 1, 2)
    kernel = max(int(kernel), 1)
    if kernel <= 1:
        return n
    pad = kernel // 2
    blurred = F.avg_pool2d(n, kernel_size=kernel, stride=1, padding=pad)
    return n - blurred


def _gradient_orientation(x: torch.Tensor) -> torch.Tensor:
    # x: [B, 1, H, W], return [B, 2, H, W]
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    orient = torch.cat([dx, dy], dim=1)
    return F.normalize(orient, dim=1, eps=1e-6)


def _sample_grid(map_bchw: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    return F.grid_sample(
        map_bchw,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )


def _project_i_to_j(
    depth_i: torch.Tensor,
    intr_i: torch.Tensor,
    intr_j: torch.Tensor,
    c2w_i: torch.Tensor,
    c2w_j: torch.Tensor,
    valid_i: torch.Tensor,
    *,
    detach_grid: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if detach_grid:
        depth_i = depth_i.detach()
        intr_i = intr_i.detach()
        intr_j = intr_j.detach()
        c2w_i = c2w_i.detach()
        c2w_j = c2w_j.detach()

    batch, height, width = depth_i.shape
    device = depth_i.device
    dtype = depth_i.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.view(1, height, width).expand(batch, -1, -1)
    ys = ys.view(1, height, width).expand(batch, -1, -1)

    fx_i = intr_i[:, 0, 0].view(batch, 1, 1).clamp_min(eps)
    fy_i = intr_i[:, 1, 1].view(batch, 1, 1).clamp_min(eps)
    cx_i = intr_i[:, 0, 2].view(batch, 1, 1)
    cy_i = intr_i[:, 1, 2].view(batch, 1, 1)
    z_i = depth_i.clamp_min(eps)
    x_i = (xs - cx_i) * z_i / fx_i
    y_i = (ys - cy_i) * z_i / fy_i
    cam_i = torch.stack([x_i, y_i, z_i], dim=-1)

    rot_i = c2w_i[:, :3, :3]
    trans_i = c2w_i[:, :3, 3]
    world = torch.einsum("bij,bhwj->bhwi", rot_i, cam_i) + trans_i[:, None, None, :]

    rot_j = c2w_j[:, :3, :3]
    trans_j = c2w_j[:, :3, 3]
    cam_j = torch.einsum("bij,bhwj->bhwi", rot_j.transpose(-1, -2), world - trans_j[:, None, None, :])
    z_j = cam_j[..., 2]

    fx_j = intr_j[:, 0, 0].view(batch, 1, 1)
    fy_j = intr_j[:, 1, 1].view(batch, 1, 1)
    cx_j = intr_j[:, 0, 2].view(batch, 1, 1)
    cy_j = intr_j[:, 1, 2].view(batch, 1, 1)
    u = fx_j * cam_j[..., 0] / z_j.clamp_min(eps) + cx_j
    v = fy_j * cam_j[..., 1] / z_j.clamp_min(eps) + cy_j

    valid = (
        valid_i
        & torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(z_j)
        & (z_j > eps)
        & (u >= 0)
        & (u <= width - 1)
        & (v >= 0)
        & (v <= height - 1)
    )
    grid_x = 2.0 * u / max(width - 1, 1) - 1.0
    grid_y = 2.0 * v / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    if detach_grid:
        grid = grid.detach()
        valid = valid.detach()
    return grid, valid


def _pair_indices(seq_len: int, *, bidirectional: bool, max_pairs: int) -> Sequence[Tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(max(seq_len - 1, 0))]
    if bidirectional:
        pairs = pairs + [(j, i) for i, j in pairs]
    if max_pairs and max_pairs > 0:
        pairs = pairs[: int(max_pairs)]
    return pairs


class EventSupportedMultiViewLoss(nn.Module):
    def __init__(
        self,
        *,
        normal_weight: float = 0.0,
        presence_weight: float = 0.0,
        hf_weight: float = 0.0,
        orient_weight: float = 0.0,
        presence_margin: float = 0.04,
        event_blur_kernel: int = 5,
        event_dilate_kernel: int = 3,
        event_threshold: float = 0.02,
        event_power: float = 1.0,
        hf_kernel: int = 7,
        bidirectional: bool = False,
        max_pairs: int = 4,
        detach_warp_grid: bool = True,
        projection_pose: str = "gt",
    ):
        super().__init__()
        self.normal_weight = float(normal_weight)
        self.presence_weight = float(presence_weight)
        self.hf_weight = float(hf_weight)
        self.orient_weight = float(orient_weight)
        self.presence_margin = float(presence_margin)
        self.event_blur_kernel = int(event_blur_kernel)
        self.event_dilate_kernel = int(event_dilate_kernel)
        self.event_threshold = float(event_threshold)
        self.event_power = float(event_power)
        self.hf_kernel = int(hf_kernel)
        self.bidirectional = bool(bidirectional)
        self.max_pairs = int(max_pairs)
        self.detach_warp_grid = bool(detach_warp_grid)
        self.projection_pose = str(projection_pose).lower()

    @property
    def enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (self.normal_weight, self.presence_weight, self.hf_weight, self.orient_weight)
        )

    def forward(
        self,
        *,
        depth_pred: torch.Tensor,
        pred_normals: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w_for_projection: torch.Tensor,
        valid_mask: torch.Tensor,
        views: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.enabled or depth_pred.shape[1] <= 1:
            zero = depth_pred.new_tensor(0.0)
            return zero, {
                "mv_event_loss": 0.0,
                "mv_event_normal_loss": 0.0,
                "mv_event_presence_loss": 0.0,
                "mv_event_hf_loss": 0.0,
                "mv_event_orient_loss": 0.0,
                "mv_event_weight_mean": 0.0,
            }

        batch, seq_len, height, width = depth_pred.shape
        event_support = _make_event_support(
            views,
            height=height,
            width=width,
            device=depth_pred.device,
            dtype=depth_pred.dtype,
            blur_kernel=self.event_blur_kernel,
            dilate_kernel=self.event_dilate_kernel,
            threshold=self.event_threshold,
            power=self.event_power,
        ).detach()

        normal_grad = _normal_gradient_magnitude(pred_normals.flatten(0, 1)).view(batch, seq_len, 1, height, width)
        hf_normals = _high_frequency_normals(pred_normals.flatten(0, 1), kernel=self.hf_kernel).view(
            batch, seq_len, 3, height, width
        )
        orient = _gradient_orientation(normal_grad.flatten(0, 1)).view(batch, seq_len, 2, height, width)

        normal_terms = []
        presence_terms = []
        hf_terms = []
        orient_terms = []
        weight_sums = []

        for i, j in _pair_indices(seq_len, bidirectional=self.bidirectional, max_pairs=self.max_pairs):
            grid, valid = _project_i_to_j(
                depth_pred[:, i],
                intrinsics[:, i],
                intrinsics[:, j],
                c2w_for_projection[:, i],
                c2w_for_projection[:, j],
                valid_mask[:, i],
                detach_grid=self.detach_warp_grid,
            )

            s_i = event_support[:, i].unsqueeze(1)
            s_j = event_support[:, j].unsqueeze(1)
            s_j_warp = _sample_grid(s_j, grid)
            valid_w = _as_weight_map(valid.unsqueeze(1), depth_pred.dtype)
            weight = (s_i * s_j_warp * valid_w).detach()
            denom = weight.sum().clamp_min(1.0)
            weight_sums.append(weight.mean())

            n_i = pred_normals[:, i].permute(0, 3, 1, 2)
            n_j = pred_normals[:, j].permute(0, 3, 1, 2)
            n_j_warp = _sample_grid(n_j, grid)

            rot_i = c2w_for_projection[:, i, :3, :3]
            rot_j = c2w_for_projection[:, j, :3, :3]
            rot_j_to_i = torch.matmul(rot_i.transpose(-1, -2), rot_j).to(dtype=depth_pred.dtype)
            n_j_to_i = torch.einsum("bij,bjhw->bihw", rot_j_to_i, n_j_warp)
            n_i_norm = F.normalize(n_i, dim=1, eps=1e-6)
            n_j_to_i = F.normalize(n_j_to_i, dim=1, eps=1e-6)

            if self.normal_weight > 0:
                cos = (n_i_norm * n_j_to_i).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                normal_terms.append(((1.0 - cos) * weight).sum() / denom)

            if self.presence_weight > 0:
                gn_i = normal_grad[:, i]
                gn_j = normal_grad[:, j]
                gn_j_warp = _sample_grid(gn_j, grid)
                presence = F.relu(self.presence_margin - gn_i) + F.relu(self.presence_margin - gn_j_warp)
                presence_terms.append((presence * weight).sum() / denom)

            if self.hf_weight > 0:
                hf_i = hf_normals[:, i]
                hf_j = hf_normals[:, j]
                hf_j_warp = _sample_grid(hf_j, grid)
                hf_terms.append(((hf_i - hf_j_warp).abs().mean(dim=1, keepdim=True) * weight).sum() / denom)

            if self.orient_weight > 0:
                oi = orient[:, i]
                oj = orient[:, j]
                oj_warp = _sample_grid(oj, grid)
                align = (oi * oj_warp).sum(dim=1, keepdim=True).abs().clamp(0.0, 1.0)
                orient_terms.append(((1.0 - align) * weight).sum() / denom)

        if not weight_sums:
            zero = depth_pred.new_tensor(0.0)
            return zero, {
                "mv_event_loss": 0.0,
                "mv_event_normal_loss": 0.0,
                "mv_event_presence_loss": 0.0,
                "mv_event_hf_loss": 0.0,
                "mv_event_orient_loss": 0.0,
                "mv_event_weight_mean": 0.0,
            }

        def mean_or_zero(values: Sequence[torch.Tensor]) -> torch.Tensor:
            if not values:
                return depth_pred.new_tensor(0.0)
            return torch.stack(list(values)).mean()

        normal_loss = mean_or_zero(normal_terms)
        presence_loss = mean_or_zero(presence_terms)
        hf_loss = mean_or_zero(hf_terms)
        orient_loss = mean_or_zero(orient_terms)
        total = (
            self.normal_weight * normal_loss
            + self.presence_weight * presence_loss
            + self.hf_weight * hf_loss
            + self.orient_weight * orient_loss
        )

        details = {
            "mv_event_loss": float(total.detach()),
            "mv_event_normal_loss": float(normal_loss.detach()),
            "mv_event_presence_loss": float(presence_loss.detach()),
            "mv_event_hf_loss": float(hf_loss.detach()),
            "mv_event_orient_loss": float(orient_loss.detach()),
            "mv_event_weight_mean": float(torch.stack(weight_sums).mean().detach()),
        }
        return total, details


class MultiViewEventSupervisedLoss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        mv_normal_weight: float = 0.0,
        mv_presence_weight: float = 0.0,
        mv_hf_weight: float = 0.0,
        mv_orient_weight: float = 0.0,
        mv_presence_margin: float = 0.04,
        mv_event_blur_kernel: int = 5,
        mv_event_dilate_kernel: int = 3,
        mv_event_threshold: float = 0.02,
        mv_event_power: float = 1.0,
        mv_hf_kernel: int = 7,
        mv_bidirectional: bool = False,
        mv_max_pairs: int = 4,
        mv_detach_warp_grid: bool = True,
        mv_projection_pose: str = "gt",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mv_loss = EventSupportedMultiViewLoss(
            normal_weight=mv_normal_weight,
            presence_weight=mv_presence_weight,
            hf_weight=mv_hf_weight,
            orient_weight=mv_orient_weight,
            presence_margin=mv_presence_margin,
            event_blur_kernel=mv_event_blur_kernel,
            event_dilate_kernel=mv_event_dilate_kernel,
            event_threshold=mv_event_threshold,
            event_power=mv_event_power,
            hf_kernel=mv_hf_kernel,
            bidirectional=mv_bidirectional,
            max_pairs=mv_max_pairs,
            detach_warp_grid=mv_detach_warp_grid,
            projection_pose=mv_projection_pose,
        )

    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]):
        base_loss, details, aux = super().forward(model_output, views)
        if not self.mv_loss.enabled:
            details.update(
                {
                    "mv_event_loss": 0.0,
                    "mv_event_normal_loss": 0.0,
                    "mv_event_presence_loss": 0.0,
                    "mv_event_hf_loss": 0.0,
                    "mv_event_orient_loss": 0.0,
                    "mv_event_weight_mean": 0.0,
                }
            )
            return base_loss, details, aux

        pred = model_output.ress
        depth_pred = torch.stack([res["depth"] for res in pred], dim=1).squeeze(-1)
        pose_pred = torch.stack([res["camera_pose"] for res in pred], dim=1)
        intrinsics_gt = fe.stack_view_field(views, "camera_intrinsics").to(
            device=depth_pred.device,
            dtype=depth_pred.dtype,
        )
        pose_matrix_gt = fe.stack_view_field(views, "camera_pose").to(device=depth_pred.device, dtype=depth_pred.dtype)
        depth_gt = fe.stack_view_field(views, "depthmap").to(device=depth_pred.device, dtype=depth_pred.dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)
        height, width = depth_gt.shape[-2:]
        pred_normals = fe.depth_to_normals(depth_pred.clamp_min(self.depth_min), intrinsics_gt)

        if self.mv_loss.projection_pose == "pred":
            pred_c2w, _ = fe.pose_encoding_to_c2w(pose_pred, image_size_hw=(height, width))
            c2w_for_projection, _ = fe.align_c2w_by_first_frame(pred_c2w, pose_matrix_gt)
        else:
            c2w_for_projection = pose_matrix_gt

        mv_loss, mv_details = self.mv_loss(
            depth_pred=depth_pred,
            pred_normals=pred_normals,
            intrinsics=intrinsics_gt,
            c2w_for_projection=c2w_for_projection,
            valid_mask=valid_mask,
            views=views,
        )
        details.update(mv_details)
        return base_loss + mv_loss, details, aux
