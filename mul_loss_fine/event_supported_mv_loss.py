from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import finetune_event as fe
from mul_loss_fine.antigrid_loss import final_depth_patch_phase_antigrid_loss


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
    top_fraction: float = 0.0,
    mode: str = "abs",
) -> torch.Tensor:
    if "event_voxel" not in views[0]:
        batch = views[0]["img"].shape[0]
        return torch.zeros((batch, len(views), height, width), device=device, dtype=dtype)

    voxels = fe.stack_view_field(views, "event_voxel").to(device=device, dtype=dtype)
    if voxels.ndim != 5 or voxels.shape[2] == 0:
        batch = views[0]["img"].shape[0]
        return torch.zeros((batch, len(views), height, width), device=device, dtype=dtype)

    mode = str(mode or "abs").lower()
    channels = voxels.shape[2]
    num_bins = max(channels // 2, 1)
    pos = voxels[:, :, :num_bins].clamp_min(0.0)
    neg = voxels[:, :, num_bins : 2 * num_bins].clamp_min(0.0)
    activity = pos + neg
    activity_sum = activity.sum(dim=2)

    support = torch.log1p(activity_sum)
    if mode in {"temporal", "temporal_contrast", "bin", "bin_aware", "temporal_polarity", "polarity"}:
        # Persistent highlight/noise often fires across many bins at the same pixel.
        # True swept edges tend to be more temporally concentrated at each pixel.
        temporal_peak = activity.amax(dim=2) / activity_sum.clamp_min(1e-6)
        support = support * (0.5 + 0.5 * temporal_peak)

    if mode in {"polarity", "signed", "temporal_polarity", "polarity_aware"}:
        pos_sum = pos.sum(dim=2)
        neg_sum = neg.sum(dim=2)
        polarity_conf = (pos_sum - neg_sum).abs() / (pos_sum + neg_sum).clamp_min(1e-6)
        support = support * (0.5 + 0.5 * polarity_conf)

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
    if 0.0 < float(top_fraction) < 1.0:
        # The event stream can be almost dense; retain only its strongest
        # support so event weighting targets difficult local detail.
        flat = support.flatten(2)
        keep = max(1, min(flat.shape[-1], int(round(flat.shape[-1] * float(top_fraction)))))
        _, top_indices = torch.topk(flat, k=keep, dim=-1, sorted=False)
        top_mask = torch.zeros_like(flat).scatter_(-1, top_indices, 1.0)
        support = (flat * top_mask).view_as(support)
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


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def _stack_output_field(model_output, key: str) -> Optional[torch.Tensor]:
    ress = getattr(model_output, "ress", None)
    if not ress or not all(key in res for res in ress):
        return None
    value = torch.stack([res[key] for res in ress], dim=1)
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    return value


def _residual_edge_aware_smoothness(
    residual: torch.Tensor,
    views: List[Dict[str, torch.Tensor]],
    valid_mask: torch.Tensor,
    edge_alpha: float,
) -> torch.Tensor:
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    mask = valid_mask.to(device=residual.device).bool()
    dx = (residual[..., :, 1:] - residual[..., :, :-1]).abs()
    dy = (residual[..., 1:, :] - residual[..., :-1, :]).abs()
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]
    weight_x = torch.ones_like(dx)
    weight_y = torch.ones_like(dy)
    if views and all("img" in view for view in views):
        rgb = fe.stack_view_field(views, "img").to(device=residual.device, dtype=residual.dtype)
        rgb_dx = (rgb[..., :, 1:] - rgb[..., :, :-1]).abs().mean(dim=2)
        rgb_dy = (rgb[..., 1:, :] - rgb[..., :-1, :]).abs().mean(dim=2)
        weight_x = torch.exp(-float(edge_alpha) * rgb_dx).detach()
        weight_y = torch.exp(-float(edge_alpha) * rgb_dy).detach()
    weight_x = weight_x * mask_x.to(dtype=residual.dtype)
    weight_y = weight_y * mask_y.to(dtype=residual.dtype)
    return (dx * weight_x).sum() / weight_x.sum().clamp_min(1.0) + (
        dy * weight_y
    ).sum() / weight_y.sum().clamp_min(1.0)


def _residual_second_order_smoothness(residual: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    mask = valid_mask.to(device=residual.device).bool()
    dxx = (residual[..., :, 2:] - 2.0 * residual[..., :, 1:-1] + residual[..., :, :-2]).abs()
    dyy = (residual[..., 2:, :] - 2.0 * residual[..., 1:-1, :] + residual[..., :-2, :]).abs()
    mask_x = mask[..., :, 2:] & mask[..., :, 1:-1] & mask[..., :, :-2]
    mask_y = mask[..., 2:, :] & mask[..., 1:-1, :] & mask[..., :-2, :]
    mask_x = mask_x.to(dtype=residual.dtype)
    mask_y = mask_y.to(dtype=residual.dtype)
    return (dxx * mask_x).sum() / mask_x.sum().clamp_min(1.0) + (
        dyy * mask_y
    ).sum() / mask_y.sum().clamp_min(1.0)


def _normalize_detail_support(
    detail: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float = 0.03,
    power: float = 1.0,
) -> torch.Tensor:
    # detail: [B, S, 1, H, W], valid_mask: [B, S, H, W]
    valid = valid_mask.unsqueeze(2).to(dtype=detail.dtype)
    masked = detail.detach().clamp_min(0.0) * valid
    flat = masked.flatten(2)
    robust = torch.quantile(flat.float(), 0.95, dim=-1).to(dtype=detail.dtype)
    maximum = flat.amax(dim=-1)
    scale = torch.where(robust > 1e-6, robust, maximum).view(
        detail.shape[0], detail.shape[1], 1, 1, 1
    ).clamp_min(1e-6)
    support = (masked / scale - float(threshold)).clamp_min(0.0) / max(1.0 - float(threshold), 1e-6)
    if power != 1.0:
        support = support.pow(float(power))
    return support.clamp(0.0, 1.0) * valid


def _normalize_detail_support_flat(
    detail: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float = 0.03,
    power: float = 1.0,
) -> torch.Tensor:
    # detail: [N, 1, H, W], valid_mask: [N, H, W]
    valid = valid_mask.unsqueeze(1).to(dtype=detail.dtype)
    masked = detail.detach().clamp_min(0.0) * valid
    flat = masked.flatten(2)
    robust = torch.quantile(flat.float(), 0.95, dim=-1).to(dtype=detail.dtype)
    maximum = flat.amax(dim=-1)
    scale = torch.where(robust > 1e-6, robust, maximum).view(
        detail.shape[0], 1, 1, 1
    ).clamp_min(1e-6)
    support = (masked / scale - float(threshold)).clamp_min(0.0) / max(1.0 - float(threshold), 1e-6)
    if power != 1.0:
        support = support.pow(float(power))
    return support.clamp(0.0, 1.0) * valid


def _gt_normals_from_views(
    views: List[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    key = None
    if all("normal" in view for view in views):
        key = "normal"
    elif all("normal_gt" in view for view in views):
        key = "normal_gt"
    if key is None:
        return None

    normals = fe.stack_view_field(views, key).to(device=device, dtype=dtype)
    if normals.ndim != 5:
        return None
    if normals.shape[2] == 3:
        normals = normals.permute(0, 1, 3, 4, 2)
    elif normals.shape[-1] != 3:
        return None
    if normals.detach().abs().amax() > 2.0:
        normals = normals / 127.5 - 1.0
    return F.normalize(normals, dim=-1, eps=1e-6)


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
        event_top_fraction: float = 0.0,
        event_support_mode: str = "abs",
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
        self.event_top_fraction = float(event_top_fraction)
        self.event_support_mode = str(event_support_mode)
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
            top_fraction=self.event_top_fraction,
            mode=self.event_support_mode,
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
        mv_event_top_fraction: float = 0.0,
        mv_event_support_mode: str = "abs",
        mv_hf_kernel: int = 7,
        mv_bidirectional: bool = False,
        mv_max_pairs: int = 4,
        mv_detach_warp_grid: bool = True,
        mv_projection_pose: str = "gt",
        detail_gt_normal_weight: float = 0.0,
        detail_gt_hf_weight: float = 0.0,
        detail_gt_grad_weight: float = 0.0,
        detail_gt_event_boost: float = 0.5,
        detail_gt_threshold: float = 0.03,
        detail_gt_weight_power: float = 1.0,
        detail_gt_normal_source: str = "auto",
        detail_gt_salient_hf_weight: float = 0.0,
        detail_gt_salient_mag_weight: float = 0.0,
        detail_gt_salient_presence_weight: float = 0.0,
        detail_gt_salient_threshold: float = 0.35,
        detail_gt_salient_power: float = 2.0,
        detail_gt_salient_presence_ratio: float = 0.8,
        detail_gt_chunk_size: int = 1,
        residual_smooth_weight: float = 0.0,
        residual_second_order_weight: float = 0.0,
        residual_abs_weight: float = 0.0,
        residual_smooth_alpha: float = 10.0,
        final_grid_weight: float = 0.0,
        final_phase_weight: float = 0.0,
        final_grid_patch_size: int = 14,
        final_grid_band: int = 1,
        final_grid_detail_threshold: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detail_gt_normal_weight = float(detail_gt_normal_weight)
        self.detail_gt_hf_weight = float(detail_gt_hf_weight)
        self.detail_gt_grad_weight = float(detail_gt_grad_weight)
        self.detail_gt_event_boost = float(detail_gt_event_boost)
        self.detail_gt_threshold = float(detail_gt_threshold)
        self.detail_gt_weight_power = float(detail_gt_weight_power)
        self.detail_gt_normal_source = str(detail_gt_normal_source).lower()
        self.detail_gt_salient_hf_weight = float(detail_gt_salient_hf_weight)
        self.detail_gt_salient_mag_weight = float(detail_gt_salient_mag_weight)
        self.detail_gt_salient_presence_weight = float(detail_gt_salient_presence_weight)
        self.detail_gt_salient_threshold = float(detail_gt_salient_threshold)
        self.detail_gt_salient_power = float(detail_gt_salient_power)
        self.detail_gt_salient_presence_ratio = float(detail_gt_salient_presence_ratio)
        self.detail_gt_chunk_size = max(1, int(detail_gt_chunk_size))
        self.residual_smooth_weight = float(residual_smooth_weight)
        self.residual_second_order_weight = float(residual_second_order_weight)
        self.residual_abs_weight = float(residual_abs_weight)
        self.residual_smooth_alpha = float(residual_smooth_alpha)
        self.final_grid_weight = float(final_grid_weight)
        self.final_phase_weight = float(final_phase_weight)
        self.final_grid_patch_size = max(2, int(final_grid_patch_size))
        self.final_grid_band = max(1, int(final_grid_band))
        self.final_grid_detail_threshold = float(final_grid_detail_threshold)
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
            event_top_fraction=mv_event_top_fraction,
            event_support_mode=mv_event_support_mode,
            hf_kernel=mv_hf_kernel,
            bidirectional=mv_bidirectional,
            max_pairs=mv_max_pairs,
            detach_warp_grid=mv_detach_warp_grid,
            projection_pose=mv_projection_pose,
        )

    @property
    def detail_gt_enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.detail_gt_normal_weight,
                self.detail_gt_hf_weight,
                self.detail_gt_grad_weight,
                self.detail_gt_salient_hf_weight,
                self.detail_gt_salient_mag_weight,
                self.detail_gt_salient_presence_weight,
            )
        )

    @property
    def residual_regularization_enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.residual_smooth_weight,
                self.residual_second_order_weight,
                self.residual_abs_weight,
                self.final_grid_weight,
                self.final_phase_weight,
            )
        )

    def _detail_gt_loss(
        self,
        *,
        pred_normals: torch.Tensor,
        gt_normals: torch.Tensor,
        event_support: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch, seq_len, height, width = valid_mask.shape
        # Normals/normal gradients use neighboring depth samples. Exclude a
        # one-pixel band around invalid depth so silhouette/background jumps
        # cannot dominate the detail objective.
        valid_flat = valid_mask.float().reshape(batch * seq_len, 1, height, width)
        interior_valid = (
            F.avg_pool2d(valid_flat, kernel_size=3, stride=1, padding=1) >= (1.0 - 1e-6)
        ).reshape(batch, seq_len, height, width)
        pred_flat = pred_normals.flatten(0, 1)
        gt_flat = gt_normals.flatten(0, 1)
        gt_grad = _normal_gradient_magnitude(gt_flat).view(batch, seq_len, 1, height, width).detach()

        detail_weight = _normalize_detail_support(
            gt_grad,
            interior_valid,
            threshold=self.detail_gt_threshold,
            power=self.detail_gt_weight_power,
        )
        if self.detail_gt_event_boost > 0.0:
            detail_weight = detail_weight * (1.0 + self.detail_gt_event_boost * event_support.unsqueeze(2).detach())
        detail_weight = detail_weight * interior_valid.unsqueeze(2).to(dtype=detail_weight.dtype)

        normal_loss = pred_normals.new_tensor(0.0)
        hf_loss = pred_normals.new_tensor(0.0)
        grad_loss = pred_normals.new_tensor(0.0)
        salient_hf_loss = pred_normals.new_tensor(0.0)
        salient_mag_loss = pred_normals.new_tensor(0.0)
        salient_presence_loss = pred_normals.new_tensor(0.0)
        salient_weight_mean = pred_normals.new_tensor(0.0)

        salient_enabled = (
            self.detail_gt_salient_hf_weight > 0.0
            or self.detail_gt_salient_mag_weight > 0.0
            or self.detail_gt_salient_presence_weight > 0.0
        )
        hf_needed = self.detail_gt_hf_weight > 0.0 or salient_enabled

        if self.detail_gt_normal_weight > 0.0:
            pred_n = F.normalize(pred_normals, dim=-1, eps=1e-6)
            gt_n = F.normalize(gt_normals.detach(), dim=-1, eps=1e-6)
            cos_loss = 1.0 - (pred_n * gt_n).sum(dim=-1, keepdim=False).clamp(-1.0, 1.0)
            normal_loss = _weighted_mean(cos_loss.unsqueeze(2), detail_weight)

        if self.detail_gt_grad_weight > 0.0:
            flat_count = batch * seq_len
            flat_detail_weight = detail_weight.reshape(flat_count, 1, height, width)
            flat_gt_grad = gt_grad.reshape(flat_count, 1, height, width)
            grad_num = pred_normals.new_tensor(0.0)
            grad_den = pred_normals.new_tensor(0.0)
            for start in range(0, flat_count, self.detail_gt_chunk_size):
                end = min(start + self.detail_gt_chunk_size, flat_count)
                pred_grad_chunk = _normal_gradient_magnitude(pred_flat[start:end])
                gt_grad_chunk = flat_gt_grad[start:end]
                weight_chunk = flat_detail_weight[start:end]
                grad_scale = gt_grad_chunk.flatten(2).amax(dim=-1).view(end - start, 1, 1, 1).clamp_min(1e-6)
                grad_num = grad_num + (((pred_grad_chunk - gt_grad_chunk).abs() / grad_scale) * weight_chunk).sum()
                grad_den = grad_den + weight_chunk.sum()
            grad_loss = grad_num / grad_den.clamp_min(1.0)

        if hf_needed:
            flat_count = batch * seq_len
            flat_detail_weight = detail_weight.reshape(flat_count, 1, height, width)
            flat_valid = interior_valid.reshape(flat_count, height, width)
            flat_gt_grad = gt_grad.reshape(flat_count, 1, height, width)
            flat_event_support = event_support.reshape(flat_count, height, width)

            hf_num = pred_normals.new_tensor(0.0)
            hf_den = pred_normals.new_tensor(0.0)
            salient_hf_num = pred_normals.new_tensor(0.0)
            salient_hf_den = pred_normals.new_tensor(0.0)
            salient_mag_num = pred_normals.new_tensor(0.0)
            salient_mag_den = pred_normals.new_tensor(0.0)
            salient_presence_num = pred_normals.new_tensor(0.0)
            salient_presence_den = pred_normals.new_tensor(0.0)
            salient_weight_sum = pred_normals.new_tensor(0.0)
            salient_weight_count = pred_normals.new_tensor(0.0)

            for start in range(0, flat_count, self.detail_gt_chunk_size):
                end = min(start + self.detail_gt_chunk_size, flat_count)
                pred_hf = _high_frequency_normals(pred_flat[start:end], kernel=self.mv_loss.hf_kernel)
                gt_hf = _high_frequency_normals(gt_flat[start:end].detach(), kernel=self.mv_loss.hf_kernel)

                if self.detail_gt_hf_weight > 0.0:
                    loss_map = (pred_hf - gt_hf.detach()).abs().mean(dim=1, keepdim=True)
                    weight = flat_detail_weight[start:end]
                    hf_num = hf_num + (loss_map * weight).sum()
                    hf_den = hf_den + weight.sum()

                if salient_enabled:
                    gt_hf_mag = gt_hf.detach().abs().mean(dim=1, keepdim=True)
                    pred_hf_mag = pred_hf.abs().mean(dim=1, keepdim=True)
                    grad_support = _normalize_detail_support_flat(
                        flat_gt_grad[start:end],
                        flat_valid[start:end],
                        threshold=0.0,
                        power=1.0,
                    )
                    hf_support = _normalize_detail_support_flat(
                        gt_hf_mag,
                        flat_valid[start:end],
                        threshold=0.0,
                        power=1.0,
                    )
                    salient_score = torch.maximum(grad_support, hf_support).detach()
                    salient_weight = (salient_score - self.detail_gt_salient_threshold).clamp_min(0.0)
                    salient_weight = salient_weight / max(1.0 - self.detail_gt_salient_threshold, 1e-6)
                    if self.detail_gt_salient_power != 1.0:
                        salient_weight = salient_weight.pow(self.detail_gt_salient_power)
                    if self.detail_gt_event_boost > 0.0:
                        salient_weight = salient_weight * (
                            1.0 + self.detail_gt_event_boost * flat_event_support[start:end].unsqueeze(1).detach()
                        )
                    salient_weight = salient_weight * flat_valid[start:end].unsqueeze(1).to(
                        dtype=salient_weight.dtype
                    )
                    salient_weight_sum = salient_weight_sum + salient_weight.detach().sum()
                    salient_weight_count = salient_weight_count + salient_weight.detach().numel()

                    if self.detail_gt_salient_hf_weight > 0.0:
                        loss_map = (pred_hf - gt_hf.detach()).abs().mean(dim=1, keepdim=True)
                        salient_hf_num = salient_hf_num + (loss_map * salient_weight).sum()
                        salient_hf_den = salient_hf_den + salient_weight.sum()

                    mag_scale = gt_hf_mag.flatten(2).amax(dim=-1).view(end - start, 1, 1, 1).clamp_min(1e-6)
                    if self.detail_gt_salient_mag_weight > 0.0:
                        loss_map = (pred_hf_mag - gt_hf_mag).abs() / mag_scale
                        salient_mag_num = salient_mag_num + (loss_map * salient_weight).sum()
                        salient_mag_den = salient_mag_den + salient_weight.sum()

                    if self.detail_gt_salient_presence_weight > 0.0:
                        target_mag = self.detail_gt_salient_presence_ratio * gt_hf_mag
                        loss_map = F.relu(target_mag - pred_hf_mag) / mag_scale
                        salient_presence_num = salient_presence_num + (loss_map * salient_weight).sum()
                        salient_presence_den = salient_presence_den + salient_weight.sum()

            if self.detail_gt_hf_weight > 0.0:
                hf_loss = hf_num / hf_den.clamp_min(1.0)
            if salient_enabled:
                salient_weight_mean = salient_weight_sum / salient_weight_count.clamp_min(1.0)
            if self.detail_gt_salient_hf_weight > 0.0:
                salient_hf_loss = salient_hf_num / salient_hf_den.clamp_min(1.0)
            if self.detail_gt_salient_mag_weight > 0.0:
                salient_mag_loss = salient_mag_num / salient_mag_den.clamp_min(1.0)
            if self.detail_gt_salient_presence_weight > 0.0:
                salient_presence_loss = salient_presence_num / salient_presence_den.clamp_min(1.0)

        salient_loss = (
            self.detail_gt_salient_hf_weight * salient_hf_loss
            + self.detail_gt_salient_mag_weight * salient_mag_loss
            + self.detail_gt_salient_presence_weight * salient_presence_loss
        )
        total = (
            self.detail_gt_normal_weight * normal_loss
            + self.detail_gt_hf_weight * hf_loss
            + self.detail_gt_grad_weight * grad_loss
            + salient_loss
        )
        details = {
            "detail_gt_loss": float(total.detach()),
            "detail_gt_normal_loss": float(normal_loss.detach()),
            "detail_gt_hf_loss": float(hf_loss.detach()),
            "detail_gt_grad_loss": float(grad_loss.detach()),
            "detail_gt_normal_contribution": float(
                (self.detail_gt_normal_weight * normal_loss).detach()
            ),
            "detail_gt_hf_contribution": float(
                (self.detail_gt_hf_weight * hf_loss).detach()
            ),
            "detail_gt_grad_contribution": float(
                (self.detail_gt_grad_weight * grad_loss).detach()
            ),
            "detail_gt_weight_mean": float(detail_weight.mean().detach()),
            "detail_gt_salient_loss": float(salient_loss.detach()),
            "detail_gt_salient_hf_loss": float(salient_hf_loss.detach()),
            "detail_gt_salient_mag_loss": float(salient_mag_loss.detach()),
            "detail_gt_salient_presence_loss": float(salient_presence_loss.detach()),
            "detail_gt_salient_weight_mean": float(salient_weight_mean.detach()),
        }
        return total, details

    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]):
        base_loss, details, aux = super().forward(model_output, views)
        details["base_supervised_loss"] = float(base_loss.detach())
        if not self.mv_loss.enabled and not self.detail_gt_enabled and not self.residual_regularization_enabled:
            details.update(
                {
                    "mv_event_loss": 0.0,
                    "mv_event_normal_loss": 0.0,
                    "mv_event_presence_loss": 0.0,
                    "mv_event_hf_loss": 0.0,
                    "mv_event_orient_loss": 0.0,
                    "mv_event_weight_mean": 0.0,
                    "detail_gt_loss": 0.0,
                    "detail_gt_normal_loss": 0.0,
                    "detail_gt_hf_loss": 0.0,
                    "detail_gt_grad_loss": 0.0,
                    "detail_gt_weight_mean": 0.0,
                    "detail_gt_salient_loss": 0.0,
                    "detail_gt_salient_hf_loss": 0.0,
                    "detail_gt_salient_mag_loss": 0.0,
                    "detail_gt_salient_presence_loss": 0.0,
                    "detail_gt_salient_weight_mean": 0.0,
                    "residual_smooth_loss": 0.0,
                    "residual_second_order_loss": 0.0,
                    "depth_residual_abs": 0.0,
                    "depth_residual_relative_abs": 0.0,
                    "residual_regularization_loss": 0.0,
                    "final_grid_suppress_loss": 0.0,
                    "final_patch_phase_loss": 0.0,
                    "final_antigrid_loss": 0.0,
                    "event_gate_mean": 0.0,
                    "coarse_depth_loss": 0.0,
                    "depth_refinement_gain": 0.0,
                    "coarse_depth_normal_loss": 0.0,
                    "final_depth_normal_loss": 0.0,
                    "normal_refinement_gain": 0.0,
                    "extra_loss_total": 0.0,
                    "total_loss_with_extra": float(base_loss.detach()),
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
        total_extra = depth_pred.new_tensor(0.0)

        if self.mv_loss.enabled:
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
            total_extra = total_extra + mv_loss
            details.update(mv_details)
        else:
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

        if self.detail_gt_enabled:
            gt_normals = None
            if self.detail_gt_normal_source in {"auto", "rendered", "normal"}:
                gt_normals = _gt_normals_from_views(views, device=depth_pred.device, dtype=depth_pred.dtype)
            if gt_normals is None or self.detail_gt_normal_source == "depth":
                gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics_gt)
            if self.detail_gt_event_boost > 0.0:
                event_support = _make_event_support(
                    views,
                    height=height,
                    width=width,
                    device=depth_pred.device,
                    dtype=depth_pred.dtype,
                    blur_kernel=self.mv_loss.event_blur_kernel,
                    dilate_kernel=self.mv_loss.event_dilate_kernel,
                    threshold=self.mv_loss.event_threshold,
                    power=self.mv_loss.event_power,
                    top_fraction=self.mv_loss.event_top_fraction,
                    mode=self.mv_loss.event_support_mode,
                ).detach()
            else:
                event_support = depth_pred.new_zeros((depth_pred.shape[0], depth_pred.shape[1], height, width))
            detail_gt_loss, detail_gt_details = self._detail_gt_loss(
                pred_normals=pred_normals,
                gt_normals=gt_normals,
                event_support=event_support,
                valid_mask=valid_mask,
            )
            total_extra = total_extra + detail_gt_loss
            details.update(detail_gt_details)
        else:
            details.update(
                {
                    "detail_gt_loss": 0.0,
                    "detail_gt_normal_loss": 0.0,
                    "detail_gt_hf_loss": 0.0,
                    "detail_gt_grad_loss": 0.0,
                    "detail_gt_weight_mean": 0.0,
                    "detail_gt_salient_loss": 0.0,
                    "detail_gt_salient_hf_loss": 0.0,
                    "detail_gt_salient_mag_loss": 0.0,
                    "detail_gt_salient_presence_loss": 0.0,
                    "detail_gt_salient_weight_mean": 0.0,
                }
            )
        depth_coarse = _stack_output_field(model_output, "depth_coarse")
        if depth_coarse is not None:
            depth_coarse = depth_coarse.to(device=depth_pred.device, dtype=depth_pred.dtype)
            aux["depth_coarse"] = depth_coarse.detach()
            depth_target = aux.get("depth_gt_aligned", depth_gt).to(device=depth_pred.device, dtype=depth_pred.dtype)
            coarse_depth_loss = fe.masked_l1(depth_coarse, depth_target, valid_mask)
            final_depth_loss = fe.masked_l1(depth_pred, depth_target, valid_mask)

            diagnostic_gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics_gt)
            coarse_normals = fe.depth_to_normals(depth_coarse.clamp_min(self.depth_min), intrinsics_gt)
            diagnostic_mask = valid_mask.clone()
            diagnostic_mask[..., 0, :] = False
            diagnostic_mask[..., -1, :] = False
            diagnostic_mask[..., :, 0] = False
            diagnostic_mask[..., :, -1] = False
            coarse_normal_loss = fe.masked_cosine_loss(coarse_normals, diagnostic_gt_normals, diagnostic_mask)
            final_normal_loss = fe.masked_cosine_loss(pred_normals, diagnostic_gt_normals, diagnostic_mask)
            details.update(
                {
                    "coarse_depth_loss": float(coarse_depth_loss.detach()),
                    "depth_refinement_gain": float((coarse_depth_loss - final_depth_loss).detach()),
                    "coarse_depth_normal_loss": float(coarse_normal_loss.detach()),
                    "final_depth_normal_loss": float(final_normal_loss.detach()),
                    "normal_refinement_gain": float((coarse_normal_loss - final_normal_loss).detach()),
                }
            )
        else:
            details.update(
                {
                    "coarse_depth_loss": 0.0,
                    "depth_refinement_gain": 0.0,
                    "coarse_depth_normal_loss": 0.0,
                    "final_depth_normal_loss": 0.0,
                    "normal_refinement_gain": 0.0,
                }
            )
        depth_residual = _stack_output_field(model_output, "depth_residual")
        residual_smooth_loss = depth_pred.new_tensor(0.0)
        residual_second_order_loss = depth_pred.new_tensor(0.0)
        residual_abs = depth_pred.new_tensor(0.0)
        residual_relative_abs = depth_pred.new_tensor(0.0)
        if depth_residual is not None:
            depth_residual = depth_residual.to(device=depth_pred.device, dtype=depth_pred.dtype)
            residual_abs = _weighted_mean(
                depth_residual.abs().unsqueeze(2),
                valid_mask.unsqueeze(2).to(dtype=depth_pred.dtype),
            )
            if self.residual_smooth_weight > 0.0:
                residual_smooth_loss = _residual_edge_aware_smoothness(
                    depth_residual,
                    views,
                    valid_mask,
                    edge_alpha=self.residual_smooth_alpha,
                )
            if self.residual_second_order_weight > 0.0:
                residual_second_order_loss = _residual_second_order_smoothness(depth_residual, valid_mask)
            relative_reference = (
                depth_coarse.clamp_min(self.depth_min) if depth_coarse is not None else depth_pred.clamp_min(self.depth_min)
            )
            residual_relative_abs = _weighted_mean(
                (depth_residual.abs() / relative_reference).unsqueeze(2),
                valid_mask.unsqueeze(2).to(dtype=depth_pred.dtype),
            )
            aux["depth_residual"] = depth_residual.detach()
        residual_regularization_loss = (
            self.residual_smooth_weight * residual_smooth_loss
            + self.residual_second_order_weight * residual_second_order_loss
            + self.residual_abs_weight * residual_abs
        )
        final_grid_loss = depth_pred.new_tensor(0.0)
        final_phase_loss = depth_pred.new_tensor(0.0)
        final_antigrid_loss = depth_pred.new_tensor(0.0)
        if self.final_grid_weight > 0.0 or self.final_phase_weight > 0.0:
            grid_gt_normals = fe.depth_to_normals(depth_gt.clamp_min(self.depth_min), intrinsics_gt)
            grid_gt_grad = _normal_gradient_magnitude(grid_gt_normals.flatten(0, 1)).view(
                depth_pred.shape[0], depth_pred.shape[1], 1, height, width
            )
            grid_detail_support = _normalize_detail_support(
                grid_gt_grad,
                valid_mask,
                threshold=self.final_grid_detail_threshold,
                power=1.0,
            )
            final_grid_loss, final_phase_loss = final_depth_patch_phase_antigrid_loss(
                depth_pred,
                valid_mask,
                grid_detail_support,
                patch_size=self.final_grid_patch_size,
                band=self.final_grid_band,
                eps=self.depth_min,
            )
            final_antigrid_loss = (
                self.final_grid_weight * final_grid_loss + self.final_phase_weight * final_phase_loss
            )
        residual_regularization_loss = residual_regularization_loss + final_antigrid_loss
        total_extra = total_extra + residual_regularization_loss
        details.update(
            {
                "residual_smooth_loss": float(residual_smooth_loss.detach()),
                "residual_second_order_loss": float(residual_second_order_loss.detach()),
                "depth_residual_abs": float(residual_abs.detach()),
                "depth_residual_relative_abs": float(residual_relative_abs.detach()),
                "residual_regularization_loss": float(residual_regularization_loss.detach()),
                "final_grid_suppress_loss": float(final_grid_loss.detach()),
                "final_patch_phase_loss": float(final_phase_loss.detach()),
                "final_antigrid_loss": float(final_antigrid_loss.detach()),
            }
        )
        event_gate = _stack_output_field(model_output, "event_gate")
        if event_gate is not None:
            event_gate = event_gate.to(device=depth_pred.device, dtype=depth_pred.dtype)
            gate_weight = valid_mask.to(dtype=depth_pred.dtype)
            details["event_gate_mean"] = float(
                ((event_gate * gate_weight).sum() / gate_weight.sum().clamp_min(1.0)).detach()
            )
            aux["event_gate"] = event_gate.detach()
        else:
            details["event_gate_mean"] = 0.0
        total_loss = base_loss + total_extra
        details["extra_loss_total"] = float(total_extra.detach())
        details["total_loss_with_extra"] = float(total_loss.detach())
        return total_loss, details, aux
