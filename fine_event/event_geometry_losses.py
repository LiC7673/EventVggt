from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import finetune_event as fe


def gradient(value: torch.Tensor) -> torch.Tensor:
    dx = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=1)


def normal_gradient(normal: torch.Tensor) -> torch.Tensor:
    return gradient(normal).abs().sum(dim=1, keepdim=True) / max(normal.shape[1], 1)


def gaussian_blur(value: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        return value

    channels = value.shape[1]
    radius = max(1, int(3.0 * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    kernel = kernel / kernel.sum().clamp_min(1e-12)

    kernel_x = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    value = F.pad(value, (radius, radius, 0, 0), mode="replicate")
    value = F.conv2d(value, kernel_x, groups=channels)
    value = F.pad(value, (0, 0, radius, radius), mode="replicate")
    value = F.conv2d(value, kernel_y, groups=channels)
    return value


def geometry_boundary(
    depth: torch.Tensor,
    normal: torch.Tensor,
    occupancy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    grad_d = gradient(torch.log(depth.clamp_min(1e-6))).abs().sum(dim=1, keepdim=True)
    grad_n = normal_gradient(normal)

    if occupancy is not None:
        grad_o = gradient(occupancy).abs().sum(dim=1, keepdim=True)
    else:
        grad_o = torch.zeros_like(grad_d)

    return torch.sigmoid(5.0 * grad_d + 3.0 * grad_n + 5.0 * grad_o - 1.0)


def boundary_sweep_loss(
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    normal0: torch.Tensor,
    normal1: torch.Tensor,
    event_pos: torch.Tensor,
    event_neg: torch.Tensor,
    occ0: Optional[torch.Tensor] = None,
    occ1: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b0 = geometry_boundary(depth0, normal0, occ0)
    b1 = geometry_boundary(depth1, normal1, occ1)

    sweep = torch.maximum(b0, b1)
    sweep = gaussian_blur(sweep, sigma=2.0).clamp(0, 1)

    event_abs = (event_pos.abs() + event_neg.abs()).detach()
    event_support = gaussian_blur(event_abs, sigma=1.5).clamp(0, 1)

    if valid_mask is not None:
        valid = valid_mask.to(device=sweep.device, dtype=sweep.dtype)
        sweep = sweep * valid
        event_abs = event_abs * valid
        event_support = event_support * valid

    event_to_geo = (event_abs * (1.0 - sweep)).sum() / (event_abs.sum() + 1e-6)
    geo_to_event = (sweep * (1.0 - event_support)).sum() / (sweep.sum() + 1e-6)
    return event_to_geo + 0.3 * geo_to_event


def signed_occupancy_event_loss(
    event_pos: torch.Tensor,
    event_neg: torch.Tensor,
    occ0: torch.Tensor,
    occ1: torch.Tensor,
    contrast_sign: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    event_abs = event_pos.abs() + event_neg.abs()
    event_sgn = (event_pos - event_neg) / (event_abs + 1e-6)

    delta_occ = gaussian_blur(occ1 - occ0, sigma=1.5)
    logits = float(contrast_sign) * event_sgn.detach() * delta_occ

    loss = F.softplus(-logits)
    weight = event_abs.detach()
    if valid_mask is not None:
        weight = weight * valid_mask.to(device=weight.device, dtype=weight.dtype)
    return (weight * loss).sum() / (weight.sum() + 1e-6)


def build_event_polarity_maps(
    views: List[Dict[str, torch.Tensor]],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    count_cmax: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = views[0]["img"].shape[0]
    seq_len = len(views)
    event_pos = torch.zeros(batch_size, seq_len, 1, height, width, device=device, dtype=dtype)
    event_neg = torch.zeros_like(event_pos)
    count_cmax = max(1.0, float(count_cmax))

    for frame_idx, view in enumerate(views):
        xy_values = view.get("event_xy")
        p_values = view.get("event_p")
        if xy_values is None or p_values is None:
            continue

        for batch_idx in range(batch_size):
            xy = xy_values[batch_idx]
            polarity = p_values[batch_idx]
            if xy.numel() == 0 or polarity.numel() == 0:
                continue

            xy = xy.to(device=device)
            polarity = polarity.to(device=device)
            x = xy[:, 0].long().clamp_(0, width - 1)
            y = xy[:, 1].long().clamp_(0, height - 1)
            flat_idx = y * width + x

            pos_mask = polarity > 0
            neg_mask = ~pos_mask
            if pos_mask.any():
                event_pos[batch_idx, frame_idx, 0].view(-1).index_add_(
                    0,
                    flat_idx[pos_mask],
                    torch.ones_like(flat_idx[pos_mask], dtype=dtype),
                )
            if neg_mask.any():
                event_neg[batch_idx, frame_idx, 0].view(-1).index_add_(
                    0,
                    flat_idx[neg_mask],
                    torch.ones_like(flat_idx[neg_mask], dtype=dtype),
                )

    denom = torch.log1p(event_pos.new_tensor(count_cmax))
    event_pos = torch.log1p(event_pos.clamp(max=count_cmax)) / denom
    event_neg = torch.log1p(event_neg.clamp(max=count_cmax)) / denom
    return event_pos, event_neg


def stack_occupancy_from_views(
    views: List[Dict[str, torch.Tensor]],
    fallback_mask: torch.Tensor,
) -> torch.Tensor:
    if all("mask" in view for view in views):
        occupancy = fe.stack_view_field(views, "mask").to(device=fallback_mask.device, dtype=fallback_mask.dtype)
    else:
        occupancy = fallback_mask.to(dtype=fallback_mask.dtype)
    return occupancy.unsqueeze(2)


def predicted_normals_chw(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    normals = fe.depth_to_normals(depth, intrinsics)
    return normals.permute(0, 1, 4, 2, 3).contiguous()


def flatten_pairs(value: torch.Tensor, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
    value = value[:, start:end]
    return value.reshape(-1, *value.shape[2:])


def soft_occupancy_from_depth(
    depth: torch.Tensor,
    threshold: float = 0.02,
    temperature: float = 0.02,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    return torch.sigmoid((depth - float(threshold)) / temperature).unsqueeze(2)
