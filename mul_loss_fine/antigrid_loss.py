"""Patch-phase anti-grid losses for final depth maps."""

from typing import Tuple

import torch


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _phase_band_mask(length: int, patch_size: int, band: int, device: torch.device, dtype: torch.dtype):
    if length <= 0:
        return torch.empty((0,), device=device, dtype=dtype)
    target_index = torch.arange(1, length + 1, device=device)
    phase = torch.remainder(target_index, patch_size)
    distance = torch.minimum(phase, patch_size - phase)
    return (distance < band).to(dtype=dtype)


def _phase_variance_loss(
    grad: torch.Tensor,
    weight: torch.Tensor,
    *,
    patch_size: int,
    axis: str,
) -> torch.Tensor:
    if grad.numel() == 0:
        return grad.new_tensor(0.0)
    phase_len = grad.shape[-1] if axis == "x" else grad.shape[-2]
    target_index = torch.arange(1, phase_len + 1, device=grad.device)
    phases = torch.remainder(target_index, patch_size)
    means = []
    counts = []
    for phase in range(patch_size):
        phase_mask = (phases == phase).to(dtype=grad.dtype)
        if axis == "x":
            phase_mask = phase_mask.view(1, 1, 1, phase_len)
        else:
            phase_mask = phase_mask.view(1, 1, phase_len, 1)
        phase_weight = weight * phase_mask
        means.append(_weighted_mean(grad, phase_weight))
        counts.append(phase_weight.detach().sum())
    means = torch.stack(means)
    counts = torch.stack(counts)
    valid_phase = (counts > 0).to(dtype=grad.dtype)
    center = (means * counts).sum() / counts.sum().clamp_min(1.0)
    return ((means - center).abs() * valid_phase).sum() / valid_phase.sum().clamp_min(1.0)


def final_depth_patch_phase_antigrid_loss(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    detail_support: torch.Tensor,
    *,
    patch_size: int = 14,
    band: int = 1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Suppress fixed patch-grid frequency in final log-depth.

    The penalty is masked by ``1 - detail_support`` so real GT high-frequency
    geometry is left available for the detail losses to learn.
    """
    patch_size = max(int(patch_size), 2)
    band = max(int(band), 1)
    if depth.ndim != 4:
        zero = depth.new_tensor(0.0)
        return zero, zero

    log_depth = torch.log(depth.clamp_min(eps))
    finite_mask = torch.isfinite(log_depth)
    valid = valid_mask.to(device=depth.device).bool() & finite_mask
    non_detail = (1.0 - detail_support.squeeze(2).detach()).clamp(0.0, 1.0)
    base_weight = valid.to(dtype=depth.dtype) * non_detail.square()
    _, _, height, width = depth.shape
    boundary_terms = []
    phase_terms = []

    if width > 1:
        dx = (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs()
        wx = torch.minimum(base_weight[..., :, 1:], base_weight[..., :, :-1])
        x_boundary = _phase_band_mask(width - 1, patch_size, band, depth.device, depth.dtype)
        boundary_terms.append(_weighted_mean(dx, wx * x_boundary.view(1, 1, 1, -1)))
        phase_terms.append(_phase_variance_loss(dx, wx, patch_size=patch_size, axis="x"))

    if height > 1:
        dy = (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs()
        wy = torch.minimum(base_weight[..., 1:, :], base_weight[..., :-1, :])
        y_boundary = _phase_band_mask(height - 1, patch_size, band, depth.device, depth.dtype)
        boundary_terms.append(_weighted_mean(dy, wy * y_boundary.view(1, 1, -1, 1)))
        phase_terms.append(_phase_variance_loss(dy, wy, patch_size=patch_size, axis="y"))

    zero = depth.new_tensor(0.0)
    boundary_loss = torch.stack(boundary_terms).mean() if boundary_terms else zero
    phase_loss = torch.stack(phase_terms).mean() if phase_terms else zero
    return boundary_loss, phase_loss
