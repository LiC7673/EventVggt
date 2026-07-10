from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {str(k).removeprefix("module."): v for k, v in state.items()}


def move_views_to_device(views: List[dict], device: torch.device) -> List[dict]:
    moved = []
    for view in views:
        item = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                item[key] = value.to(device, non_blocking=True)
            elif isinstance(value, list):
                item[key] = [v.to(device, non_blocking=True) if torch.is_tensor(v) else v for v in value]
            else:
                item[key] = value
        moved.append(item)
    return moved


def normalize_event_voxel(voxel: torch.Tensor, count_cmax: float = 3.0) -> torch.Tensor:
    denominator = torch.log1p(voxel.new_tensor(max(float(count_cmax), 1.0)))
    return torch.log1p(voxel.clamp_min(0.0).clamp_max(float(count_cmax))) / denominator


def robust_unit_map(value: torch.Tensor, valid: torch.Tensor, quantile: float = 0.95) -> torch.Tensor:
    """Normalize each [B,S,H,W] map without allowing invalid pixels to set its scale."""
    flat_value = value.flatten(2)
    flat_valid = valid.flatten(2)
    outputs = []
    for batch_index in range(value.shape[0]):
        frames = []
        for frame_index in range(value.shape[1]):
            current = flat_value[batch_index, frame_index]
            mask = flat_valid[batch_index, frame_index]
            selected = current[mask]
            if selected.numel() == 0:
                scale = current.new_tensor(1.0)
            else:
                scale = torch.quantile(selected.float(), float(quantile)).to(current.dtype).clamp_min(1.0e-6)
            frames.append((current / scale).clamp(0.0, 1.0).reshape(value.shape[-2:]))
        outputs.append(torch.stack(frames))
    return torch.stack(outputs)


def infer_patch_grid(num_tokens: int, height: int, width: int) -> tuple[int, int]:
    target_ratio = float(width) / max(float(height), 1.0)
    candidates = []
    for grid_h in range(1, int(num_tokens**0.5) + 1):
        if num_tokens % grid_h == 0:
            grid_w = num_tokens // grid_h
            candidates.extend(((grid_h, grid_w), (grid_w, grid_h)))
    if not candidates:
        raise ValueError(f"Cannot factor {num_tokens} patch tokens into an image grid.")
    return min(candidates, key=lambda shape: abs(float(shape[1]) / shape[0] - target_ratio))


def build_reliability_target(
    event_voxel: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    token_a: torch.Tensor,
    token_b: torch.Tensor,
    *,
    token_cosine_floor: float = 0.80,
    token_agreement_mode: str = "fixed_floor",
    token_quantile_low: float = 0.05,
    token_quantile_high: float = 0.95,
    dilate_kernel: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build R_gt = event support * geometry detail * exposure-token agreement."""
    import finetune_event as fe

    valid = torch.isfinite(depth) & (depth > 1.0e-6)
    normals = fe.depth_to_normals(depth.float(), intrinsics.float())
    grad_x = torch.linalg.vector_norm(normals[..., 2:, 1:-1, :] - normals[..., :-2, 1:-1, :], dim=-1)
    grad_y = torch.linalg.vector_norm(normals[..., 1:-1, 2:, :] - normals[..., 1:-1, :-2, :], dim=-1)
    geometry = depth.new_zeros(depth.shape, dtype=torch.float32)
    geometry[..., 1:-1, 1:-1] = 0.5 * (grad_x + grad_y)
    geometry = robust_unit_map(geometry, valid, quantile=0.95).sqrt()

    activity = event_voxel.float().abs().sum(dim=2)
    event_support = (activity > 0.0).float()
    batch, seq_len, height, width = depth.shape
    if dilate_kernel > 1:
        event_support = F.max_pool2d(
            event_support.reshape(batch * seq_len, 1, height, width),
            kernel_size=dilate_kernel,
            stride=1,
            padding=dilate_kernel // 2,
        ).reshape(batch, seq_len, height, width)

    cosine = F.cosine_similarity(token_a.float(), token_b.float(), dim=-1)
    if token_agreement_mode == "robust_quantile":
        # Paired-token teachers deliberately make cosine values very high.
        # Normalize their *relative spatial confidence* instead of mapping all
        # values above a low fixed floor to almost one.
        flat = cosine.flatten(2)
        lo = torch.quantile(flat, token_quantile_low, dim=-1, keepdim=True)
        hi = torch.quantile(flat, token_quantile_high, dim=-1, keepdim=True)
        scale = (hi - lo).clamp_min(1.0e-4)
        agreement = ((flat - lo) / scale).clamp(0.0, 1.0).view_as(cosine)
    elif token_agreement_mode == "fixed_floor":
        agreement = ((cosine - token_cosine_floor) / max(1.0 - token_cosine_floor, 1.0e-4)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown token_agreement_mode={token_agreement_mode!r}")
    grid_h, grid_w = infer_patch_grid(agreement.shape[-1], height, width)
    agreement = F.interpolate(
        agreement.reshape(batch * seq_len, 1, grid_h, grid_w),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, seq_len, height, width)

    target = (event_support * geometry * agreement * valid.float()).clamp(0.0, 1.0)
    if dilate_kernel > 1:
        target = F.max_pool2d(
            target.reshape(batch * seq_len, 1, height, width),
            kernel_size=dilate_kernel,
            stride=1,
            padding=dilate_kernel // 2,
        ).reshape(batch, seq_len, height, width)
    weight = valid.float() * (0.15 + 0.85 * event_support)
    components = {
        "event_support": event_support,
        "geometry": geometry,
        "token_agreement": agreement,
    }
    return target, weight, components


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)


def as_uint8(value: torch.Tensor) -> np.ndarray:
    return (value.detach().clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy())

