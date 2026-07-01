"""Training-time temporal-bin visualization for event branch ablations."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch


TARGET_ROWS = (
    ("gt_geometry", "event_geometry_voxel"),
    ("gt_material", "event_material_voxel"),
    ("gt_noise", "event_noise_voxel"),
)
PREDICTION_ROWS = (
    ("pred_geometry", "pred_event_geometry_token"),
    ("pred_material", "pred_event_material_token"),
    ("pred_noise", "pred_event_noise_token"),
)


def _to_chw(value, sample_idx: int, frame_idx: Optional[int] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    # View fields are [B,C,H,W], while aux predictions are [B,S,C,H,W].
    if value.ndim == 5:
        if frame_idx is None:
            return None
        value = value[sample_idx, frame_idx]
    elif value.ndim == 4:
        value = value[sample_idx]
    elif value.ndim != 3:
        return None
    return value.detach().float().cpu()


def _split_polarity(voxel: torch.Tensor, max_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    channels = int(voxel.shape[0])
    source_bins = channels // 2
    bins = min(source_bins, max(1, int(max_bins)))
    if bins <= 0:
        raise ValueError(f"Expected polarity-separated voxel channels, got {tuple(voxel.shape)}")
    positive = voxel[:bins].clamp_min(0).numpy()
    negative = voxel[source_bins : source_bins + bins].clamp_min(0).numpy()
    return positive, negative


def _row_scale(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate([positive.reshape(-1), negative.reshape(-1)])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    return max(float(np.percentile(values, 99.5)), 1e-6)


def _polarity_rgb(positive: np.ndarray, negative: np.ndarray, scale: float) -> np.ndarray:
    pos = np.clip(positive / scale, 0.0, 1.0)
    neg = np.clip(negative / scale, 0.0, 1.0)
    image = np.zeros((*positive.shape, 3), dtype=np.float32)
    image[..., 0] = pos
    image[..., 1] = 0.25 * np.minimum(pos, neg)
    image[..., 2] = neg
    return np.rint(image * 255.0).astype(np.uint8)


def _resize_panel(image: Image.Image, panel_width: int) -> Image.Image:
    if panel_width <= 0 or image.width == panel_width:
        return image
    height = max(1, int(round(image.height * float(panel_width) / float(image.width))))
    return image.resize((panel_width, height), Image.Resampling.BILINEAR)


def _make_row(fe_module, label: str, voxel: torch.Tensor, max_bins: int, panel_width: int) -> Image.Image:
    positive, negative = _split_polarity(voxel, max_bins)
    scale = _row_scale(positive, negative)
    panels = []
    for bin_idx in range(positive.shape[0]):
        rgb = _polarity_rgb(positive[bin_idx], negative[bin_idx], scale)
        panel = fe_module.make_labeled_panel(f"{label} b{bin_idx:02d}", rgb)
        panels.append(_resize_panel(panel, panel_width))
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    row = Image.new("RGB", (width, height), color=(0, 0, 0))
    x = 0
    for panel in panels:
        row.paste(panel, (x, 0))
        x += panel.width
    return row


def _stack_rows(rows: List[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas


def save_event_bin_visuals(
    fe_module,
    cfg,
    views: List[Dict[str, torch.Tensor]],
    aux: Dict[str, torch.Tensor],
    global_step: int,
    *,
    frame_idx: Optional[int] = None,
    sample_idx: Optional[int] = None,
    vis_subdir: str = "train_vis",
    force: bool = False,
    filename_prefix: str = "",
) -> None:
    vis_cfg = getattr(cfg, "vis", None)
    if not bool(getattr(vis_cfg, "event_bins_enabled", True)):
        return
    save_every = int(getattr(vis_cfg, "save_every_steps", 200))
    if not force and (save_every <= 0 or global_step % save_every != 0):
        return
    sample_idx = int(getattr(vis_cfg, "sample_index", 0) if sample_idx is None else sample_idx)
    max_bins = int(getattr(vis_cfg, "event_bins_count", 10))
    panel_width = int(getattr(vis_cfg, "event_bin_panel_width", 112))
    if frame_idx is None:
        frame_ids = range(min(len(views), int(getattr(vis_cfg, "event_bins_num_views", 4))))
    else:
        frame_ids = [frame_idx]

    output_dir = Path(cfg.output_dir) / vis_subdir / "event_bins"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for current_frame in frame_ids:
        if current_frame < 0 or current_frame >= len(views):
            continue
        rows = []
        input_voxel = _to_chw(views[current_frame].get("event_voxel"), sample_idx)
        if input_voxel is not None:
            rows.append(_make_row(fe_module, "input", input_voxel, max_bins, panel_width))
        for label, key in TARGET_ROWS:
            voxel = _to_chw(views[current_frame].get(key), sample_idx)
            if voxel is not None:
                rows.append(_make_row(fe_module, label, voxel, max_bins, panel_width))
        for label, key in PREDICTION_ROWS:
            voxel = _to_chw(aux.get(key), sample_idx, current_frame)
            if voxel is not None:
                rows.append(_make_row(fe_module, label, voxel, max_bins, panel_width))
        if not rows:
            continue
        name = (
            f"{filename_prefix}{timestamp}_step_{global_step:07d}_"
            f"frame_{current_frame:02d}_event_bins.png"
        )
        _stack_rows(rows).save(output_dir / name)


def install_event_bin_visualization_hook(fe_module) -> None:
    if getattr(fe_module.save_training_visuals, "_event_bin_hook", False):
        return
    original = fe_module.save_training_visuals

    def save_with_event_bins(
        cfg,
        views,
        aux,
        global_step,
        frame_idx=None,
        sample_idx=None,
        vis_subdir="train_vis",
        force=False,
        filename_prefix="",
    ):
        original(
            cfg,
            views,
            aux,
            global_step,
            frame_idx=frame_idx,
            sample_idx=sample_idx,
            vis_subdir=vis_subdir,
            force=force,
            filename_prefix=filename_prefix,
        )
        save_event_bin_visuals(
            fe_module,
            cfg,
            views,
            aux,
            global_step,
            frame_idx=frame_idx,
            sample_idx=sample_idx,
            vis_subdir=vis_subdir,
            force=force,
            filename_prefix=filename_prefix,
        )

    save_with_event_bins._event_bin_hook = True
    fe_module.save_training_visuals = save_with_event_bins


__all__ = ["install_event_bin_visualization_hook", "save_event_bin_visuals"]
