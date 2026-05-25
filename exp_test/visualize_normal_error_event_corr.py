"""Visualize normal-error maps, adjacent-view event bins, and correlations.

The script follows the same dataset path as finetune_event.py.  If a checkpoint
is provided, it runs StreamVGGT and visualizes predicted-vs-GT normal error.
Without a checkpoint, it compares GT-depth-derived normals with loaded
``normal_gt`` when available, and still reports event/detail correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe  # noqa: E402
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset  # noqa: E402


EPS = 1e-6
VARIABLE_EVENT_KEYS = {"events", "event_xy", "event_t", "event_p"}


def safe_name(text: str, max_len: int = 80) -> str:
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")
    return text[:max_len] if text else "sample"


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_sample(view: Dict, batch_idx: int) -> Dict:
    sample = {}
    for key, value in view.items():
        if key in VARIABLE_EVENT_KEYS:
            sample[key] = value[batch_idx] if isinstance(value, (list, tuple)) else value
        elif torch.is_tensor(value):
            sample[key] = value[batch_idx] if value.ndim > 0 and value.shape[0] > batch_idx else value
        elif isinstance(value, np.ndarray):
            sample[key] = value[batch_idx] if value.ndim > 0 and value.shape[0] > batch_idx else value
        elif isinstance(value, (list, tuple)) and len(value) > batch_idx:
            sample[key] = value[batch_idx]
        else:
            sample[key] = value
    return sample


def move_views_to_device(views: List[Dict], device: torch.device) -> List[Dict]:
    moved = []
    for view in views:
        out = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                out[key] = value.to(device, non_blocking=True)
            elif isinstance(value, list):
                out[key] = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in value]
            else:
                out[key] = value
        moved.append(out)
    return moved


def tensor_image_to_uint8(image) -> np.ndarray:
    image = to_numpy(image).astype(np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.min() < -0.1:
        image = (image + 1.0) * 0.5
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def label_panel(label: str, image: np.ndarray) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    bar = Image.new("RGB", (panel.width, 24), color=(18, 18, 18))
    ImageDraw.Draw(bar).text((6, 5), label[:120], fill=(235, 235, 235))
    out = Image.new("RGB", (panel.width, panel.height + 24), color=(0, 0, 0))
    out.paste(bar, (0, 0))
    out.paste(panel, (0, 24))
    return out


def make_grid(rows: List[List[Image.Image]]) -> Image.Image:
    row_images = []
    for row in rows:
        width = sum(panel.width for panel in row)
        height = max(panel.height for panel in row)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        x = 0
        for panel in row:
            canvas.paste(panel, (x, 0))
            x += panel.width
        row_images.append(canvas)

    width = max(row.width for row in row_images)
    height = sum(row.height for row in row_images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for row in row_images:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas


def normalize_map(value: np.ndarray, mask: Optional[np.ndarray] = None, percentile: float = 99.0) -> np.ndarray:
    value = np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(value)
    if mask is not None:
        valid &= mask.astype(bool)
    if not valid.any():
        return np.zeros_like(value, dtype=np.float32)
    scale = np.percentile(np.abs(value[valid]), percentile)
    scale = max(float(scale), EPS)
    out = np.clip(value / scale, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        out = out.copy()
        out[~mask.astype(bool)] = 0.0
    return out


def gray_to_rgb(value: np.ndarray, mask: Optional[np.ndarray] = None, percentile: float = 99.0) -> np.ndarray:
    x = normalize_map(value, mask=mask, percentile=percentile)
    return (np.repeat(x[..., None], 3, axis=-1) * 255.0).round().astype(np.uint8)


def error_to_rgb(error_deg: np.ndarray, mask: Optional[np.ndarray] = None, max_deg: float = 45.0) -> np.ndarray:
    x = np.clip(np.nan_to_num(error_deg.astype(np.float32), nan=0.0) / max(max_deg, EPS), 0.0, 1.0)
    rgb = np.zeros((*x.shape, 3), dtype=np.float32)
    rgb[..., 0] = x
    rgb[..., 1] = np.sqrt(x) * 0.75
    rgb[..., 2] = 0.12 * (1.0 - x)
    if mask is not None:
        rgb[~mask.astype(bool)] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def normal_gradient(normal: torch.Tensor) -> torch.Tensor:
    # normal: [H, W, 3]
    n = torch.nn.functional.normalize(normal.float(), dim=-1, eps=1e-6)
    dx = torch.zeros(n.shape[:2], device=n.device, dtype=n.dtype)
    dy = torch.zeros_like(dx)
    dx[:, 1:-1] = 0.5 * (n[:, 2:, :] - n[:, :-2, :]).square().sum(dim=-1).sqrt()
    dy[1:-1, :] = 0.5 * (n[2:, :, :] - n[:-2, :, :]).square().sum(dim=-1).sqrt()
    dx[:, 0] = (n[:, 1, :] - n[:, 0, :]).square().sum(dim=-1).sqrt()
    dx[:, -1] = (n[:, -1, :] - n[:, -2, :]).square().sum(dim=-1).sqrt()
    dy[0, :] = (n[1, :, :] - n[0, :, :]).square().sum(dim=-1).sqrt()
    dy[-1, :] = (n[-1, :, :] - n[-2, :, :]).square().sum(dim=-1).sqrt()
    return dx + dy


def normal_error_deg(pred_normal: torch.Tensor, gt_normal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_n = torch.nn.functional.normalize(pred_normal.float(), dim=-1, eps=1e-6)
    gt_n = torch.nn.functional.normalize(gt_normal.float(), dim=-1, eps=1e-6)
    cos = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
    err = torch.rad2deg(torch.acos(cos))
    err = torch.where(mask.bool() & torch.isfinite(err), err, torch.zeros_like(err))
    return err


def get_gt_normals(sample_views: List[Dict], depth_gt: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    if all(("normal_gt" in view) for view in sample_views):
        normals = torch.stack([view["normal_gt"].float() for view in sample_views], dim=0)
    elif all(("normal" in view) for view in sample_views):
        normals = torch.stack([view["normal"].float() for view in sample_views], dim=0)
    else:
        return fe.depth_to_normals(depth_gt, intrinsics)

    if normals.ndim == 4 and normals.shape[1] == 3:
        normals = normals.permute(0, 2, 3, 1)
    if normals.detach().abs().amax() > 2.0:
        normals = normals / 127.5 - 1.0
    if normals.detach().abs().amax() < 1e-5:
        return fe.depth_to_normals(depth_gt, intrinsics)
    return torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)


def event_voxel_parts(view: Dict, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    voxel = to_numpy(view.get("event_voxel", np.zeros((0, 1, 1), dtype=np.float32))).astype(np.float32)
    if voxel.ndim != 3 or voxel.shape[0] < 2:
        return raw_events_to_voxel(view, num_bins)
    bins = max(voxel.shape[0] // 2, 1)
    pos = voxel[:bins].clip(min=0.0)
    neg = voxel[bins : 2 * bins].clip(min=0.0)
    return pos, neg


def raw_events_to_voxel(view: Dict, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    width, height = infer_resolution(view)
    pos = np.zeros((num_bins, height, width), dtype=np.float32)
    neg = np.zeros_like(pos)
    xy = to_numpy(view.get("event_xy", np.zeros((0, 2), dtype=np.int32))).astype(np.int64)
    t = to_numpy(view.get("event_t", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    p = to_numpy(view.get("event_p", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    if xy.size == 0 or t.size == 0:
        return pos, neg
    time_range = to_numpy(view.get("event_time_range", np.array([t.min(), t.max()], dtype=np.float32))).reshape(-1)
    t0 = float(time_range[0]) if time_range.size >= 2 else float(t.min())
    t1 = float(time_range[1]) if time_range.size >= 2 else float(t.max())
    if t1 <= t0:
        t0, t1 = float(t.min()), float(t.max())
    bins = np.floor((t - t0) / max(t1 - t0, EPS) * num_bins).astype(np.int64)
    bins = np.clip(bins, 0, num_bins - 1)
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    for b in range(num_bins):
        m = bins == b
        if not m.any():
            continue
        pm = m & (p > 0)
        nm = m & ~pm
        np.add.at(pos[b], (y[pm], x[pm]), np.abs(p[pm]).astype(np.float32))
        np.add.at(neg[b], (y[nm], x[nm]), np.abs(p[nm]).astype(np.float32))
    return pos, neg


def event_support_from_parts(pos: np.ndarray, neg: np.ndarray, mode: str = "temporal_polarity") -> np.ndarray:
    mode = str(mode or "abs").lower()
    activity = pos + neg
    activity_sum = activity.sum(axis=0)
    support = np.log1p(activity_sum)
    if mode in {"temporal", "temporal_contrast", "bin", "bin_aware", "temporal_polarity", "polarity"}:
        temporal_peak = activity.max(axis=0) / np.maximum(activity_sum, EPS)
        support = support * (0.5 + 0.5 * temporal_peak)
    if mode in {"polarity", "signed", "temporal_polarity", "polarity_aware"}:
        pos_sum = pos.sum(axis=0)
        neg_sum = neg.sum(axis=0)
        polarity_conf = np.abs(pos_sum - neg_sum) / np.maximum(pos_sum + neg_sum, EPS)
        support = support * (0.5 + 0.5 * polarity_conf)
    support = support.astype(np.float32)
    support = support / max(float(support.max()), EPS)
    return support


def event_rgb_from_parts(pos: np.ndarray, neg: np.ndarray, bin_idx: Optional[int] = None) -> np.ndarray:
    if bin_idx is None:
        pos_map = pos.sum(axis=0)
        neg_map = neg.sum(axis=0)
    else:
        idx = int(np.clip(bin_idx, 0, max(pos.shape[0] - 1, 0)))
        pos_map = pos[idx]
        neg_map = neg[idx]
    pos_map = np.log1p(pos_map)
    neg_map = np.log1p(neg_map)
    scale = max(float(pos_map.max()), float(neg_map.max()), EPS)
    rgb = np.zeros((*pos_map.shape, 3), dtype=np.float32)
    rgb[..., 0] = pos_map / scale
    rgb[..., 1] = 0.25 * (pos_map + neg_map) / scale
    rgb[..., 2] = neg_map / scale
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def infer_resolution(view: Dict) -> Tuple[int, int]:
    if "event_resolution" in view:
        resolution = to_numpy(view["event_resolution"]).astype(int).reshape(-1)
        if resolution.size >= 2 and resolution[0] > 0 and resolution[1] > 0:
            return int(resolution[0]), int(resolution[1])
    img = tensor_image_to_uint8(view["img"])
    height, width = img.shape[:2]
    return width, height


def pearson_corr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.sum() < 8:
        return float("nan")
    x = a[valid] - a[valid].mean()
    y = b[valid] - b[valid].mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > EPS else float("nan")


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.sum() < 8:
        return float("nan")
    return pearson_corr(rankdata(a[valid]), rankdata(b[valid]))


def build_dataset(args, split: str):
    active_count = args.active_scene_count if args.active_scene_count > 0 else 1
    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=args.scene_names if args.scene_names else None,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=active_count,
        split=split,
        test_frame_count=args.test_frame_count,
        ldr_event_id=args.ldr_event_id,
        event_spatial_transform=args.event_spatial_transform,
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_resize_bins,
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    if args.active_scene_count <= 0:
        dataset.set_active_scenes(dataset.scenes)
    return dataset


def select_scene_samples(dataset, samples_per_scene: int):
    if samples_per_scene <= 0:
        return dataset, None

    selected_indices = []
    scene_counts = {}
    for sample_idx, (scene_name, _) in enumerate(dataset.start_img_ids):
        count = scene_counts.get(scene_name, 0)
        if count >= samples_per_scene:
            continue
        selected_indices.append(sample_idx)
        scene_counts[scene_name] = count + 1

    return Subset(dataset, selected_indices), scene_counts


def build_model(args, device: torch.device):
    if not args.checkpoint:
        return None
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            variant=args.model_variant,
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            event_hidden_dim=args.event_hidden_dim,
            head_frames_chunk_size=args.head_frames_chunk_size,
            refiner_hidden_dim=args.refiner_hidden_dim,
            refiner_num_blocks=args.refiner_num_blocks,
            refiner_residual_scale=args.refiner_residual_scale,
            refiner_refine_points=args.refiner_refine_points,
            refiner_use_checkpoint=False,
        )
    )
    model = fe.build_event_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    msg = model.load_state_dict(fe.unwrap_state_dict(ckpt), strict=False)
    print(f"Loaded checkpoint {args.checkpoint}: {msg}")
    model.eval()
    return model


@torch.inference_mode()
def predict_depths(model, views: List[Dict], device: torch.device):
    if model is None:
        return None
    device_views = move_views_to_device(views, device)
    device_views = fe.maybe_denormalize_views(device_views)
    output = model(device_views)
    return torch.stack([res["depth"] for res in output.ress], dim=1).squeeze(-1).detach().cpu()


def visualize_pair(
    *,
    out_path: Path,
    sample_views: List[Dict],
    pair: Tuple[int, int],
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    normal_errors: torch.Tensor,
    valid_mask: torch.Tensor,
    args,
) -> Dict[str, float]:
    i, j = pair
    records = {}
    rows: List[List[Image.Image]] = []

    row = []
    for idx in pair:
        rgb = tensor_image_to_uint8(sample_views[idx]["img"])
        mask_np = to_numpy(valid_mask[idx]).astype(bool)
        row.append(label_panel(f"rgb view{idx}", rgb))
        row.append(label_panel(f"pred_normal view{idx}", fe.normal_to_uint8(pred_normals[idx], valid_mask[idx])))
        row.append(label_panel(f"gt_normal view{idx}", fe.normal_to_uint8(gt_normals[idx], valid_mask[idx])))
        row.append(label_panel(f"normal_err view{idx}", error_to_rgb(to_numpy(normal_errors[idx]), mask_np)))
    rows.append(row)

    row = []
    event_supports = {}
    for idx in pair:
        pos, neg = event_voxel_parts(sample_views[idx], args.num_bins)
        support = event_support_from_parts(pos, neg, mode=args.event_support_mode)
        event_supports[idx] = support
        mask_np = to_numpy(valid_mask[idx]).astype(bool)
        gt_detail = to_numpy(normal_gradient(gt_normals[idx]))
        pred_detail = to_numpy(normal_gradient(pred_normals[idx]))
        err_np = to_numpy(normal_errors[idx])
        records[f"view{idx}_corr_event_error_pearson"] = pearson_corr(support, err_np, mask_np)
        records[f"view{idx}_corr_event_error_spearman"] = spearman_corr(support, err_np, mask_np)
        records[f"view{idx}_corr_event_gt_detail_pearson"] = pearson_corr(support, gt_detail, mask_np)
        records[f"view{idx}_corr_event_pred_detail_pearson"] = pearson_corr(support, pred_detail, mask_np)
        records[f"view{idx}_normal_error_mean_deg"] = float(err_np[mask_np].mean()) if mask_np.any() else float("nan")
        records[f"view{idx}_event_pixels_ratio"] = float((support[mask_np] > args.event_threshold).mean()) if mask_np.any() else float("nan")
        row.append(label_panel(f"event all view{idx}", event_rgb_from_parts(pos, neg)))
        row.append(label_panel(f"event_support view{idx}", gray_to_rgb(support, mask_np)))
        row.append(label_panel(f"gt_detail view{idx}", gray_to_rgb(gt_detail, mask_np)))
    rows.append(row)

    if args.save_bin_rows:
        for idx in pair:
            pos, neg = event_voxel_parts(sample_views[idx], args.num_bins)
            bin_count = min(args.num_bins, pos.shape[0])
            rows.append([label_panel(f"view{idx} bin{b}", event_rgb_from_parts(pos, neg, b)) for b in range(bin_count)])

    pair_support = 0.5 * (event_supports[i] + event_supports[j])
    pair_error = 0.5 * (to_numpy(normal_errors[i]) + to_numpy(normal_errors[j]))
    pair_mask = to_numpy(valid_mask[i]).astype(bool) & to_numpy(valid_mask[j]).astype(bool)
    records["pair_corr_event_error_pearson"] = pearson_corr(pair_support, pair_error, pair_mask)
    records["pair_corr_event_error_spearman"] = spearman_corr(pair_support, pair_error, pair_mask)

    make_grid(rows).save(out_path)
    return records


def process_split(args, split: str, model, device: torch.device) -> List[Dict]:
    dataset = build_dataset(args, split)
    selected_dataset, selected_scene_counts = select_scene_samples(dataset, args.samples_per_scene)
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=event_multiview_collate,
    )
    out_dir = Path(args.output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{split}: active_scenes={dataset.get_active_scenes()}, "
        f"dataset_samples={len(dataset)}, visualized_samples={len(selected_dataset)}, "
        f"samples_per_scene={args.samples_per_scene}, selected={selected_scene_counts}, out={out_dir}"
    )

    records = []
    sample_global_idx = 0
    for batch_idx, views in enumerate(loader):
        if args.max_samples is not None and sample_global_idx >= args.max_samples:
            break
        pred_depths = predict_depths(model, views, device)
        batch_size = views[0]["img"].shape[0]
        for b in range(batch_size):
            if args.max_samples is not None and sample_global_idx >= args.max_samples:
                break
            sample_views = [select_sample(view, b) for view in views]
            depth_gt = torch.stack([view["depthmap"].float() for view in sample_views], dim=0)
            intrinsics = torch.stack([view["camera_intrinsics"].float() for view in sample_views], dim=0)
            valid_mask = torch.stack([view["valid_mask"].bool() for view in sample_views], dim=0)
            gt_normals = get_gt_normals(sample_views, depth_gt, intrinsics)
            if pred_depths is not None:
                pred_depth = pred_depths[b].float()
                pred_normals = fe.depth_to_normals(pred_depth, intrinsics)
                source = "checkpoint"
            else:
                pred_normals = fe.depth_to_normals(depth_gt, intrinsics)
                source = "gt_depth_baseline"
            normal_errors = normal_error_deg(pred_normals, gt_normals, valid_mask)

            label = sample_views[0].get("label", f"sample_{sample_global_idx}")
            label_text = label if isinstance(label, str) else str(label)
            for i in range(len(sample_views) - 1):
                j = i + 1
                img_name = f"sample_{sample_global_idx:06d}_pair_{i:02d}_{j:02d}_{safe_name(label_text)}.png"
                pair_record = visualize_pair(
                    out_path=out_dir / img_name,
                    sample_views=sample_views,
                    pair=(i, j),
                    pred_normals=pred_normals,
                    gt_normals=gt_normals,
                    normal_errors=normal_errors,
                    valid_mask=valid_mask,
                    args=args,
                )
                pair_record.update(
                    {
                        "split": split,
                        "sample_index": sample_global_idx,
                        "batch_index": batch_idx,
                        "batch_sample_index": b,
                        "pair_i": i,
                        "pair_j": j,
                        "image": str((out_dir / img_name).relative_to(Path(args.output_dir))),
                        "label": label_text,
                        "normal_error_source": source,
                    }
                )
                records.append(pair_record)
            sample_global_idx += 1
    return records


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def write_metrics(args, records: List[Dict]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    keys = sorted({key for record in records for key in record.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    metric_keys = [key for key in keys if key.startswith("view") or key.startswith("pair_corr")]
    summary = {
        "num_records": len(records),
        "root": args.root,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "event_support_mode": args.event_support_mode,
        "metrics_mean": {key: nanmean(record.get(key, float("nan")) for record in records) for key in metric_keys},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {csv_path} and {out_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize normal error, adjacent events, and correlations")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/normal_error_event_corr")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint for pred-vs-GT normal error")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional hard limit after scene sampling")
    parser.add_argument("--num-views", type=int, default=4, help="Number of consecutive views in each visualization")
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--event-spatial-transform", default="auto")
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-resize-bins", type=int, default=10)
    parser.add_argument("--event-support-mode", default="temporal_polarity", choices=["abs", "temporal", "polarity", "temporal_polarity"])
    parser.add_argument("--event-threshold", type=float, default=0.05)
    parser.add_argument("--save-bin-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=4, help="Default visualizes only four scenes; set <=0 for all")
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=1,
        help="Sliding-window samples rendered per scene; set <=0 to render every sample",
    )
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--model-variant", default="base")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--event-hidden-dim", type=int, default=32)
    parser.add_argument("--head-frames-chunk-size", type=int, default=2)
    parser.add_argument("--refiner-hidden-dim", type=int, default=16)
    parser.add_argument("--refiner-num-blocks", type=int, default=2)
    parser.add_argument("--refiner-residual-scale", type=float, default=0.03)
    parser.add_argument("--refiner-refine-points", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    model = build_model(args, device)

    splits = ["train", "test"] if args.split == "all" else [args.split]
    records = []
    for split in splits:
        records.extend(process_split(args, split, model, device))
    write_metrics(args, records)


if __name__ == "__main__":
    main()
