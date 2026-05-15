"""Visualize event bins through finetune_event.py's real dataloader path.

This is a training-loader sanity-check script. It builds the loader with
finetune_event.build_event_loader(), so the visualized views are after
DataLoader + event_multiview_collate, matching what the training loop receives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import finetune_event as fe


EPS = 1e-6
VARIABLE_EVENT_KEYS = {"events", "event_xy", "event_t", "event_p"}


def to_numpy(value):
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_to_text(value) -> str:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return str(value.item())
        if value.numel() == 1:
            return str(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0].item())
    return str(value)


def select_batch_sample(view: Dict, batch_index: int) -> Dict:
    """Convert one collated view from [B, ...] to one single-sample view."""
    sample = {}
    for key, value in view.items():
        if key in VARIABLE_EVENT_KEYS:
            if isinstance(value, (list, tuple)):
                sample[key] = value[batch_index]
            elif torch is not None and torch.is_tensor(value) and value.ndim > 0:
                sample[key] = value[batch_index]
            else:
                sample[key] = value
            continue

        if torch is not None and torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] > batch_index:
                sample[key] = value[batch_index]
            else:
                sample[key] = value
        elif isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] > batch_index:
                sample[key] = value[batch_index]
            else:
                sample[key] = value
        elif isinstance(value, (list, tuple)) and len(value) > batch_index:
            sample[key] = value[batch_index]
        else:
            sample[key] = value
    return sample


def infer_collated_batch_size(views: List[Dict]) -> int:
    if not views:
        return 0
    first_view = views[0]
    if "img" in first_view and torch is not None and torch.is_tensor(first_view["img"]):
        return int(first_view["img"].shape[0])
    if "event_xy" in first_view and isinstance(first_view["event_xy"], (list, tuple)):
        return len(first_view["event_xy"])
    for value in first_view.values():
        if torch is not None and torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return 1


def image_to_uint8(img) -> np.ndarray:
    img = to_numpy(img).astype(np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.min() < -0.1:
        img = (img + 1.0) * 0.5
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def label_panel(label: str, image: np.ndarray) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    bar = Image.new("RGB", (panel.width, 24), color=(18, 18, 18))
    ImageDraw.Draw(bar).text((6, 5), label, fill=(235, 235, 235))
    out = Image.new("RGB", (panel.width, panel.height + 24), color=(0, 0, 0))
    out.paste(bar, (0, 0))
    out.paste(panel, (0, 24))
    return out


def make_grid(rows: List[List[Image.Image]]) -> Image.Image:
    row_canvases = []
    for row in rows:
        width = sum(panel.width for panel in row)
        height = max(panel.height for panel in row)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        x = 0
        for panel in row:
            canvas.paste(panel, (x, 0))
            x += panel.width
        row_canvases.append(canvas)

    width = max(canvas.width for canvas in row_canvases)
    height = sum(canvas.height for canvas in row_canvases)
    grid = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for canvas in row_canvases:
        grid.paste(canvas, (0, y))
        y += canvas.height
    return grid


def infer_resolution(view: Dict) -> Tuple[int, int]:
    if "event_resolution" in view:
        resolution = to_numpy(view["event_resolution"]).astype(int).reshape(-1)
        if resolution.size >= 2:
            return int(resolution[0]), int(resolution[1])
    img = image_to_uint8(view["img"])
    height, width = img.shape[:2]
    return width, height


def event_bin_image(
    view: Dict,
    *,
    bin_idx: int,
    num_bins: int,
    width: int,
    height: int,
    log_scale: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    event_xy = to_numpy(view.get("event_xy", np.zeros((0, 2), dtype=np.int32))).astype(np.int64)
    event_t = to_numpy(view.get("event_t", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    event_p = to_numpy(view.get("event_p", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    time_range = to_numpy(view.get("event_time_range", np.array([0.0, 0.0], dtype=np.float32))).astype(np.float32)

    if event_xy.size == 0 or event_t.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8), {
            "events": 0,
            "positive": 0,
            "negative": 0,
            "t_start": float(time_range[0]) if time_range.size >= 2 else 0.0,
            "t_end": float(time_range[1]) if time_range.size >= 2 else 0.0,
        }

    t0 = float(time_range[0]) if time_range.size >= 2 else float(event_t.min())
    t1 = float(time_range[1]) if time_range.size >= 2 else float(event_t.max())
    if t1 <= t0:
        t0 = float(event_t.min())
        t1 = float(event_t.max())
    duration = max(t1 - t0, 1.0)
    bin_start = t0 + duration * bin_idx / max(num_bins, 1)
    bin_end = t0 + duration * (bin_idx + 1) / max(num_bins, 1)
    if bin_idx == num_bins - 1:
        bin_mask = (event_t >= bin_start) & (event_t <= bin_end)
    else:
        bin_mask = (event_t >= bin_start) & (event_t < bin_end)

    xy = event_xy[bin_mask]
    polarity = event_p[bin_mask]
    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)
    if xy.size > 0:
        x = np.clip(xy[:, 0], 0, width - 1)
        y = np.clip(xy[:, 1], 0, height - 1)
        flat = y * width + x
        pos_mask = polarity > 0
        np.add.at(pos.reshape(-1), flat[pos_mask], 1.0)
        np.add.at(neg.reshape(-1), flat[~pos_mask], 1.0)

    if log_scale:
        pos_vis = np.log1p(pos)
        neg_vis = np.log1p(neg)
    else:
        pos_vis = pos
        neg_vis = neg
    scale = max(float(pos_vis.max()), float(neg_vis.max()), EPS)
    pos_vis = np.clip(pos_vis / scale, 0.0, 1.0)
    neg_vis = np.clip(neg_vis / scale, 0.0, 1.0)

    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = pos_vis
    image[..., 1] = 0.25 * np.clip(pos_vis + neg_vis, 0.0, 1.0)
    image[..., 2] = neg_vis
    stats = {
        "events": int(xy.shape[0]),
        "positive": int((polarity > 0).sum()),
        "negative": int((polarity <= 0).sum()),
        "t_start": float(bin_start),
        "t_end": float(bin_end),
    }
    return (image * 255.0).round().astype(np.uint8), stats


def overlay_events(rgb: np.ndarray, event_img: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    event_mask = event_img.max(axis=-1, keepdims=True) > 0
    out = rgb.astype(np.float32)
    event_float = event_img.astype(np.float32)
    out = np.where(event_mask, (1.0 - alpha) * out + alpha * event_float, out)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def visualize_view(view: Dict, out_dir: Path, sample_idx: int, frame_idx: int, args) -> Dict:
    width, height = infer_resolution(view)
    rgb = image_to_uint8(view["img"])
    if rgb.shape[:2] != (height, width):
        rgb = np.array(Image.fromarray(rgb).resize((width, height), resample=Image.BILINEAR))
    event_xy = to_numpy(view.get("event_xy", np.zeros((0, 2), dtype=np.int32))).reshape(-1, 2)
    event_t = to_numpy(view.get("event_t", np.zeros((0,), dtype=np.float32))).reshape(-1)

    all_event_img, all_stats = event_bin_image(
        view,
        bin_idx=0,
        num_bins=1,
        width=width,
        height=height,
        log_scale=not args.linear_scale,
    )
    overlay = overlay_events(rgb, all_event_img, alpha=args.overlay_alpha)

    bin_panels = []
    bin_stats = []
    for bin_idx in range(args.num_bins):
        bin_img, stats = event_bin_image(
            view,
            bin_idx=bin_idx,
            num_bins=args.num_bins,
            width=width,
            height=height,
            log_scale=not args.linear_scale,
        )
        bin_stats.append(stats)
        bin_panels.append(label_panel(f"bin_{bin_idx:02d} n={stats['events']}", bin_img))
        if args.save_individual_bins:
            Image.fromarray(bin_img).save(out_dir / f"batch_{sample_idx:04d}_view_{frame_idx:02d}_bin_{bin_idx:02d}.png")

    rows = [
        [
            label_panel("rgb", rgb),
            label_panel(f"events_all n={all_stats['events']}", all_event_img),
            label_panel("overlay", overlay),
        ]
    ]
    for start in range(0, len(bin_panels), args.grid_cols):
        rows.append(bin_panels[start : start + args.grid_cols])

    image_name = f"batch_{sample_idx:04d}_view_{frame_idx:02d}_event_bins.png"
    grid = make_grid(rows)
    grid.save(out_dir / image_name)

    return {
        "batch_idx": sample_idx,
        "view_idx": frame_idx,
        "label": scalar_to_text(view.get("label", "")),
        "instance": scalar_to_text(view.get("instance", "")),
        "ldr_event_id": scalar_to_text(view.get("ldr_event_id", "")),
        "resolution": [width, height],
        "event_time_range": to_numpy(view.get("event_time_range", np.array([0.0, 0.0]))).astype(float).tolist(),
        "event_source_resolution": to_numpy(view.get("event_source_resolution", np.array([0, 0]))).astype(int).tolist(),
        "event_y_flip": bool(np.asarray(to_numpy(view.get("event_y_flip", False))).reshape(-1)[0]),
        "has_event": bool(np.asarray(to_numpy(view.get("has_event", False))).reshape(-1)[0]),
        "total_events": all_stats["events"],
        "positive_events": all_stats["positive"],
        "negative_events": all_stats["negative"],
        "event_t_min": float(event_t.min()) if event_t.size > 0 else None,
        "event_t_max": float(event_t.max()) if event_t.size > 0 else None,
        "event_xy_min": event_xy.min(axis=0).astype(int).tolist() if event_xy.size > 0 else None,
        "event_xy_max": event_xy.max(axis=0).astype(int).tolist() if event_xy.size > 0 else None,
        "bins": bin_stats,
        "image": image_name,
    }


def make_contact_sheet(records: List[Dict], out_dir: Path, *, max_width: int = 1600) -> str:
    panels = []
    for record in records:
        image_path = out_dir / record["image"]
        if not image_path.is_file():
            continue
        panel = Image.open(image_path).convert("RGB")
        if panel.width > max_width:
            scale = max_width / float(panel.width)
            new_size = (max(1, int(panel.width * scale)), max(1, int(panel.height * scale)))
            panel = panel.resize(new_size, resample=Image.BILINEAR)
        panels.append(panel)

    if not panels:
        return ""

    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    image_name = "all_views_contact_sheet.png"
    canvas.save(out_dir / image_name)
    return image_name


def build_loader_cfg(args) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        data=SimpleNamespace(
            root=args.root,
            num_views=args.num_views,
            resolution=tuple(args.resolution),
            fps=args.fps,
            scene_names=args.scene_names,
            initial_scene_idx=args.initial_scene_idx,
            active_scene_count=args.active_scene_count,
            test_frame_count=args.test_frame_count,
            ldr_event_id=args.ldr_event_id,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize event bins from finetune_event.py's dataloader")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/finetune_loader_event_bins")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--batch-idx", type=int, default=0, help="Which DataLoader batch to visualize")
    parser.add_argument("--batch-sample-idx", type=int, default=0, help="Which sample inside the selected batch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Frame indices in the sampled clip. Default: all views")
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=1)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay-alpha", type=float, default=0.75)
    parser.add_argument("--linear-scale", action="store_true", help="Use linear count visualization instead of log1p")
    parser.add_argument("--save-individual-bins", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    fe.printer = logging.getLogger("visualize_finetune_event_loader")

    cfg = build_loader_cfg(args)
    data_loader = fe.build_event_loader(cfg, split=args.split)
    dataset = getattr(data_loader, "dataset", None)
    if dataset is None or len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.root}")

    selected_views = None
    for batch_idx, views in enumerate(data_loader):
        if batch_idx < args.batch_idx:
            continue
        selected_views = views
        break
    if selected_views is None:
        raise RuntimeError(f"batch_idx={args.batch_idx} is out of range for loader with {len(data_loader)} batches")

    actual_batch_size = infer_collated_batch_size(selected_views)
    if args.batch_sample_idx < 0 or args.batch_sample_idx >= actual_batch_size:
        raise RuntimeError(f"batch_sample_idx={args.batch_sample_idx} must be in [0, {actual_batch_size})")

    records = []
    frame_indices = args.frames if args.frames is not None else list(range(len(selected_views)))
    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(selected_views):
            continue
        view = select_batch_sample(selected_views[frame_idx], args.batch_sample_idx)
        records.append(visualize_view(view, out_dir, args.batch_idx, frame_idx, args))

    contact_sheet = make_contact_sheet(records, out_dir)

    summary = {
        "root": args.root,
        "active_scenes": dataset.get_active_scenes(),
        "event_time_info_by_scene": {
            scene_name: scene_meta.get("event_time_info", {})
            for scene_name, scene_meta in getattr(dataset, "active_scene_data", {}).items()
        },
        "loader_path": "finetune_event.build_event_loader -> DataLoader -> event_multiview_collate",
        "ldr_event_id": args.ldr_event_id,
        "split": args.split,
        "batch_idx": args.batch_idx,
        "batch_sample_idx": args.batch_sample_idx,
        "batch_size": actual_batch_size,
        "num_views": args.num_views,
        "num_bins": args.num_bins,
        "contact_sheet": contact_sheet,
        "records": records,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved event bin visualizations to {out_dir}")


if __name__ == "__main__":
    main()
