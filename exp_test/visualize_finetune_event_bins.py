"""Visualize per-view event bins exactly from finetune_event.py's loader.

The script builds the same event DataLoader path as finetune_event.py:

    finetune_event.build_event_loader -> DataLoader -> event_multiview_collate

For one selected batch/sample, it saves one PNG per view. Each PNG contains the
RGB frame, all events in that view, and five temporal event bins split by that
view's event_time_range.
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
    sample = {}
    for key, value in view.items():
        if key in VARIABLE_EVENT_KEYS:
            sample[key] = value[batch_index] if isinstance(value, (list, tuple)) else value
            continue

        if torch is not None and torch.is_tensor(value):
            sample[key] = value[batch_index] if value.ndim > 0 and value.shape[0] > batch_index else value
        elif isinstance(value, np.ndarray):
            sample[key] = value[batch_index] if value.ndim > 0 and value.shape[0] > batch_index else value
        elif isinstance(value, (list, tuple)) and len(value) > batch_index:
            sample[key] = value[batch_index]
        else:
            sample[key] = value
    return sample


def image_to_uint8(img) -> np.ndarray:
    img = to_numpy(img).astype(np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.min() < -0.1:
        img = (img + 1.0) * 0.5
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def infer_resolution(view: Dict) -> Tuple[int, int]:
    if "event_resolution" in view:
        resolution = to_numpy(view["event_resolution"]).astype(int).reshape(-1)
        if resolution.size >= 2:
            return int(resolution[0]), int(resolution[1])
    img = image_to_uint8(view["img"])
    height, width = img.shape[:2]
    return width, height


def label_panel(label: str, image: np.ndarray) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    bar = Image.new("RGB", (panel.width, 24), color=(18, 18, 18))
    ImageDraw.Draw(bar).text((6, 5), label, fill=(235, 235, 235))
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


def render_event_image(
    event_xy: np.ndarray,
    event_p: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.float32)
    if event_xy.size == 0:
        return image.astype(np.uint8)

    x = np.clip(event_xy[:, 0].astype(np.int64), 0, width - 1)
    y = np.clip(event_xy[:, 1].astype(np.int64), 0, height - 1)
    flat = y * width + x
    pos = np.zeros(height * width, dtype=np.float32)
    neg = np.zeros(height * width, dtype=np.float32)
    pos_mask = event_p > 0
    np.add.at(pos, flat[pos_mask], np.abs(event_p[pos_mask]).astype(np.float32, copy=False))
    np.add.at(neg, flat[~pos_mask], np.abs(event_p[~pos_mask]).astype(np.float32, copy=False))

    pos = np.log1p(pos.reshape(height, width))
    neg = np.log1p(neg.reshape(height, width))
    scale = max(float(pos.max()), float(neg.max()), EPS)
    image[..., 0] = pos / scale
    image[..., 1] = 0.25 * np.clip((pos + neg) / scale, 0.0, 1.0)
    image[..., 2] = neg / scale
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def render_time_bin(view: Dict, bin_idx: int, num_bins: int, width: int, height: int):
    event_xy = to_numpy(view.get("event_xy", np.zeros((0, 2), dtype=np.int32))).astype(np.int64)
    event_t = to_numpy(view.get("event_t", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    event_p = to_numpy(view.get("event_p", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    time_range = to_numpy(view.get("event_time_range", np.array([0.0, 0.0], dtype=np.float32))).astype(np.float32)

    if time_range.size >= 2:
        t0 = float(time_range[0])
        t1 = float(time_range[1])
    elif event_t.size > 0:
        t0 = float(event_t.min())
        t1 = float(event_t.max())
    else:
        t0 = 0.0
        t1 = 0.0

    if t1 <= t0 and event_t.size > 0:
        t0 = float(event_t.min())
        t1 = float(event_t.max())

    duration = max(t1 - t0, 0.0)
    bin_start = t0 + duration * bin_idx / max(num_bins, 1)
    bin_end = t0 + duration * (bin_idx + 1) / max(num_bins, 1)

    if event_xy.size == 0 or event_t.size == 0 or duration <= 0.0:
        selected = np.zeros((0, 2), dtype=np.int64)
        selected_p = np.zeros((0,), dtype=np.float32)
    elif bin_idx == num_bins - 1:
        mask = (event_t >= bin_start) & (event_t <= bin_end)
        selected = event_xy[mask]
        selected_p = event_p[mask]
    else:
        mask = (event_t >= bin_start) & (event_t < bin_end)
        selected = event_xy[mask]
        selected_p = event_p[mask]

    image = render_event_image(selected, selected_p, width, height)
    stats = {
        "events": int(selected.shape[0]),
        "positive": int((selected_p > 0).sum()),
        "negative": int((selected_p <= 0).sum()),
        "t_start": float(bin_start),
        "t_end": float(bin_end),
    }
    return image, stats


def visualize_view(view: Dict, out_dir: Path, batch_idx: int, view_idx: int, args) -> Dict:
    width, height = infer_resolution(view)
    rgb = image_to_uint8(view["img"])
    if rgb.shape[:2] != (height, width):
        rgb = np.array(Image.fromarray(rgb).resize((width, height), resample=Image.BILINEAR))

    event_xy = to_numpy(view.get("event_xy", np.zeros((0, 2), dtype=np.int32))).astype(np.int64)
    event_t = to_numpy(view.get("event_t", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    event_p = to_numpy(view.get("event_p", np.zeros((0,), dtype=np.float32))).astype(np.float32)
    all_event_img = render_event_image(event_xy, event_p, width, height)

    panels = [label_panel("rgb", rgb), label_panel(f"all n={len(event_t)}", all_event_img)]
    bin_stats = []
    for bin_idx in range(args.num_bins):
        bin_img, stats = render_time_bin(view, bin_idx, args.num_bins, width, height)
        bin_stats.append(stats)
        panels.append(label_panel(f"bin{bin_idx} n={stats['events']} [{stats['t_start']:.6f},{stats['t_end']:.6f}]", bin_img))

    rows = [panels[:2], panels[2:]]
    image_name = f"batch_{batch_idx:04d}_view_{view_idx:02d}_bins{args.num_bins}.png"
    make_grid(rows).save(out_dir / image_name)

    return {
        "view_idx": int(view_idx),
        "image": image_name,
        "label": scalar_to_text(view.get("label", "")),
        "instance": scalar_to_text(view.get("instance", "")),
        "event_time_range": to_numpy(view.get("event_time_range", np.array([0.0, 0.0]))).astype(float).tolist(),
        "event_source_resolution": to_numpy(view.get("event_source_resolution", np.array([0, 0]))).astype(int).tolist(),
        "event_spatial_transform": scalar_to_text(view.get("event_spatial_transform", "")),
        "total_events": int(len(event_t)),
        "event_t_min": float(event_t.min()) if event_t.size else None,
        "event_t_max": float(event_t.max()) if event_t.size else None,
        "bins": bin_stats,
    }


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
            event_spatial_transform=args.event_spatial_transform,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize finetune_event loader views as 5 event time bins")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/finetune_event_view_bins")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--batch-sample-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--event-spatial-transform", default="auto", choices=["auto", "none", "hflip", "vflip", "rot180", "hflip_rot180"])
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=1)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    fe.printer = logging.getLogger("visualize_finetune_event_bins")

    loader = fe.build_event_loader(build_loader_cfg(args), split=args.split)
    dataset = getattr(loader, "dataset", None)

    selected_views = None
    for batch_idx, views in enumerate(loader):
        if batch_idx == args.batch_idx:
            selected_views = views
            break
    if selected_views is None:
        raise RuntimeError(f"batch_idx={args.batch_idx} is out of range")

    records = []
    for view_idx, collated_view in enumerate(selected_views):
        view = select_batch_sample(collated_view, args.batch_sample_idx)
        records.append(visualize_view(view, out_dir, args.batch_idx, view_idx, args))

    summary = {
        "loader_path": "finetune_event.build_event_loader",
        "root": args.root,
        "active_scenes": dataset.get_active_scenes() if dataset is not None else [],
        "event_time_info_by_scene": {
            scene_name: scene_meta.get("event_time_info", {})
            for scene_name, scene_meta in getattr(dataset, "active_scene_data", {}).items()
        },
        "num_bins": args.num_bins,
        "records": records,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved PNG visualizations to {out_dir}")


if __name__ == "__main__":
    main()
