"""Visualize event bins exactly as loaded by MyEventDataset.

This is a dataloader sanity-check script. It uses get_combined_dataset(), then
renders event_xy/event_t/event_p after resize/mask processing, so the output is
what the training code actually sees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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

from eventvggt.datasets.my_event_dataset import get_combined_dataset


EPS = 1e-6


def to_numpy(value):
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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
            Image.fromarray(bin_img).save(out_dir / f"sample_{sample_idx:04d}_frame_{frame_idx:02d}_bin_{bin_idx:02d}.png")

    rows = [
        [
            label_panel("rgb", rgb),
            label_panel(f"events_all n={all_stats['events']}", all_event_img),
            label_panel("overlay", overlay),
        ]
    ]
    for start in range(0, len(bin_panels), args.grid_cols):
        rows.append(bin_panels[start : start + args.grid_cols])

    image_name = f"sample_{sample_idx:04d}_frame_{frame_idx:02d}_event_bins.png"
    make_grid(rows).save(out_dir / image_name)

    return {
        "sample_idx": sample_idx,
        "frame_idx": frame_idx,
        "label": str(view.get("label", "")),
        "instance": str(view.get("instance", "")),
        "ldr_event_id": str(view.get("ldr_event_id", "")),
        "resolution": [width, height],
        "event_time_range": to_numpy(view.get("event_time_range", np.array([0.0, 0.0]))).astype(float).tolist(),
        "total_events": all_stats["events"],
        "positive_events": all_stats["positive"],
        "negative_events": all_stats["negative"],
        "bins": bin_stats,
        "image": image_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataloader event bins")
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="exp_test/dataset_event_bins")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Frame indices in the sampled clip. Default: all views")
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--ldr-event-id", default="5")
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=3)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay-alpha", type=float, default=0.75)
    parser.add_argument("--linear-scale", action="store_true", help="Use linear count visualization instead of log1p")
    parser.add_argument("--save-individual-bins", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=args.scene_names,
        initial_scene_idx=args.initial_scene_idx,
        active_scene_count=args.active_scene_count,
        split=args.split,
        test_frame_count=args.test_frame_count,
        ldr_event_id=args.ldr_event_id,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.root}")

    records = []
    for offset in range(args.num_samples):
        sample_idx = min(args.sample_idx + offset, len(dataset) - 1)
        views = dataset[sample_idx]
        frame_indices = args.frames if args.frames is not None else list(range(len(views)))
        for frame_idx in frame_indices:
            if frame_idx < 0 or frame_idx >= len(views):
                continue
            records.append(visualize_view(views[frame_idx], out_dir, sample_idx, frame_idx, args))

    summary = {
        "root": args.root,
        "active_scenes": dataset.get_active_scenes(),
        "ldr_event_id": args.ldr_event_id,
        "sample_idx": args.sample_idx,
        "num_samples": args.num_samples,
        "num_bins": args.num_bins,
        "records": records,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved event bin visualizations to {out_dir}")


if __name__ == "__main__":
    main()
