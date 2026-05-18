import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eventvggt.datasets.mvsec_event_dataset import get_mvsec_dataset


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_from_view(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    arr = _to_numpy(img).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.min() < -0.05:
        arr = arr * 0.5 + 0.5
    elif arr.max() > 2.0:
        arr = arr / 255.0
    arr = (np.clip(arr[..., :3], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _depth_to_rgb(depth, mask=None):
    depth = _to_numpy(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if mask is None:
        valid = np.isfinite(depth) & (depth > 0)
    else:
        valid = _to_numpy(mask).astype(bool) & np.isfinite(depth) & (depth > 0)

    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return Image.fromarray(out, mode="RGB")

    values = depth[valid]
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # A small built-in blue-cyan-yellow-red ramp; keeps the script independent of matplotlib.
    stops = np.array(
        [
            [20, 30, 80],
            [25, 120, 190],
            [80, 200, 160],
            [250, 220, 90],
            [210, 60, 45],
        ],
        dtype=np.float32,
    )
    x = norm * (len(stops) - 1)
    idx0 = np.floor(x).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, len(stops) - 1)
    w = (x - idx0)[..., None]
    color = stops[idx0] * (1.0 - w) + stops[idx1] * w
    out[valid] = np.clip(color[valid], 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _event_sum_to_rgb(event_voxel, size=None):
    voxel = _to_numpy(event_voxel).astype(np.float32)
    if voxel.ndim != 3 or voxel.shape[0] < 2:
        width, height = size if size is not None else (1, 1)
        return Image.new("RGB", (width, height), color=(0, 0, 0))

    bins = voxel.shape[0] // 2
    pos = np.log1p(np.maximum(voxel[:bins].sum(axis=0), 0.0))
    neg = np.log1p(np.maximum(voxel[bins : 2 * bins].sum(axis=0), 0.0))
    scale = max(float(pos.max()), float(neg.max()), 1e-6)

    rgb = np.zeros((*pos.shape, 3), dtype=np.float32)
    rgb[..., 0] = pos / scale
    rgb[..., 1] = 0.25 * np.clip((pos + neg) / scale, 0.0, 1.0)
    rgb[..., 2] = neg / scale
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")


def _event_bins_sheet(event_voxel, size=None):
    voxel = _to_numpy(event_voxel).astype(np.float32)
    if voxel.ndim != 3 or voxel.shape[0] < 2:
        width, height = size if size is not None else (1, 1)
        return Image.new("RGB", (width, height), color=(0, 0, 0))

    bins = voxel.shape[0] // 2
    height, width = voxel.shape[1:]
    cols = min(5, bins)
    rows = int(np.ceil(bins / cols))
    sheet = Image.new("RGB", (cols * width, rows * height), color=(0, 0, 0))

    for bin_idx in range(bins):
        pos = np.log1p(np.maximum(voxel[bin_idx], 0.0))
        neg = np.log1p(np.maximum(voxel[bins + bin_idx], 0.0))
        scale = max(float(pos.max()), float(neg.max()), 1e-6)
        panel = np.zeros((height, width, 3), dtype=np.float32)
        panel[..., 0] = pos / scale
        panel[..., 1] = 0.25 * np.clip((pos + neg) / scale, 0.0, 1.0)
        panel[..., 2] = neg / scale
        panel = Image.fromarray((np.clip(panel, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(panel)
        draw.text((5, 5), f"bin {bin_idx:02d}", fill=(255, 255, 255))
        row, col = divmod(bin_idx, cols)
        sheet.paste(panel, (col * width, row * height))

    return sheet


def _label(image, text):
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 24), color=(0, 0, 0))
    canvas.paste(image, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), text, fill=(255, 255, 255))
    return canvas


def _hstack(images):
    height = max(image.height for image in images)
    width = sum(image.width for image in images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _vstack(images):
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MVSECEventDataset samples.")
    parser.add_argument("--root", required=True, help="MVSEC hdf5 root or one hdf5 file.")
    parser.add_argument("--out", default="exp_test/mvsec_event_vis", help="Output directory.")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("W", "H"))
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--camera", default="left", choices=["left", "right"])
    parser.add_argument("--depth-key", default="depth_image_rect")
    parser.add_argument("--pose-key", default="pose")
    parser.add_argument("--event-format", default="xytp")
    parser.add_argument("--event-bins", type=int, default=10)
    parser.add_argument("--sequence-name", default=None)
    parser.add_argument("--event-spatial-transform", default="none", choices=["none", "hflip", "vflip", "rot180"])
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_mvsec_dataset(
        root=args.root,
        split=args.split,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=0,
        sequence_names=[args.sequence_name] if args.sequence_name else None,
        camera=args.camera,
        depth_key=args.depth_key,
        pose_key=args.pose_key,
        event_format=args.event_format,
        event_resize_bins=args.event_bins,
        spatial_transform=args.event_spatial_transform,
        return_debug_event_fields=False,
    )
    if len(dataset) <= 0:
        raise RuntimeError(f"No MVSEC samples found under {args.root}")

    sample_index = min(max(args.sample_index, 0), len(dataset) - 1)
    views = dataset[sample_index]
    summary_rows = []

    for view_idx, view in enumerate(views):
        prefix = f"sample_{sample_index:06d}_view_{view_idx:02d}"
        rgb = _rgb_from_view(view["img"])
        depth = _depth_to_rgb(view["depthmap"], view.get("mask", view.get("valid_mask")))
        event_sum = _event_sum_to_rgb(view.get("event_voxel"), size=rgb.size)
        event_bins = _event_bins_sheet(view.get("event_voxel"), size=rgb.size)

        rgb.save(out_dir / f"{prefix}_rgb.png")
        depth.save(out_dir / f"{prefix}_depth.png")
        event_sum.save(out_dir / f"{prefix}_event_sum.png")
        event_bins.save(out_dir / f"{prefix}_event_bins.png")

        label = view.get("label", f"view_{view_idx}")
        time_range = view.get("event_time_range", np.array([0.0, 0.0], dtype=np.float32))
        event_count = int(np.count_nonzero(_to_numpy(view.get("event_voxel", np.zeros((0, 1, 1))))))
        row = _hstack(
            [
                _label(rgb, f"{label}"),
                _label(depth, "depth"),
                _label(event_sum, f"events nz={event_count} t={float(time_range[0]):.6f}-{float(time_range[1]):.6f}"),
            ]
        )
        summary_rows.append(row)

    summary = _vstack(summary_rows)
    summary.save(out_dir / f"sample_{sample_index:06d}_summary.png")

    print(f"Saved MVSEC visualization to {os.fspath(out_dir)}")
    print(f"Dataset samples: {len(dataset)}, active scenes: {dataset.get_active_scenes()}")


if __name__ == "__main__":
    main()
