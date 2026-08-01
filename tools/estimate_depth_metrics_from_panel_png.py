"""Approximate depth metrics by inverting Viridis panels in a saved PNG.

This is a recovery/inspection utility, not a replacement for evaluating the
original floating-point depth arrays.  It targets the 2x3 panel produced by
``paired_token_reliability/evaluate_rgb_pretrained_dsec.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# Axes bounds measured from the fixed ``figsize=(16, 10), dpi=130`` renderer.
# Coordinates are normalized so the same extraction also works after a uniform
# resize of the complete panel image.
PRED_RECT = (706 / 2080, 151 / 1300, 1284 / 2080, 588 / 1300)
GT_RECT = (1393 / 2080, 151 / 1300, 1971 / 2080, 588 / 1300)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="panel PNG(s) or directories")
    parser.add_argument("--glob", default="batch_*_view_*.png")
    parser.add_argument("--vmin", type=float, default=5.0,
                        help="shared lower depth color limit shown by the panel")
    parser.add_argument("--vmax", type=float, default=50.0,
                        help="shared upper depth color limit shown by the panel")
    parser.add_argument("--background-tolerance", type=float, default=10.0,
                        help="RGB distance from Viridis(vmin) treated as invalid GT")
    parser.add_argument("--output", default=None,
                        help="JSON path; default: beside one input or metrics_from_png.json")
    return parser.parse_args()


def collect(values, pattern):
    paths = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise RuntimeError("no matching panel PNGs")
    return paths


def crop_normalized(image, rect):
    width, height = image.size
    x0, y0, x1, y1 = rect
    box = (
        round(x0 * width), round(y0 * height),
        round(x1 * width), round(y1 * height),
    )
    return np.asarray(image.crop(box).convert("RGB"), dtype=np.uint8)


def viridis_lut(samples=4096):
    try:
        from matplotlib import colormaps
        cmap = colormaps["viridis"]
    except ImportError:
        from matplotlib import cm
        cmap = cm.get_cmap("viridis")
    values = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    colors = np.rint(cmap(values)[:, :3] * 255.0).astype(np.uint8)
    return values, colors


def invert_viridis(rgb, lut_values, lut_colors):
    shape = rgb.shape[:2]
    unique, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    decoded = np.empty(len(unique), dtype=np.float32)
    # Chunking bounds memory even for anti-aliased PNGs with many RGB colors.
    colors_i = lut_colors.astype(np.int16)
    for start in range(0, len(unique), 512):
        current = unique[start:start + 512].astype(np.int32)
        difference = current[:, None, :] - colors_i[None, :, :].astype(np.int32)
        distance = (difference ** 2).sum(2)
        decoded[start:start + len(current)] = lut_values[distance.argmin(1)]
    return decoded[inverse].reshape(shape)


def evaluate(path, vmin, vmax, tolerance, lut_values, lut_colors):
    image = Image.open(path).convert("RGB")
    pred_rgb = crop_normalized(image, PRED_RECT)
    gt_rgb = crop_normalized(image, GT_RECT)
    # Matplotlib's subplot rounding can differ by one pixel between columns.
    height = min(pred_rgb.shape[0], gt_rgb.shape[0])
    width = min(pred_rgb.shape[1], gt_rgb.shape[1])
    pred_rgb = pred_rgb[:height, :width]
    gt_rgb = gt_rgb[:height, :width]

    pred_unit = invert_viridis(pred_rgb, lut_values, lut_colors)
    gt_unit = invert_viridis(gt_rgb, lut_values, lut_colors)
    pred = vmin + pred_unit * (vmax - vmin)
    gt = vmin + gt_unit * (vmax - vmin)

    background = lut_colors[0].astype(np.float32)
    gt_color_distance = np.linalg.norm(gt_rgb.astype(np.float32) - background, axis=-1)
    # The renderer writes invalid GT depth as zero, which is clipped to vmin.
    # Color distance is therefore the only mask recoverable from this PNG.
    valid = (gt_color_distance > tolerance) & (gt_unit > 1.0 / 4095.0)
    valid &= np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if not np.any(valid):
        raise RuntimeError(f"no recoverable valid depth pixels in {path}")

    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    diff = p - g
    ratio = np.maximum(p / np.maximum(g, 1e-12), g / np.maximum(p, 1e-12))
    return {
        "file": str(path),
        "approximate": True,
        "pixels": int(valid.sum()),
        "AbsRel": float(np.mean(np.abs(diff) / np.maximum(g, 1e-12))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "RMSElog": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "MAE": float(np.mean(np.abs(diff))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def main():
    args = parse_args()
    if not args.vmax > args.vmin:
        raise ValueError("--vmax must be greater than --vmin")
    paths = collect(args.inputs, args.glob)
    lut_values, lut_colors = viridis_lut()
    rows = [evaluate(path, args.vmin, args.vmax, args.background_tolerance,
                     lut_values, lut_colors) for path in paths]

    total = sum(row["pixels"] for row in rows)
    keys = ("AbsRel", "RMSE", "RMSElog", "MAE", "delta1")
    aggregate = {
        key: sum(row[key] * row["pixels"] for row in rows) / max(total, 1)
        for key in keys
    }
    aggregate.update(pixels=total, images=len(rows), approximate=True)
    payload = {
        "warning": (
            "Metrics were approximately recovered from an 8-bit rendered PNG. "
            "Use original float depth/GT tensors for paper reporting."
        ),
        "assumed_shared_depth_range": [args.vmin, args.vmax],
        "aggregate": aggregate,
        "per_image": rows,
    }
    if args.output:
        output = Path(args.output)
    elif len(paths) == 1:
        output = paths[0].with_suffix(".approx_depth_metrics.json")
    else:
        output = Path(args.inputs[0]) / "metrics_from_png.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
