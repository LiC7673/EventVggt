"""Visualize additive event components using red/blue polarity on white.

The script renders:
  1. full
  2. geometry_motion
  3. material_reflection + noise

All panels use the same spatial resolution and color normalization. Positive
events are red, negative events are blue, and pixels without events are white.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(
    r"F:\TreeOBJ\reflective_raw\Actaeon_Anodized_Red\events_additive"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <root>/vis_event_components",
    )
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Default: <scene>/Mask. All masks are unioned for full-stream rendering.",
    )
    p.add_argument("--mask-threshold", type=int, default=250)
    p.add_argument(
        "--flip-event-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip Blender bottom-left event coordinates to image/mask coordinates.",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Shared percentile for robust event-density normalization.",
    )
    p.add_argument(
        "--linear",
        action="store_true",
        help="Use linear counts instead of log(1+count) visualization.",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        return decode_attr(value.reshape(-1)[0])
    return value


def columns_from_attrs(attrs):
    text = str(decode_attr(attrs.get("columns", ""))).lower().replace(" ", "")
    fmt = str(decode_attr(attrs.get("format", ""))).lower().replace(" ", "")
    text = text or fmt
    if "x,y,t,p" in text:
        return {"x": 0, "y": 1, "t": 2, "p": 3}
    if "t,x,y,p" in text:
        return {"t": 0, "x": 1, "y": 2, "p": 3}
    return None


def infer_columns(sample):
    """Infer timestamp/polarity columns using the dataset's storage traits."""
    if sample.ndim != 2 or sample.shape[1] < 4 or len(sample) == 0:
        return {"t": 0, "x": 1, "y": 2, "p": 3}
    scores = []
    for col in range(sample.shape[1]):
        x = sample[:, col].astype(np.float64, copy=False)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            scores.append((-np.inf, col))
            continue
        monotonic = np.mean(np.diff(x) >= 0)
        binary = len(np.unique(x[:5000])) <= 4 and x.min() >= -1 and x.max() <= 1
        scores.append((50 * monotonic + np.log1p(np.ptp(x)) - 20 * binary, col))
    t_col = max(scores)[1]
    remaining = [c for c in range(sample.shape[1]) if c != t_col]
    p_col = max(
        remaining,
        key=lambda c: (
            len(np.unique(np.round(sample[:20000, c], 3))) <= 4,
            np.nanmin(sample[:, c]) >= -1 and np.nanmax(sample[:, c]) <= 1,
        ),
    )
    coords = [c for c in remaining if c != p_col][:2]
    return {"t": t_col, "x": coords[0], "y": coords[1], "p": p_col}


def inspect_h5(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        if "events" not in h5:
            raise KeyError(f"{path} has no 'events' dataset; keys={list(h5.keys())}")
        ds = h5["events"]
        attrs = {k: decode_attr(v) for k, v in h5.attrs.items()}
        attrs.update({k: decode_attr(v) for k, v in ds.attrs.items()})
        sample = ds[: min(len(ds), 100_000)]
        columns = columns_from_attrs(attrs) or infer_columns(sample)
        width = attrs.get("width", attrs.get("event_width"))
        height = attrs.get("height", attrs.get("event_height"))
        if width is None:
            width = int(np.nanmax(sample[:, columns["x"]])) + 1
        if height is None:
            height = int(np.nanmax(sample[:, columns["y"]])) + 1
        return {
            "path": path,
            "count": int(len(ds)),
            "columns": columns,
            "width": int(width),
            "height": int(height),
        }


def accumulate(meta, width, height, chunk_size):
    positive = np.zeros((height, width), dtype=np.uint64)
    negative = np.zeros((height, width), dtype=np.uint64)
    columns = meta["columns"]
    with h5py.File(meta["path"], "r") as h5:
        ds = h5["events"]
        for start in range(0, len(ds), chunk_size):
            events = np.asarray(ds[start : min(start + chunk_size, len(ds))])
            raw_x = events[:, columns["x"]]
            raw_y = events[:, columns["y"]]
            p = events[:, columns["p"]]
            valid = (
                np.isfinite(raw_x) & np.isfinite(raw_y) & np.isfinite(p)
                & (raw_x >= 0) & (raw_x < width)
                & (raw_y >= 0) & (raw_y < height)
            )
            x = raw_x[valid].astype(np.int64)
            y = raw_y[valid].astype(np.int64)
            p = p[valid]
            np.add.at(positive, (y[p > 0], x[p > 0]), 1)
            np.add.at(negative, (y[p <= 0], x[p <= 0]), 1)
    return positive, negative


def transformed_counts(pair, use_log):
    pos, neg = pair
    if use_log:
        return np.log1p(pos.astype(np.float32)), np.log1p(neg.astype(np.float32))
    return pos.astype(np.float32), neg.astype(np.float32)


def shared_limit(pairs, percentile, use_log):
    values = []
    for pair in pairs:
        pos, neg = transformed_counts(pair, use_log)
        values.extend((pos[pos > 0], neg[neg > 0]))
    values = [v for v in values if v.size]
    if not values:
        return 1.0
    return max(float(np.percentile(np.concatenate(values), percentile)), 1e-6)


def polarity_rgb(pair, limit, use_log):
    pos, neg = transformed_counts(pair, use_log)
    pos = np.clip(pos / limit, 0, 1)
    neg = np.clip(neg / limit, 0, 1)
    rgb = np.ones((*pos.shape, 3), dtype=np.float32)
    # Positive-only -> red; negative-only -> blue; overlap -> purple.
    rgb[..., 0] = 1.0 - neg
    rgb[..., 1] = 1.0 - np.maximum(pos, neg)
    rgb[..., 2] = 1.0 - pos
    return np.clip(rgb, 0, 1)


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(image * 255).astype(np.uint8)).save(path)


def load_union_mask(mask_dir: Path, width: int, height: int, threshold: int):
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Mask directory is required but was not found: {mask_dir}"
        )
    paths = sorted(
        p for p in mask_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise RuntimeError(f"No mask images found under {mask_dir}")
    union = np.zeros((height, width), dtype=bool)
    for path in paths:
        image = Image.open(path).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.NEAREST)
        rgb = np.asarray(image, dtype=np.uint8)
        union |= np.all(rgb >= int(threshold), axis=-1)
    if not union.any():
        raise RuntimeError(
            f"The union mask is empty at threshold={threshold}: {mask_dir}"
        )
    return union, paths


def main():
    args = parse_args()
    output = args.output or args.root / "vis_event_components"
    paths = {
        "full": args.root / "full" / "events.h5",
        "geometry_motion": args.root / "geometry_motion" / "events.h5",
        "material_reflection": args.root / "material_reflection" / "events.h5",
        "noise": args.root / "noise" / "events.h5",
    }
    metadata = {name: inspect_h5(path) for name, path in paths.items()}
    width = max(meta["width"] for meta in metadata.values())
    height = max(meta["height"] for meta in metadata.values())
    scene_dir = args.root.parent
    mask_dir = args.mask_dir or scene_dir / "Mask"
    mask, mask_paths = load_union_mask(
        mask_dir, width, height, args.mask_threshold
    )

    counts = {
        name: accumulate(meta, width, height, args.chunk_size)
        for name, meta in metadata.items()
    }
    # Additive renderer events use Blender's bottom-left row origin, whereas
    # PNG masks use top-left image coordinates.
    if args.flip_event_y:
        counts = {
            name: (np.flipud(pair[0]), np.flipud(pair[1]))
            for name, pair in counts.items()
        }
    counts = {
        name: (pair[0] * mask, pair[1] * mask)
        for name, pair in counts.items()
    }
    non_geometry = (
        counts["material_reflection"][0] + counts["noise"][0],
        counts["material_reflection"][1] + counts["noise"][1],
    )
    display_pairs = (
        counts["full"],
        counts["geometry_motion"],
        non_geometry,
    )
    limit = shared_limit(display_pairs, args.percentile, not args.linear)
    images = [
        polarity_rgb(pair, limit, not args.linear) for pair in display_pairs
    ]

    output.mkdir(parents=True, exist_ok=True)
    names = ("full", "geometry_motion", "material_reflection_plus_noise")
    for name, image in zip(names, images):
        save_rgb(output / f"{name}_red_blue_white.png", image)
    save_rgb(
        output / "union_mask.png",
        np.repeat(mask[..., None].astype(np.float32), 3, axis=-1),
    )

    titles = (
        r"$E_{\mathrm{full}}$",
        r"$E_{\mathrm{geo}}$ (geometry motion)",
        r"$E_{\mathrm{material}}+E_{\mathrm{noise}}$",
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=15)
        ax.axis("off")
    fig.suptitle("Positive events: red   Negative events: blue", fontsize=14)
    fig.tight_layout()
    comparison = output / "event_components_comparison.png"
    fig.savefig(comparison, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "root": str(args.root.resolve()),
        "resolution_wh": [width, height],
        "normalization": {
            "transform": "linear" if args.linear else "log1p",
            "shared_percentile": args.percentile,
            "shared_limit": limit,
        },
        "mask": {
            "directory": str(mask_dir.resolve()),
            "mode": "union of all scene masks",
            "images": len(mask_paths),
            "threshold": args.mask_threshold,
            "foreground_pixels": int(mask.sum()),
            "flip_event_y_before_mask": bool(args.flip_event_y),
        },
        "raw_event_counts": {
            name: meta["count"] for name, meta in metadata.items()
        },
        "rendered": {
            "full": "full",
            "geometry": "geometry_motion",
            "non_geometry": "material_reflection + noise",
        },
    }
    with (output / "visualization_info.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved component visualizations to: {output.resolve()}")
    print(f"Comparison: {comparison.resolve()}")
    print(json.dumps(report["raw_event_counts"], indent=2))


if __name__ == "__main__":
    main()
