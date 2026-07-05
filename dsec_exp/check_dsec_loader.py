"""Load one DSEC clip and save an alignment/voxel sanity preview."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eventvggt.datasets.dsec_event_dataset import get_dsec_dataset


def color_event(voxel):
    bins = voxel.shape[0] // 2
    pos = np.log1p(np.maximum(voxel[:bins], 0).sum(0))
    neg = np.log1p(np.maximum(voxel[bins : 2 * bins], 0).sum(0))
    scale = max(float(np.percentile(np.r_[pos.ravel(), neg.ravel()], 99.5)), 1e-6)
    rgb = np.zeros((*pos.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(pos / scale, 0, 1)
    rgb[..., 2] = np.clip(neg / scale, 0, 1)
    return (255 * rgb).astype(np.uint8)


def depth_image(depth, mask):
    values = depth[mask]
    lo, hi = np.percentile(values, [2, 98]) if values.size else (0, 1)
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    image = (255 * normalized).astype(np.uint8)
    image[~mask] = 0
    return np.repeat(image[..., None], 3, axis=-1)


def labeled(array, label):
    image = Image.fromarray(array).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 24), "black")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), label, fill="white")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--output", default="dsec_exp/results/loader_check")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    dataset = get_dsec_dataset(
        args.root,
        split=args.split,
        num_views=args.num_views,
        resolution=(518, 392),
        event_resize_bins=args.bins,
        clip_stride=args.num_views,
    )
    views = dataset[0]
    panels = []
    report = {"scenes": dataset.get_active_scenes(), "clip_count": len(dataset), "views": []}
    for index, view in enumerate(views):
        image_value = view["img"]
        if hasattr(image_value, "detach"):
            rgb = image_value.detach().cpu().float().permute(1, 2, 0).numpy()
            rgb = (255.0 * np.clip((rgb + 1.0) * 0.5, 0.0, 1.0)).astype(np.uint8)
        else:
            rgb = np.asarray(image_value)
        event = view["event_voxel"]
        depth = view["depthmap"]
        mask = view["valid_mask"]
        panels.extend(
            [labeled(rgb, f"view{index} RGB"), labeled(color_event(event), f"view{index} event"), labeled(depth_image(depth, mask), f"view{index} depth")]
        )
        report["views"].append(
            {
                "label": view["label"],
                "event_shape": list(event.shape),
                "event_abs_mean": float(np.abs(event).mean()),
                "event_nonzero_ratio": float(np.mean(event != 0)),
                "valid_depth_ratio": float(mask.mean()),
                "depth_min": float(depth[mask].min()) if mask.any() else None,
                "depth_max": float(depth[mask].max()) if mask.any() else None,
            }
        )
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black")
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / "first_clip.png")
    (output / "first_clip.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
