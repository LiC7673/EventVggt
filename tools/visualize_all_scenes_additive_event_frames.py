"""Render 120 masked additive-event frames for every reflective scene.

Frame alignment follows ``esim_event_adaptive_blender (2).py`` exactly:
120 RGB frames at 120 FPS produce 119 event intervals.  To match the project
dataloader, output frame_0001 is empty and frame_0002 contains events generated
between RGB frames 1 and 2, continuing through frame_0120.

For every frame this script renders full, geometry_motion, and the merged
material_reflection+noise stream. Positive/negative events are red/blue and
zero-event pixels are white. The same-numbered scene mask is applied.
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

from visualize_additive_event_components import (
    inspect_h5,
    polarity_rgb,
    save_rgb,
    shared_limit,
)


DEFAULT_ROOT = Path(r"F:\TreeOBJ\reflective_raw")
BRANCHES = ("full", "geometry_motion", "material_reflection", "noise")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--scene-names",
        nargs="*",
        default=None,
        help="Optional subset for debugging; default traverses every scene.",
    )
    p.add_argument("--event-dir", default="events_additive")
    p.add_argument("--mask-dir", default="Mask")
    p.add_argument("--output-name", default="vis_event_components_120")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=float, default=120.0)
    p.add_argument("--mask-threshold", type=int, default=250)
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--linear", action="store_true")
    p.add_argument(
        "--flip-event-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert Blender bottom-left event rows to top-left PNG rows.",
    )
    p.add_argument(
        "--save-individual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def numeric_key(path: Path):
    digits = "".join(c if c.isdigit() else " " for c in path.stem).split()
    return (int(digits[-1]) if digits else 10**9, path.name)


def discover_masks(mask_dir: Path, frames: int):
    if not mask_dir.is_dir():
        raise FileNotFoundError(mask_dir)
    candidates = sorted(
        (
            p for p in mask_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        key=numeric_key,
    )
    by_number = {}
    for path in candidates:
        number = numeric_key(path)[0]
        if number < 10**9:
            by_number.setdefault(number, path)
    result = []
    for frame in range(1, frames + 1):
        path = by_number.get(frame)
        if path is None and frame - 1 < len(candidates):
            path = candidates[frame - 1]
        if path is None:
            raise FileNotFoundError(
                f"No mask for frame {frame:04d} under {mask_dir}"
            )
        result.append(path)
    return result


def load_mask(path: Path, width: int, height: int, threshold: int):
    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    rgb = np.asarray(image, dtype=np.uint8)
    return np.all(rgb >= int(threshold), axis=-1)


class BranchReader:
    def __init__(self, meta, frames, fps):
        self.meta = meta
        self.h5 = h5py.File(meta["path"], "r")
        self.events = self.h5["events"]
        self.frames = int(frames)
        self.fps = float(fps)
        self.boundaries = np.arange(frames, dtype=np.float64) / self.fps
        self.indices = np.asarray(
            [self._lower_bound(t) for t in self.boundaries], dtype=np.int64
        )

    def _lower_bound(self, timestamp):
        """np.searchsorted equivalent without loading the full timestamp array."""
        column = self.meta["columns"]["t"]
        low, high = 0, len(self.events)
        while low < high:
            mid = (low + high) // 2
            if float(self.events[mid, column]) < timestamp:
                low = mid + 1
            else:
                high = mid
        return low

    def frame_counts(self, output_frame, width, height, flip_y):
        # The first RGB frame has no preceding interval.
        if output_frame == 1:
            zero = np.zeros((height, width), dtype=np.uint32)
            return zero, zero.copy(), 0
        interval = output_frame - 2
        start, end = self.indices[interval], self.indices[interval + 1]
        data = np.asarray(self.events[start:end])
        columns = self.meta["columns"]
        raw_x, raw_y = data[:, columns["x"]], data[:, columns["y"]]
        polarity = data[:, columns["p"]]
        valid = (
            np.isfinite(raw_x) & np.isfinite(raw_y) & np.isfinite(polarity)
            & (raw_x >= 0) & (raw_x < width)
            & (raw_y >= 0) & (raw_y < height)
        )
        x = raw_x[valid].astype(np.int64)
        y = raw_y[valid].astype(np.int64)
        polarity = polarity[valid]
        if flip_y:
            y = height - 1 - y
        positive = np.zeros((height, width), dtype=np.uint32)
        negative = np.zeros((height, width), dtype=np.uint32)
        np.add.at(positive, (y[polarity > 0], x[polarity > 0]), 1)
        np.add.at(negative, (y[polarity <= 0], x[polarity <= 0]), 1)
        return positive, negative, int(valid.sum())

    def close(self):
        self.h5.close()


def save_comparison(path, images, scene_name, frame, dpi):
    titles = (
        r"$E_{\mathrm{full}}$",
        r"$E_{\mathrm{geo}}$",
        r"$E_{\mathrm{material}}+E_{\mathrm{noise}}$",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), facecolor="white")
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    fig.suptitle(
        f"{scene_name} | frame {frame:04d} | + red / - blue",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_scene(scene: Path, args):
    event_root = scene / args.event_dir
    paths = {
        branch: event_root / branch / "events.h5" for branch in BRANCHES
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing event branches:\n" + "\n".join(missing))

    output = event_root / args.output_name
    complete = output / "visualization_info.json"
    if complete.is_file() and not args.overwrite:
        print(f"[skip complete] {scene.name}", flush=True)
        return "skipped"

    metadata = {name: inspect_h5(path) for name, path in paths.items()}
    width = max(meta["width"] for meta in metadata.values())
    height = max(meta["height"] for meta in metadata.values())
    for name, meta in metadata.items():
        if (meta["width"], meta["height"]) != (width, height):
            raise ValueError(
                f"{scene.name}/{name} resolution mismatch: "
                f"{meta['width']}x{meta['height']} vs {width}x{height}"
            )
    masks = discover_masks(scene / args.mask_dir, args.frames)
    readers = {
        name: BranchReader(meta, args.frames, args.fps)
        for name, meta in metadata.items()
    }
    event_totals = {name: 0 for name in BRANCHES}
    try:
        for frame in range(1, args.frames + 1):
            mask = load_mask(
                masks[frame - 1], width, height, args.mask_threshold
            )
            raw = {}
            for name, reader in readers.items():
                pos, neg, count = reader.frame_counts(
                    frame, width, height, args.flip_event_y
                )
                raw[name] = (pos * mask, neg * mask)
                event_totals[name] += count
            non_geometry = (
                raw["material_reflection"][0] + raw["noise"][0],
                raw["material_reflection"][1] + raw["noise"][1],
            )
            pairs = (raw["full"], raw["geometry_motion"], non_geometry)
            limit = shared_limit(
                pairs, args.percentile, use_log=not args.linear
            )
            images = [
                polarity_rgb(pair, limit, use_log=not args.linear)
                for pair in pairs
            ]
            filename = f"frame_{frame:04d}.png"
            if args.save_individual:
                for folder, image in zip(
                    ("full", "geometry_motion", "material_reflection_plus_noise"),
                    images,
                ):
                    save_rgb(output / folder / filename, image)
            save_comparison(
                output / "comparison" / filename,
                images, scene.name, frame, args.dpi,
            )
            if frame == 1 or frame % 10 == 0 or frame == args.frames:
                print(
                    f"  [{scene.name}] {frame:03d}/{args.frames}",
                    flush=True,
                )
    finally:
        for reader in readers.values():
            reader.close()

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scene": scene.name,
        "source_generator": r"F:\TreeOBJ\esim_event_adaptive_blender (2).py",
        "event_root": str(event_root.resolve()),
        "frames": args.frames,
        "fps": args.fps,
        "alignment": {
            "frame_0001": "empty (no preceding event interval)",
            "frame_0002": "[0/FPS, 1/FPS), generated between RGB frames 1 and 2",
            "frame_0120": "[118/FPS, 119/FPS), generated between RGB frames 119 and 120",
        },
        "resolution_wh": [width, height],
        "mask": {
            "directory": str((scene / args.mask_dir).resolve()),
            "threshold": args.mask_threshold,
            "same_numbered_mask_per_output_frame": True,
        },
        "polarity": {"positive": "red", "negative": "blue", "none": "white"},
        "flip_event_y": bool(args.flip_event_y),
        "raw_valid_events_read": event_totals,
    }
    with complete.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return "done"


def main():
    args = parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    scenes = sorted(p for p in args.root.iterdir() if p.is_dir())
    if args.scene_names:
        requested = set(args.scene_names)
        scenes = [scene for scene in scenes if scene.name in requested]
        missing = sorted(requested - {scene.name for scene in scenes})
        if missing:
            raise FileNotFoundError(f"Requested scenes not found: {missing}")
    done = skipped = failed = 0
    failures = []
    print(f"Found {len(scenes)} scene directories under {args.root}", flush=True)
    for index, scene in enumerate(scenes, 1):
        print(f"[scene {index}/{len(scenes)}] {scene.name}", flush=True)
        try:
            status = process_scene(scene, args)
            if status == "skipped":
                skipped += 1
            else:
                done += 1
        except Exception as error:
            failed += 1
            failures.append({"scene": scene.name, "error": repr(error)})
            print(f"  [failed] {error}", flush=True)
            if args.strict:
                raise

    summary = {
        "root": str(args.root.resolve()),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
    }
    summary_path = args.root / "event_component_visualization_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
