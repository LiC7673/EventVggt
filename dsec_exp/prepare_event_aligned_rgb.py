"""Rectify DSEC-Detection remapped RGB into the DSEC rectified event frame.

DSEC-Detection distributes RGB frames remapped into the *distorted* left event
camera. DSEC ``rectify_map`` maps distorted event pixels to rectified cam0
pixels. This tool numerically inverts that map once per scene, then remaps all
RGB frames to 640x480 rectified cam0 coordinates.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def files(root: Path, suffixes):
    suffixes = {value.lower() for value in suffixes}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def find_rectify_map(scene: Path) -> Path:
    candidates = [path for path in files(scene, {".h5", ".hdf5"}) if "rectif" in path.name.lower()]
    candidates.sort(key=lambda path: ("left" not in str(path.parent).lower(), len(path.parts)))
    if not candidates:
        raise FileNotFoundError(f"No rectify_map.h5 below {scene}")
    return candidates[0]


def find_distorted_images(scene: Path, remapped_root: Path) -> list[Path]:
    local_dirs = [
        path for path in scene.rglob("*")
        if path.is_dir() and "distorted" in path.name.lower()
    ]
    external_scene_dirs = [path for path in remapped_root.rglob(scene.name) if path.is_dir()]
    candidate_dirs = local_dirs[:]
    for external_scene in external_scene_dirs:
        candidate_dirs.extend(
            path for path in external_scene.rglob("*") if path.is_dir() and "distorted" in path.name.lower()
        )
    groups = []
    for directory in candidate_dirs:
        images = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            groups.append(images)
    if not groups:
        raise FileNotFoundError(
            f"No distorted event-view RGB found for {scene.name} below {scene} or {remapped_root}"
        )
    return max(groups, key=len)


def inverse_rectify_map(rectify_map: np.ndarray):
    height, width = rectify_map.shape[:2]
    raw_y, raw_x = np.mgrid[0:height, 0:width].astype(np.float32)
    rect_x = rectify_map[..., 0].astype(np.float32)
    rect_y = rectify_map[..., 1].astype(np.float32)
    x0 = np.floor(rect_x).astype(np.int32)
    y0 = np.floor(rect_y).astype(np.int32)

    weight_sum = np.zeros((height, width), dtype=np.float32)
    raw_x_sum = np.zeros_like(weight_sum)
    raw_y_sum = np.zeros_like(weight_sum)
    for target_x in (x0, x0 + 1):
        wx = 1.0 - np.abs(target_x.astype(np.float32) - rect_x)
        for target_y in (y0, y0 + 1):
            wy = 1.0 - np.abs(target_y.astype(np.float32) - rect_y)
            weight = wx * wy
            valid = (
                (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height) & (weight > 0)
            )
            flat = target_y[valid].astype(np.int64) * width + target_x[valid].astype(np.int64)
            np.add.at(weight_sum.reshape(-1), flat, weight[valid])
            np.add.at(raw_x_sum.reshape(-1), flat, weight[valid] * raw_x[valid])
            np.add.at(raw_y_sum.reshape(-1), flat, weight[valid] * raw_y[valid])

    known = weight_sum > 1e-6
    inverse_x = np.divide(raw_x_sum, weight_sum, out=np.zeros_like(raw_x_sum), where=known)
    inverse_y = np.divide(raw_y_sum, weight_sum, out=np.zeros_like(raw_y_sum), where=known)
    holes = (~known).astype(np.uint8)
    if holes.any():
        inverse_x = cv2.inpaint(inverse_x, holes, 5.0, cv2.INPAINT_NS)
        inverse_y = cv2.inpaint(inverse_y, holes, 5.0, cv2.INPAINT_NS)

    # Verify rectified -> raw -> rectified round-trip accuracy.
    roundtrip_x = cv2.remap(rect_x, inverse_x, inverse_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    roundtrip_y = cv2.remap(rect_y, inverse_x, inverse_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    target_y, target_x = np.mgrid[0:height, 0:width].astype(np.float32)
    error = np.sqrt((roundtrip_x - target_x) ** 2 + (roundtrip_y - target_y) ** 2)
    interior = known & (inverse_x >= 1) & (inverse_x < width - 2) & (inverse_y >= 1) & (inverse_y < height - 2)
    values = error[interior]
    stats = {
        "inverse_known_ratio": float(known.mean()),
        "roundtrip_median_px": float(np.median(values)) if values.size else float("inf"),
        "roundtrip_p95_px": float(np.percentile(values, 95)) if values.size else float("inf"),
    }
    if not np.isfinite(stats["roundtrip_p95_px"]) or stats["roundtrip_p95_px"] > 1.5:
        raise RuntimeError(f"Rectify-map inversion failed quality check: {stats}")
    return inverse_x, inverse_y, stats


def process_scene(scene: Path, remapped_root: Path, workers: int, force: bool):
    rectify_path = find_rectify_map(scene)
    source_images = find_distorted_images(scene, remapped_root)
    output = scene / "images" / "event_aligned"
    marker = output / "vggt_alignment.json"
    if marker.is_file() and not force:
        existing = [path for path in output.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        if len(existing) == len(source_images):
            print(f"[skip] {scene.name}: {len(existing)} aligned RGB frames")
            return

    with h5py.File(rectify_path, "r") as handle:
        if "rectify_map" not in handle:
            raise KeyError(f"rectify_map missing from {rectify_path}")
        rectify_map = np.asarray(handle["rectify_map"], dtype=np.float32)
    if rectify_map.shape != (480, 640, 2):
        raise ValueError(f"Unexpected rectify map shape {rectify_map.shape}: {rectify_path}")
    inverse_x, inverse_y, stats = inverse_rectify_map(rectify_map)
    output.mkdir(parents=True, exist_ok=True)

    def convert(source: Path):
        destination = output / f"{source.stem}.png"
        if destination.is_file() and not force:
            return
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read {source}")
        if image.shape[:2] != (480, 640):
            raise ValueError(f"Expected distorted event-view RGB at 640x480, got {image.shape[:2]}: {source}")
        aligned = cv2.remap(
            image,
            inverse_x,
            inverse_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if not cv2.imwrite(str(destination), aligned):
            raise OSError(f"Failed to write {destination}")

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        list(executor.map(convert, source_images))
    payload = {
        "scene": scene.name,
        "source_directory": str(source_images[0].parent),
        "rectify_map": str(rectify_path),
        "output_directory": str(output),
        "frame_count": len(source_images),
        "coordinate_frame": "rectified_left_event_camera_camRect0",
        **stats,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {scene.name}: {len(source_images)} frames -> {output}; {stats}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    parser.add_argument("--remapped-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    remapped_root = Path(args.remapped_root)
    failures = []
    count = 0
    for split in ("val", "test"):
        split_root = root / split
        if not split_root.is_dir():
            continue
        for scene in sorted(path for path in split_root.iterdir() if path.is_dir()):
            try:
                process_scene(scene, remapped_root, args.workers, args.force)
                count += 1
            except Exception as error:
                failures.append(f"{scene.name}: {error}")
                print(f"[failed] {scene.name}: {error}")
    if failures:
        raise RuntimeError("Failed DSEC RGB alignment:\n  " + "\n  ".join(failures))
    print(f"Prepared event-aligned RGB for {count} scenes")


if __name__ == "__main__":
    main()

