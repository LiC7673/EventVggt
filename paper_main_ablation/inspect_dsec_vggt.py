"""Inspect a DSEC export before connecting it to EventVGGT training.

Official DSEC events, global-shutter images, and event-camera disparity do not
automatically share one pixel coordinate frame. This tool reports the actual
local layout and refuses to label a sequence "VGGT ready" unless an explicit
event-camera alignment marker or clearly named aligned image directory exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
from PIL import Image


EVENT_KEYS = (
    "events/p",
    "events/t",
    "events/x",
    "events/y",
    "ms_to_idx",
    "t_offset",
)


def _files(root: Path, suffixes: Iterable[str]):
    suffixes = {suffix.lower() for suffix in suffixes}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def _relative(paths, root, limit=12):
    values = []
    for path in paths[:limit]:
        try:
            values.append(str(path.relative_to(root)))
        except ValueError:
            values.append(str(path))
    return values


def _sample_image_info(paths):
    if not paths:
        return None
    try:
        with Image.open(paths[0]) as image:
            return {"path": str(paths[0]), "width": image.width, "height": image.height, "mode": image.mode}
    except Exception as error:
        return {"path": str(paths[0]), "error": str(error)}


def _preferred_camera_files(paths, *, aligned=False, supervision=False):
    if not paths:
        return []
    scored = []
    for path in paths:
        text = str(path.parent).lower()
        score = 0
        if aligned and any(token in text for token in ("event_aligned", "cam0", "event_camera")):
            score += 100
        if supervision and any(token in text for token in ("disparity_event", "event", "cam0")):
            score += 100
        try:
            with Image.open(path) as image:
                if image.size == (640, 480):
                    score += 50
        except Exception:
            pass
        scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    best_parent = scored[0][1].parent
    return [path for _, path in scored if path.parent == best_parent]


def _inspect_event_h5(path: Path):
    report = {"path": str(path), "keys": [], "official_keys_present": False}
    try:
        with h5py.File(path, "r") as handle:
            keys = []
            handle.visit(keys.append)
            report["keys"] = keys[:80]
            report["official_keys_present"] = all(key in handle for key in EVENT_KEYS)
            for key in ("events/t", "events/x", "events/y", "events/p"):
                if key in handle:
                    report[f"{key}_shape"] = list(handle[key].shape)
            if "t_offset" in handle:
                report["t_offset_shape"] = list(handle["t_offset"].shape)
    except Exception as error:
        report["error"] = str(error)
    return report


def inspect_scene(scene: Path, dataset_root: Path | None = None):
    all_png = _files(scene, {".png"})
    image_files = [
        path for path in all_png
        if any(token in str(path.parent).lower() for token in ("image", "rgb", "frame"))
        and not any(token in str(path.parent).lower() for token in ("disparity", "depth", "flow"))
    ]
    supervision_files = [
        path for path in _files(scene, {".png", ".npy", ".npz", ".exr"})
        if any(token in str(path.parent).lower() for token in ("disparity", "depth"))
    ]
    h5_files = _files(scene, {".h5", ".hdf5"})
    event_h5 = [path for path in h5_files if "event" in path.name.lower()]
    rectify_maps = [path for path in h5_files if "rectif" in path.name.lower() or "rectif" in str(path.parent).lower()]
    calibration = [
        path for path in _files(scene, {".yaml", ".yml", ".json"})
        if any(token in path.name.lower() for token in ("calib", "cam_to_cam", "intrinsic", "extrinsic"))
    ]
    if not calibration and dataset_root is not None:
        calibration = [
            path for path in _files(dataset_root, {".yaml", ".yml", ".json"})
            if any(token in path.name.lower() for token in ("calib", "cam_to_cam", "intrinsic", "extrinsic"))
            and not any(
                part in {child.name for split in ("val", "test", "train")
                         for child in ((dataset_root / split).iterdir() if (dataset_root / split).is_dir() else [])
                         if child.is_dir()} - {scene.name}
                for part in path.parts
            )
        ]
    timestamp_files = [
        path for path in _files(scene, {".txt", ".csv"}) if "timestamp" in path.name.lower()
    ]
    pose_files = [
        path for path in _files(scene, {".txt", ".csv", ".npy", ".npz", ".json"})
        if any(token in path.name.lower() for token in ("pose", "odometry", "trajectory"))
    ]
    alignment_markers = [
        path for path in scene.rglob("*")
        if path.is_file()
        and any(token in path.name.lower() for token in ("vggt_alignment", "event_aligned", "cam0_aligned"))
    ]
    aligned_image_paths = [
        path for path in image_files
        if any(token in str(path.parent).lower() for token in ("event_aligned", "cam0", "event_camera"))
    ]

    image_sample_files = _preferred_camera_files(image_files, aligned=True)
    supervision_pngs = [path for path in supervision_files if path.suffix.lower() == ".png"]
    supervision_sample_files = _preferred_camera_files(supervision_pngs, supervision=True)
    image_info = _sample_image_info(image_sample_files)
    supervision_info = _sample_image_info(supervision_sample_files)
    same_resolution = bool(
        image_info
        and supervision_info
        and "width" in image_info
        and "width" in supervision_info
        and image_info["width"] == supervision_info["width"]
        and image_info["height"] == supervision_info["height"]
    )
    event_resolution_match = bool(
        same_resolution
        and image_info
        and image_info.get("width") == 640
        and image_info.get("height") == 480
    )
    explicit_alignment = bool(alignment_markers or aligned_image_paths)
    usable_alignment = explicit_alignment or event_resolution_match
    event_reports = [_inspect_event_h5(path) for path in event_h5[:2]]
    official_event_ok = any(report.get("official_keys_present", False) for report in event_reports)
    has_depth = any("depth" in str(path.parent).lower() for path in supervision_files)
    has_disparity = any("disparity" in str(path.parent).lower() for path in supervision_files)

    blockers = []
    warnings = []
    if not official_event_ok:
        blockers.append("official DSEC event h5 keys were not detected")
    if not image_files:
        blockers.append("no RGB/frame images detected")
    if not supervision_files:
        blockers.append("no depth/disparity supervision detected")
    if has_disparity and not calibration:
        blockers.append("disparity exists but calibration for depth conversion was not detected")
    if not rectify_maps:
        blockers.append("event rectify map was not detected")
    if not same_resolution:
        blockers.append("RGB and depth/disparity sample resolutions differ")
    if not usable_alignment:
        blockers.append("no explicit RGB-to-event-camera alignment marker was detected")
    elif not explicit_alignment:
        warnings.append("assuming the custom 640x480 RGB export is already aligned to the event camera")

    return {
        "scene": scene.name,
        "path": str(scene),
        "event_h5": event_reports,
        "rectify_maps": _relative(rectify_maps, scene),
        "image_count": len(image_files),
        "image_examples": _relative(image_files, scene),
        "image_sample": image_info,
        "selected_image_directory": str(image_sample_files[0].parent) if image_sample_files else None,
        "supervision_count": len(supervision_files),
        "supervision_examples": _relative(supervision_files, scene),
        "supervision_sample": supervision_info,
        "selected_supervision_directory": (
            str(supervision_sample_files[0].parent) if supervision_sample_files else None
        ),
        "has_depth": has_depth,
        "has_disparity": has_disparity,
        "calibration": _relative(calibration, scene),
        "timestamps": _relative(timestamp_files, scene),
        "poses": _relative(pose_files, scene),
        "alignment_markers": _relative(alignment_markers, scene),
        "same_rgb_supervision_resolution": same_resolution,
        "explicit_event_camera_alignment": explicit_alignment,
        "event_resolution_alignment_assumed": event_resolution_match and not explicit_alignment,
        "pose_supervision_available": bool(pose_files),
        "direct_vggt_ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    parser.add_argument("--output", default="abl_event_exp/dsec_preflight/layout_report.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {"root": str(root), "splits": {}, "all_scenes_ready": True}
    for split in ("val", "test"):
        split_root = root / split
        scenes = sorted(path for path in split_root.iterdir() if path.is_dir()) if split_root.is_dir() else []
        reports = [inspect_scene(scene, root) for scene in scenes]
        result["splits"][split] = {
            "scene_count": len(reports),
            "ready_count": sum(report["direct_vggt_ready"] for report in reports),
            "scenes": reports,
        }
        result["all_scenes_ready"] &= bool(reports) and all(
            report["direct_vggt_ready"] for report in reports
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(f"DSEC root: {root}")
    for split, split_report in result["splits"].items():
        print(
            f"{split}: scenes={split_report['scene_count']} "
            f"direct_vggt_ready={split_report['ready_count']}"
        )
        for scene in split_report["scenes"]:
            status = "READY" if scene["direct_vggt_ready"] else "BLOCKED"
            print(f"  [{status}] {scene['scene']}")
            for blocker in scene["blockers"]:
                print(f"    - {blocker}")
            for warning in scene.get("warnings", []):
                print(f"    ! {warning}")
    print(f"Report saved to {output}")
    if args.strict and not result["all_scenes_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
