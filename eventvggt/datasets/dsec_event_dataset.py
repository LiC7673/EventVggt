"""DSEC loader for EventVGGT.

The implementation follows the official DSEC conventions:

* event timestamps are microseconds and ``t_offset`` is added for alignment;
* raw event coordinates are rectified with ``rectify_map[y, x]``;
* uint16 disparity is divided by 256 and zero is invalid;
* a polarity-separated, temporally interpolated voxel grid is returned.

RGB, events and depth must already share the rectified event-camera frame.
Official frame-camera PNGs (1440x1080) are deliberately not resized onto the
640x480 event camera because that would create a geometrically invalid sample.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
from PIL import Image

from eventvggt.datasets.base.base_event_dataset import BaseEventMultiViewDataset
from eventvggt.datasets.my_event_dataset import event_multiview_collate

try:  # DSEC h5 files may use the Blosc filter registered by hdf5plugin.
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover - only required for compressed releases
    hdf5plugin = None


def _read_numbers(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    values = np.asarray(values)
    if values.ndim > 1:
        values = values[:, -1]
    return values.reshape(-1)


def _find_files(root: Path, suffixes: Iterable[str]) -> list[Path]:
    suffixes = {value.lower() for value in suffixes}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_map(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            value = data["depth"] if "depth" in data else data[data.files[0]]
    elif suffix == ".exr":
        value = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ValueError(f"Failed to read map: {path}")
    value = np.asarray(value)
    if value.ndim == 3:
        value = value[..., 0]
    return value


def _map_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            return _image_size(path)
        value = _load_map(path)
        return int(value.shape[1]), int(value.shape[0])
    except (OSError, ValueError):
        return None


def _matrix(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "data" not in value:
            return None
        data = np.asarray(value["data"], dtype=np.float64)
        rows = int(value.get("rows", 0))
        cols = int(value.get("cols", 0))
        if rows > 0 and cols > 0 and data.size == rows * cols:
            return data.reshape(rows, cols)
        value = data
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 16:
        return array.reshape(4, 4)
    if array.size == 12:
        return array.reshape(3, 4)
    if array.size == 9:
        return array.reshape(3, 3)
    if array.size == 4:
        return array.reshape(-1)
    return None


def _walk_mapping(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _walk_mapping(child, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_mapping(child, path + (str(index),))


def _load_calibration(path: Optional[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "Q": None,
        "K": None,
        "P0": None,
        "P3": None,
        "focal_length": None,
        "baseline": None,
        "fb": None,
    }
    if path is None:
        return result

    # OpenCV FileStorage handles native OpenCV YAML without custom constructors.
    try:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        for parent in ("cams_03", "camRect0", "cam0", "camRect3", "cam3"):
            node = storage.getNode(parent)
            if node.empty():
                continue
            direct = None if node.isMap() else node.mat()
            if parent == "cams_03" and direct is not None and direct.size == 16:
                result["Q"] = np.asarray(direct, dtype=np.float64).reshape(4, 4)
            for key in ("Q", "q", "camera_matrix", "K", "P", "projection_matrix"):
                child = node.getNode(key)
                if not child.empty():
                    mat = child.mat()
                    if mat is not None:
                        if key.lower() == "q" and mat.size == 16:
                            result["Q"] = np.asarray(mat, dtype=np.float64).reshape(4, 4)
                        elif key.lower() in {"p", "projection_matrix"} and mat.size == 12:
                            projection = np.asarray(mat, dtype=np.float64).reshape(3, 4)
                            if parent.lower() in {"camrect0", "cam0"}:
                                result["P0"] = projection
                            elif parent.lower() in {"camrect3", "cam3"}:
                                result["P3"] = projection
                        elif result["K"] is None and parent.lower() in {"camrect0", "cam0"}:
                            mat = np.asarray(mat, dtype=np.float64)
                            result["K"] = mat[:3, :3]
        storage.release()
    except (cv2.error, SystemError, TypeError):
        pass

    try:
        import yaml

        class OpenCVLoader(yaml.SafeLoader):
            pass

        def construct_matrix(loader, node):
            return loader.construct_mapping(node, deep=True)

        OpenCVLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", construct_matrix)
        text = path.read_text(encoding="utf-8-sig")
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%YAML:"))
        data = yaml.load(text, Loader=OpenCVLoader)
        for keys, mapping in _walk_mapping(data):
            joined = "/".join(keys).lower()
            for key, value in mapping.items():
                key_lower = str(key).lower()
                mat = _matrix(value)
                if mat is not None:
                    if key_lower == "q" and mat.shape == (4, 4) and ("cams_03" in joined or result["Q"] is None):
                        result["Q"] = mat
                    if key_lower == "cams_03" and mat.shape == (4, 4):
                        result["Q"] = mat
                    if key_lower in {"p", "projection_matrix"} and mat.shape == (3, 4):
                        if "camrect0" in joined or joined.endswith("cam0"):
                            result["P0"] = mat
                        elif "camrect3" in joined or joined.endswith("cam3"):
                            result["P3"] = mat
                    if key_lower in {"k", "camera_matrix"}:
                        camera_matrix = None
                        if mat.ndim == 1 and mat.size == 4:
                            fx, fy, cx, cy = mat.tolist()
                            camera_matrix = np.array(
                                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                                dtype=np.float64,
                            )
                        elif mat.ndim == 2 and mat.shape[0] >= 3 and mat.shape[1] >= 3:
                            camera_matrix = mat[:3, :3]
                        if camera_matrix is not None and (
                            "camrect0" in joined or "cam0" in joined or result["K"] is None
                        ):
                            result["K"] = camera_matrix
                if key_lower in {"baseline", "base_line"}:
                    try:
                        result["baseline"] = abs(float(value))
                    except (TypeError, ValueError):
                        pass
                if key_lower in {"focal_length", "focal", "fx"} and ("cams_03" in joined or result["focal_length"] is None):
                    try:
                        result["focal_length"] = abs(float(value))
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass

    # Official DSEC YAML stores disparity_to_depth/cams_03 as a plain
    # nested 4x4 list. Keep a dependency-free fallback so a missing or
    # incompatible YAML parser cannot silently disable metric depth.
    if result["Q"] is None:
        try:
            raw_text = path.read_text(encoding="utf-8-sig")
            lines = raw_text.splitlines()
            for index, line in enumerate(lines):
                match = re.match(r"^(\s*)cams_03\s*:\s*(.*)$", line)
                if match is None:
                    continue
                base_indent = len(match.group(1))
                block = [match.group(2)]
                for child in lines[index + 1 :]:
                    if child.strip():
                        child_indent = len(child) - len(child.lstrip())
                        if child_indent < base_indent:
                            break
                        if child_indent == base_indent and not child.lstrip().startswith("-"):
                            break
                    block.append(child)
                values = re.findall(
                    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                    "\n".join(block),
                )
                if len(values) >= 16:
                    result["Q"] = np.asarray(values[:16], dtype=np.float64).reshape(4, 4)
                    break
        except (OSError, UnicodeError, ValueError):
            pass

    if result["K"] is None and result["Q"] is not None:
        q = result["Q"]
        result["K"] = np.array(
            [[q[2, 3], 0.0, -q[0, 3]], [0.0, q[2, 3], -q[1, 3]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if result["P0"] is not None:
        result["K"] = result["P0"][:, :3].copy()
    if result["P0"] is not None and result["P3"] is not None:
        p0 = result["P0"]
        p3 = result["P3"]
        fx = abs(float(p0[0, 0]))
        tx0 = float(p0[0, 3]) / float(p0[0, 0])
        tx3 = float(p3[0, 3]) / float(p3[0, 0])
        baseline = abs(tx3 - tx0)
        if fx > 0 and baseline > 0:
            result["focal_length"] = fx
            result["baseline"] = baseline
            result["fb"] = fx * baseline
    if result["fb"] is None and result["focal_length"] is not None and result["baseline"] is not None:
        result["fb"] = float(result["focal_length"]) * float(result["baseline"])
    return result


def _choose_timestamp_file(scene: Path, token: str) -> Optional[Path]:
    candidates = [
        path for path in _find_files(scene, {".txt", ".csv"})
        if "timestamp" in path.name.lower() and token in str(path.parent).lower()
    ]
    return candidates[0] if candidates else None


def _choose_rgb_files(scene: Path, target_size: Tuple[int, int]) -> tuple[list[Path], bool]:
    excluded = ("disparity", "depth", "flow", "mask", "label", "semantic")
    candidates = [
        path for path in _find_files(scene, {".png", ".jpg", ".jpeg"})
        if not any(token in str(path.parent).lower() for token in excluded)
    ]
    groups: Dict[Path, list[Path]] = {}
    for path in candidates:
        groups.setdefault(path.parent, []).append(path)

    scored = []
    for parent, files in groups.items():
        files.sort()
        try:
            size = _image_size(files[0])
        except OSError:
            continue
        parent_text = str(parent).lower()
        explicit = any(token in parent_text for token in ("event_aligned", "event-camera", "event_camera", "cam0"))
        score = (100 if explicit else 0) + (50 if size == target_size else 0) + min(len(files), 1000) / 1000.0
        scored.append((score, files, explicit, size))
    if not scored:
        return [], False
    explicit_groups = [item for item in scored if item[2]]
    if explicit_groups:
        _, files, explicit, size = max(
            explicit_groups,
            key=lambda item: (item[3] == target_size, len(item[1])),
        )
    else:
        _, files, explicit, size = max(scored, key=lambda item: item[0])
    # A 640x480 RGB export is accepted as an already warped custom export.
    # Untouched DSEC frame-camera images are 1440x1080 and fail below.
    return files, bool(explicit or size == target_size)


def _pair_by_stem_or_time(
    reference_files: Sequence[Path],
    reference_ts: np.ndarray,
    source_files: Sequence[Path],
    source_ts: Optional[np.ndarray],
) -> np.ndarray:
    by_stem = {path.stem: index for index, path in enumerate(source_files)}
    if all(path.stem in by_stem for path in reference_files):
        return np.asarray([by_stem[path.stem] for path in reference_files], dtype=np.int64)
    if source_ts is not None and len(source_ts) == len(source_files):
        indices = np.searchsorted(source_ts, reference_ts, side="left")
        indices = np.clip(indices, 0, len(source_ts) - 1)
        previous = np.clip(indices - 1, 0, len(source_ts) - 1)
        use_previous = np.abs(source_ts[previous] - reference_ts) <= np.abs(source_ts[indices] - reference_ts)
        return np.where(use_previous, previous, indices).astype(np.int64)
    if len(source_files) == len(reference_files):
        return np.arange(len(reference_files), dtype=np.int64)
    raise ValueError(
        f"Cannot align {len(source_files)} RGB frames with {len(reference_files)} supervision frames; "
        "provide matching filenames or image timestamps."
    )


class DSECEventDataset(BaseEventMultiViewDataset):
    def __init__(
        self,
        *args,
        ROOT: str,
        dsec_split: str,
        sequence_names=None,
        event_window_ms: float = 50.0,
        event_resize_bins: int = 10,
        clip_stride: int = 4,
        allow_unaligned_rgb: bool = False,
        depth_scale: float = 1.0,
        disparity_fx: Optional[float] = None,
        disparity_baseline: Optional[float] = None,
        max_depth: float = 80.0,
        **kwargs,
    ):
        self.ROOT = Path(ROOT)
        self.dsec_split = str(dsec_split)
        self.sequence_names = set(sequence_names or [])
        self.event_window_us = int(round(float(event_window_ms) * 1000.0))
        self.event_resize_bins = max(int(event_resize_bins), 1)
        self.clip_stride = max(int(clip_stride), 1)
        self.allow_unaligned_rgb = bool(allow_unaligned_rgb)
        self.depth_scale = float(depth_scale)
        self.disparity_fx = None if disparity_fx is None else float(disparity_fx)
        self.disparity_baseline = None if disparity_baseline is None else float(disparity_baseline)
        self.max_depth = float(max_depth)
        self.scene_data: Dict[str, Dict[str, Any]] = {}
        self.scenes: list[str] = []
        self.start_img_ids: list[tuple[str, int]] = []
        self.is_metric = True
        self.video = True
        super().__init__(*args, **kwargs)
        self._discover()

    def __len__(self):
        return len(self.start_img_ids)

    def get_stats(self):
        return f"{len(self)} DSEC clips from {len(self.scene_data)} {self.dsec_split} scenes"

    def get_active_scenes(self):
        return list(self.scene_data)

    @staticmethod
    def _event_paths(scene: Path) -> tuple[Path, Path]:
        event_candidates = [path for path in _find_files(scene, {".h5", ".hdf5"}) if path.name == "events.h5"]
        event_candidates.sort(key=lambda path: ("left" not in str(path.parent).lower(), len(path.parts)))
        rect_candidates = [
            path for path in _find_files(scene, {".h5", ".hdf5"})
            if "rectif" in path.name.lower() and ("left" in str(path.parent).lower() or len(event_candidates) == 1)
        ]
        if not event_candidates or not rect_candidates:
            raise FileNotFoundError(f"Missing events/left/events.h5 or rectify_map.h5 in {scene}")
        return event_candidates[0], rect_candidates[0]

    @staticmethod
    def _select_supervision_group(paths: Sequence[Path]) -> list[Path]:
        groups: Dict[Path, list[Path]] = {}
        for path in paths:
            groups.setdefault(path.parent, []).append(path)
        ranked = []
        for parent, files in groups.items():
            files.sort()
            size = _map_size(files[0])
            parent_text = str(parent).lower()
            camera_score = 30 if any(token in parent_text for token in ("event", "cam0", "left")) else 0
            resolution_score = 50 if size == (640, 480) else 0
            score = camera_score + resolution_score + min(len(files), 1000) / 1000.0
            ranked.append((score, files))
        return max(ranked, key=lambda item: item[0])[1] if ranked else []

    def _supervision(self, scene: Path):
        all_maps = _find_files(scene, {".npy", ".npz", ".exr", ".png", ".tif", ".tiff"})
        depth_candidates = [
            path for path in all_maps
            if "depth" in str(path).lower()
            and not any(token in str(path).lower() for token in ("preview", "visual", "blend", "normal"))
        ]
        disparity_candidates = [
            path for path in all_maps
            if "disparity" in str(path).lower()
            and not any(token in str(path).lower() for token in ("preview", "visual", "blend", "color"))
        ]
        depth_files = self._select_supervision_group(depth_candidates)
        disparity_files = self._select_supervision_group(disparity_candidates)
        files = sorted(depth_files or disparity_files)
        if not files:
            candidates = [str(path.relative_to(scene)) for path in all_maps[:20]]
            raise FileNotFoundError(
                f"No event-camera depth or disparity supervision found in {scene}; map candidates={candidates}"
            )
        kind = "depth" if depth_files else "disparity"
        timestamp_path = _choose_timestamp_file(scene, kind)
        if timestamp_path is None and kind == "disparity":
            candidate = scene / "disparity" / "timestamps.txt"
            timestamp_path = candidate if candidate.is_file() else None
        if timestamp_path is None:
            timestamp_candidates = [
                path for path in _find_files(scene, {".txt", ".csv"}) if "timestamp" in path.name.lower()
            ]
            timestamp_candidates.sort(
                key=lambda path: (
                    path.parent not in files[0].parents and path.parent != files[0].parent,
                    kind not in str(path.parent).lower(),
                    len(path.parts),
                )
            )
            timestamp_path = timestamp_candidates[0] if timestamp_candidates else None
        if timestamp_path is None:
            raise FileNotFoundError(f"No {kind} timestamps found in {scene}")
        timestamps = _read_numbers(timestamp_path)
        if len(timestamps) != len(files):
            raise ValueError(f"{scene.name}: {len(files)} {kind} files but {len(timestamps)} timestamps")
        return files, timestamps, kind

    def _calibration_files(self, scene: Path) -> list[Path]:
        candidates = []
        known_scene_names = {
            child.name
            for split_name in ("val", "test", "train")
            for child in ((self.ROOT / split_name).iterdir() if (self.ROOT / split_name).is_dir() else [])
            if child.is_dir()
        }
        search_roots = []
        for root in (scene, scene.parent, self.ROOT):
            if root.is_dir() and root not in search_roots:
                search_roots.append(root)
        for root in search_roots:
            for path in _find_files(root, {".yaml", ".yml"}):
                if "cam_to_cam" in path.name.lower() or "calib" in path.name.lower():
                    foreign_scenes = (set(path.parts) & known_scene_names) - {scene.name}
                    if foreign_scenes:
                        continue
                    candidates.append(path)
        unique = list(dict.fromkeys(candidates))
        unique.sort(
            key=lambda path: (
                scene.name not in str(path),
                "cam_to_cam" not in path.name.lower(),
                len(path.parts),
            )
        )
        if not unique:
            foreign = [
                path for path in _find_files(self.ROOT, {".yaml", ".yml"})
                if "cam_to_cam" in path.name.lower() or "calib" in path.name.lower()
            ]
            calibrated = []
            for path in foreign:
                parsed = _load_calibration(path)
                signature = parsed["Q"]
                if signature is None and parsed["fb"] is not None:
                    signature = np.asarray([parsed["fb"]], dtype=np.float64)
                if signature is not None:
                    calibrated.append((path, signature))
            if calibrated and all(
                first.shape == signature.shape and np.allclose(first, signature, rtol=1e-6, atol=1e-8)
                for _, signature in calibrated[1:]
                for first in [calibrated[0][1]]
            ):
                unique = [calibrated[0][0]]
                print(
                    f"DSEC {scene.name}: using shared disparity calibration from {unique[0]} "
                    "because all available sequence calibrations are identical."
                )
        return unique

    def _build_scene(self, scene: Path) -> Dict[str, Any]:
        event_h5, rectify_h5 = self._event_paths(scene)
        supervision, timestamps, supervision_kind = self._supervision(scene)
        target_size = _image_size(supervision[0]) if supervision[0].suffix.lower() == ".png" else None
        if target_size is None:
            first = _load_map(supervision[0])
            target_size = (int(first.shape[1]), int(first.shape[0]))

        rgb_files, explicitly_aligned = _choose_rgb_files(scene, target_size)
        if not rgb_files:
            raise FileNotFoundError(f"No RGB/frame images found in {scene}")
        rgb_size = _image_size(rgb_files[0])
        if rgb_size != target_size or (not explicitly_aligned and not self.allow_unaligned_rgb):
            raise ValueError(
                f"{scene.name}: RGB {rgb_size} and event-camera supervision {target_size} are not proven aligned. "
                f"Selected RGB={rgb_files[0].parent}; supervision={supervision[0].parent}. "
                "Export RGB into an event_aligned/cam0 directory, or set data.allow_unaligned_rgb=true only "
                "after independently verifying the warp."
            )

        image_timestamp_path = _choose_timestamp_file(scene, "image")
        image_ts = _read_numbers(image_timestamp_path) if image_timestamp_path else None
        rgb_for_frame = _pair_by_stem_or_time(supervision, timestamps, rgb_files, image_ts)
        calibration_files = self._calibration_files(scene)
        calibration = {"Q": None, "K": None, "fb": None}
        calibration_path = None
        calibration_debug = []
        for candidate in calibration_files:
            parsed = _load_calibration(candidate)
            calibration_debug.append(
                {
                    "path": str(candidate),
                    "Q": parsed["Q"] is not None,
                    "K": parsed["K"] is not None,
                    "P0": parsed["P0"] is not None,
                    "P3": parsed["P3"] is not None,
                    "focal_length": parsed["focal_length"],
                    "baseline": parsed["baseline"],
                    "fb": parsed["fb"],
                }
            )
            if parsed["Q"] is not None or parsed["fb"] is not None:
                calibration = parsed
                calibration_path = candidate
                break
        if supervision_kind == "disparity" and calibration["Q"] is None and calibration["fb"] is None:
            if self.disparity_fx is None or self.disparity_baseline is None:
                raise ValueError(
                    f"{scene.name}: disparity requires cams_03/Q, camRect0/P+camRect3/P, "
                    "focal_length+baseline, or explicit "
                    "data.disparity_fx and data.disparity_baseline. "
                    f"Parsed calibration candidates={calibration_debug[:8]}"
                )

        intrinsics = calibration["K"]
        if intrinsics is None:
            width, height = target_size
            focal = self.disparity_fx or float(max(width, height))
            intrinsics = np.array(
                [[focal, 0.0, (width - 1) * 0.5], [0.0, focal, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        with h5py.File(rectify_h5, "r") as handle:
            if "rectify_map" not in handle:
                raise KeyError(f"rectify_map missing in {rectify_h5}")
            rect_shape = tuple(handle["rectify_map"].shape)
        if rect_shape[:2] != (480, 640):
            raise ValueError(f"Unexpected DSEC rectify_map shape {rect_shape} in {rectify_h5}")

        return {
            "name": scene.name,
            "scene": scene,
            "event_h5": event_h5,
            "rectify_h5": rectify_h5,
            "supervision": supervision,
            "supervision_kind": supervision_kind,
            "timestamps": timestamps.astype(np.int64),
            "rgb_files": rgb_files,
            "rgb_for_frame": rgb_for_frame,
            "src_resolution": np.asarray(target_size, dtype=np.int32),
            "intrinsics": np.asarray(intrinsics, dtype=np.float32),
            "Q": calibration["Q"],
            "fb": calibration["fb"],
            "calibration_path": str(calibration_path) if calibration_path else None,
        }

    def _discover(self):
        split_root = self.ROOT / self.dsec_split
        if not split_root.is_dir():
            raise FileNotFoundError(f"DSEC split directory does not exist: {split_root}")
        errors = []
        for scene in sorted(path for path in split_root.iterdir() if path.is_dir()):
            if self.sequence_names and scene.name not in self.sequence_names:
                continue
            try:
                meta = self._build_scene(scene)
            except (FileNotFoundError, KeyError, ValueError, OSError) as error:
                errors.append(f"{scene.name}: {error}")
                continue
            count = len(meta["supervision"])
            starts = list(range(0, max(count - self.num_views + 1, 0), self.clip_stride))
            if not starts:
                errors.append(f"{scene.name}: only {count} frames for num_views={self.num_views}")
                continue
            meta["start_ids"] = starts
            self.scene_data[scene.name] = meta
            self.scenes.append(scene.name)
            self.start_img_ids.extend((scene.name, start) for start in starts)
        if not self.start_img_ids:
            detail = "\n  ".join(errors[:20])
            raise RuntimeError(f"No usable DSEC {self.dsec_split} clips under {split_root}.\n  {detail}")
        if errors:
            print("DSEC skipped scenes:\n  " + "\n  ".join(errors))

    def _depth(self, meta: Dict[str, Any], frame_idx: int):
        value = _load_map(meta["supervision"][frame_idx])
        if meta["supervision_kind"] == "depth":
            depth = value.astype(np.float32) * self.depth_scale
            valid = np.isfinite(depth) & (depth > 0.0)
        else:
            raw = value.astype(np.float32)
            valid = raw > 0.0
            disparity = raw / 256.0
            if meta["Q"] is not None:
                q = np.asarray(meta["Q"], dtype=np.float64)
                denominator = q[3, 2] * disparity + q[3, 3]
                depth = np.divide(
                    q[2, 3], denominator, out=np.zeros_like(disparity, dtype=np.float32), where=np.abs(denominator) > 1e-8
                )
                depth = np.abs(depth)
            elif meta.get("fb") is not None:
                depth = float(meta["fb"]) / np.maximum(disparity, 1e-6)
            else:
                depth = (self.disparity_fx * self.disparity_baseline) / np.maximum(disparity, 1e-6)
        valid &= np.isfinite(depth) & (depth > 0.0)
        if self.max_depth > 0:
            valid &= depth <= self.max_depth
        return np.where(valid, depth, 0.0).astype(np.float32), valid

    @staticmethod
    def _slice_events(handle: h5py.File, start_abs: int, end_abs: int):
        offset = int(np.asarray(handle["t_offset"][()]).reshape(-1)[0])
        start = int(start_abs) - offset
        end = int(end_abs) - offset
        mapping = handle["ms_to_idx"]
        raw_start_ms = start // 1000
        raw_end_ms = end // 1000 + 2
        start_ms = int(np.clip(raw_start_ms, 0, max(len(mapping) - 1, 0)))
        end_ms = int(np.clip(raw_end_ms, 0, max(len(mapping) - 1, 0)))
        i0 = int(mapping[start_ms])
        i1 = len(handle["events/t"]) if raw_end_ms >= len(mapping) else int(mapping[end_ms])
        t = np.asarray(handle["events/t"][i0:i1], dtype=np.int64)
        keep = (t >= start) & (t < end)
        return (
            np.asarray(handle["events/x"][i0:i1], dtype=np.int32)[keep],
            np.asarray(handle["events/y"][i0:i1], dtype=np.int32)[keep],
            np.asarray(handle["events/p"][i0:i1], dtype=np.float32)[keep],
            t[keep].astype(np.float64),
        )

    @staticmethod
    def _voxel(x, y, p, t, rectify_map, bins: int):
        height, width = rectify_map.shape[:2]
        if len(t) == 0:
            return np.zeros((2 * bins, height, width), dtype=np.float32)
        coordinates = rectify_map[y, x].astype(np.float32)
        xr, yr = coordinates[:, 0], coordinates[:, 1]
        duration = max(float(t[-1] - t[0]), 1.0)
        tr = (bins - 1) * (t.astype(np.float32) - float(t[0])) / duration
        x0, y0, t0 = np.floor(xr).astype(np.int32), np.floor(yr).astype(np.int32), np.floor(tr).astype(np.int32)
        voxel = np.zeros((2 * bins, height, width), dtype=np.float32)
        polarity_offset = np.where(p > 0, 0, bins).astype(np.int32)
        for xi in (x0, x0 + 1):
            wx = 1.0 - np.abs(xi.astype(np.float32) - xr)
            for yi in (y0, y0 + 1):
                wy = 1.0 - np.abs(yi.astype(np.float32) - yr)
                for ti in (t0, t0 + 1):
                    wt = 1.0 - np.abs(ti.astype(np.float32) - tr)
                    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & (ti >= 0) & (ti < bins)
                    if not np.any(valid):
                        continue
                    channel = ti[valid] + polarity_offset[valid]
                    flat = channel.astype(np.int64) * height * width + yi[valid].astype(np.int64) * width + xi[valid]
                    np.add.at(voxel.reshape(-1), flat, (wx[valid] * wy[valid] * wt[valid]).astype(np.float32))
        return voxel

    @staticmethod
    def _resize_voxel(voxel: np.ndarray, resolution: Tuple[int, int]):
        target_w, target_h = resolution
        source_h, source_w = voxel.shape[-2:]
        interpolation = cv2.INTER_AREA if target_w <= source_w and target_h <= source_h else cv2.INTER_LINEAR
        scale = float(source_w * source_h) / float(target_w * target_h)
        return np.stack(
            [cv2.resize(channel, (target_w, target_h), interpolation=interpolation) * scale for channel in voxel], axis=0
        ).astype(np.float32)

    def _get_views(self, idx, resolution, rng, num_views):
        scene_name, start = self.start_img_ids[idx]
        meta = self.scene_data[scene_name]
        views = []
        with h5py.File(meta["event_h5"], "r") as events_h5, h5py.File(meta["rectify_h5"], "r") as rect_h5:
            rectify_map = np.asarray(rect_h5["rectify_map"])
            for local_idx, frame_idx in enumerate(range(start, start + num_views)):
                timestamp = int(meta["timestamps"][frame_idx])
                image = _load_rgb(meta["rgb_files"][int(meta["rgb_for_frame"][frame_idx])])
                depth, mask = self._depth(meta, frame_idx)
                source_size = image.size
                if source_size != tuple(meta["src_resolution"]):
                    raise ValueError(f"Runtime RGB size changed in {scene_name}: {source_size}")
                x, y, p, t = self._slice_events(events_h5, timestamp - self.event_window_us, timestamp)
                voxel = self._voxel(x, y, p, t, rectify_map, self.event_resize_bins)

                image = self.resize_image(image, resolution)
                depth = self.resize_hw_map(depth, resolution, mode="bilinear", mask=mask)
                mask = self.resize_hw_map(mask.astype(np.float32), resolution, mode="bilinear") > 0.5
                voxel = self._resize_voxel(voxel, resolution)
                intrinsics = self.scale_intrinsics(meta["intrinsics"], source_size, resolution)
                width, height = image.size
                views.append(
                    {
                        "img": image,
                        "depthmap": depth.astype(np.float32),
                        "camera_pose": np.eye(4, dtype=np.float32),
                        "camera_intrinsics": intrinsics.astype(np.float32),
                        "mask": mask.astype(bool),
                        "event_xy": np.zeros((0, 2), dtype=np.int32),
                        "event_t": np.zeros((0,), dtype=np.float32),
                        "event_p": np.zeros((0,), dtype=np.float32),
                        "event_voxel": voxel,
                        "event_time_range": np.asarray([timestamp - self.event_window_us, timestamp], dtype=np.float32),
                        "event_resolution": np.asarray([width, height], dtype=np.int32),
                        "event_source_resolution": np.asarray([640, 480], dtype=np.int32),
                        "has_event": np.asarray(len(t) > 0, dtype=bool),
                        "pose_valid": np.asarray(False, dtype=bool),
                        "dataset": "dsec_event_dataset",
                        "scene_name": scene_name,
                        "frame_index": np.asarray(frame_idx, dtype=np.int64),
                        "timestamp_us": np.asarray(timestamp, dtype=np.int64),
                        "label": f"{scene_name}_{frame_idx:06d}",
                        "instance": f"{scene_name}_{frame_idx:06d}",
                        "is_metric": True,
                        "is_video": True,
                        "img_mask": np.asarray(True, dtype=bool),
                        "reset": np.asarray(local_idx == 0, dtype=bool),
                    }
                )
        return views


def get_dsec_dataset(
    root: str,
    *,
    split: str,
    num_views: int = 4,
    resolution=(518, 392),
    seed: int = 0,
    sequence_names=None,
    **kwargs,
):
    dsec_split = "val" if split in {"train", "val"} else "test"
    return DSECEventDataset(
        ROOT=root,
        dsec_split=dsec_split,
        split=split,
        num_views=num_views,
        resolution=resolution,
        seed=seed,
        sequence_names=sequence_names,
        **kwargs,
    )


__all__ = ["DSECEventDataset", "get_dsec_dataset", "event_multiview_collate"]
