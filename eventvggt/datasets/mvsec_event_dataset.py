import os
import os.path as osp
import re

import h5py
import numpy as np
from PIL import Image

from eventvggt.datasets.base.base_event_dataset import BaseEventMultiViewDataset
from eventvggt.datasets.my_event_dataset import MyEventDataset, event_multiview_collate


def _sequence_key(path):
    stem = osp.splitext(osp.basename(path))[0]
    stem = re.sub(r"(_)?(data|gt|groundtruth|ground_truth)$", "", stem, flags=re.IGNORECASE)
    return stem


def _candidate_h5_files(root):
    if osp.isfile(root) and osp.splitext(root)[1].lower() in {".h5", ".hdf5"}:
        return [root]

    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if osp.splitext(name)[1].lower() in {".h5", ".hdf5"}:
                files.append(osp.join(dirpath, name))
    return sorted(files)


def _normalize_h5_path(path):
    return path.strip("/")


def _h5_obj(h5_file, path):
    path = _normalize_h5_path(path)
    return h5_file[path] if path in h5_file else None


def _as_dataset(obj):
    if isinstance(obj, h5py.Dataset):
        return obj
    if isinstance(obj, h5py.Group):
        for name in ("data", "value", "values", "images", "image", "events"):
            if name in obj and isinstance(obj[name], h5py.Dataset):
                return obj[name]
    return None


def _find_dataset(h5_file, candidates):
    for path in candidates:
        ds = _as_dataset(_h5_obj(h5_file, path))
        if ds is not None:
            return path, ds
    return None, None


def _find_timestamps(h5_file, base_paths):
    candidates = []
    for path in base_paths:
        path = _normalize_h5_path(path)
        candidates.extend(
            [
                f"{path}_ts",
                f"{path}_timestamps",
                f"{path}/ts",
                f"{path}/time",
                f"{path}/times",
                f"{path}/timestamp",
                f"{path}/timestamps",
            ]
        )

    _, ds = _find_dataset(h5_file, candidates)
    if ds is not None:
        return np.asarray(ds[:], dtype=np.float64).reshape(-1)

    for path in base_paths:
        obj = _h5_obj(h5_file, path)
        if obj is None:
            continue
        for attr in ("ts", "time", "times", "timestamp", "timestamps"):
            if attr in obj.attrs:
                return np.asarray(obj.attrs[attr], dtype=np.float64).reshape(-1)
    return None


def _nearest_index(timestamps, target):
    if timestamps is None or len(timestamps) == 0:
        return 0
    idx = int(np.searchsorted(timestamps, target, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    left = timestamps[idx - 1]
    right = timestamps[idx]
    return idx - 1 if abs(target - left) <= abs(right - target) else idx


def _image_to_rgb(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.dtype != np.uint8:
        finite = np.isfinite(image)
        if finite.any():
            lo = float(np.nanmin(image[finite]))
            hi = float(np.nanmax(image[finite]))
            if hi > lo:
                image = (np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0).round()
            else:
                image = np.zeros_like(image)
        else:
            image = np.zeros_like(image)
        image = image.astype(np.uint8)

    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(image[..., :3]).convert("RGB")


def _clean_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    return depth, valid.astype(bool)


def _quat_to_rot(qx, qy, qz, qw):
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = quat / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _pose_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float32)
    pose = np.squeeze(pose)
    if pose.shape == (4, 4):
        return pose.astype(np.float32)
    if pose.size >= 16:
        return pose.reshape(-1)[-16:].reshape(4, 4).astype(np.float32)
    if pose.size >= 7:
        values = pose.reshape(-1)[-7:].astype(np.float32)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = _quat_to_rot(values[3], values[4], values[5], values[6])
        mat[:3, 3] = values[:3]
        return mat
    return np.eye(4, dtype=np.float32)


class MVSECEventDataset(BaseEventMultiViewDataset):
    """MVSEC HDF5 loader that returns EventVGGT-style multi-view samples.

    Expected MVSEC-style paths are searched under paired files:
      *_data.hdf5: davis/{left,right}/events, image_raw,
                  image_raw_ts, image_raw_event_inds
      *_gt.hdf5:   davis/{left,right}/depth_image_rect or depth_image_raw,
                  pose or odometry

    Events are sliced by aligning each depth timestamp to the nearest APS frame
    and then using image_raw_event_inds. If that index dataset is absent, the
    loader falls back to chunked timestamp search over davis/{camera}/events.
    """

    def __init__(
        self,
        *args,
        ROOT,
        sequence_names=None,
        camera="left",
        davis_group="davis",
        depth_key="depth_image_rect",
        pose_key="pose",
        event_format="xytp",
        intrinsics=None,
        spatial_transform="none",
        test_frame_count=20,
        event_resize_method="voxel_antialias",
        event_resize_bins=10,
        return_debug_event_fields=False,
        **kwargs,
    ):
        self.ROOT = ROOT
        if isinstance(sequence_names, str):
            sequence_names = [sequence_names]
        self.sequence_names = sequence_names
        self.sequence_name_set = set(sequence_names) if sequence_names else None
        self.camera = str(camera).strip("/")
        self.davis_group = str(davis_group).strip("/")
        self.depth_key = depth_key
        self.pose_key = pose_key
        self.event_format = event_format
        self.user_intrinsics = intrinsics
        self.spatial_transform = spatial_transform
        self.test_frame_count = int(test_frame_count)
        self.event_resize_method = event_resize_method
        self.event_resize_bins = event_resize_bins
        self.return_debug_event_fields = return_debug_event_fields
        self.start_img_ids = []
        self.scene_data = {}
        self.scenes = []
        self.is_metric = True
        self.video = True
        super().__init__(*args, **kwargs)
        self._discover_sequences()

    @property
    def _base(self):
        return f"{self.davis_group}/{self.camera}"

    def __len__(self):
        return len(self.start_img_ids)

    def get_stats(self):
        return f"{len(self)} MVSEC event clips from {len(self.scene_data)} sequences"

    def get_active_scenes(self):
        return list(self.scene_data.keys())

    def _paths(self, name):
        return [f"{self._base}/{name}", f"/{self._base}/{name}"]

    def _image_paths(self):
        return self._paths("image_raw")

    def _depth_paths(self):
        preferred = self.depth_key
        keys = [preferred, "depth_image_rect", "depth_image_raw"]
        deduped = []
        for key in keys:
            if key and key not in deduped:
                deduped.append(key)
        return [path for key in deduped for path in self._paths(key)]

    def _pose_paths(self):
        preferred = self.pose_key
        keys = [preferred, "pose", "odometry"]
        deduped = []
        for key in keys:
            if key and key not in deduped:
                deduped.append(key)
        return [path for key in deduped for path in self._paths(key)]

    def _event_paths(self):
        return self._paths("events")

    def _event_index_paths(self):
        return self._paths("image_raw_event_inds")

    def _has_data_payload(self, path):
        try:
            with h5py.File(path, "r") as h5_file:
                _, events = _find_dataset(h5_file, self._event_paths())
                _, images = _find_dataset(h5_file, self._image_paths())
                return events is not None and images is not None
        except OSError:
            return False

    def _has_gt_payload(self, path):
        try:
            with h5py.File(path, "r") as h5_file:
                _, depth = _find_dataset(h5_file, self._depth_paths())
                return depth is not None
        except OSError:
            return False

    def _pair_files(self):
        candidates = _candidate_h5_files(self.ROOT)
        by_key = {}
        for path in candidates:
            key = _sequence_key(path)
            entry = by_key.setdefault(key, {"data": [], "gt": []})
            if self._has_data_payload(path):
                entry["data"].append(path)
            if self._has_gt_payload(path):
                entry["gt"].append(path)

        pairs = []
        for key, entry in sorted(by_key.items()):
            if self.sequence_name_set and key not in self.sequence_name_set:
                continue
            data_path = entry["data"][0] if entry["data"] else None
            gt_path = entry["gt"][0] if entry["gt"] else data_path
            if data_path is None or gt_path is None:
                continue
            pairs.append((key, data_path, gt_path))
        return pairs

    def _parse_event_columns(self, sample, attrs=None, image_resolution=None):
        fmt = str(self.event_format or "auto").lower().replace(",", "").replace("_", "")
        if len(fmt) == 4 and set(fmt) == {"x", "y", "t", "p"}:
            letters = list(fmt)
            return {name: letters.index(name) for name in ("t", "x", "y", "p")}
        if fmt in {"mvsec", "default"}:
            return {"x": 0, "y": 1, "t": 2, "p": 3}

        columns = BaseEventMultiViewDataset._event_columns_from_attrs(attrs or {})
        if columns is not None:
            return columns

        if sample.size > 0 and sample.ndim == 2 and sample.shape[1] >= 4:
            if image_resolution is not None:
                width, height = image_resolution
                maybe_x = sample[:, 0]
                maybe_y = sample[:, 1]
                maybe_t = sample[:, 2]
                xy_ok = (
                    np.nanmin(maybe_x) >= 0
                    and np.nanmax(maybe_x) < max(width, 1)
                    and np.nanmin(maybe_y) >= 0
                    and np.nanmax(maybe_y) < max(height, 1)
                )
                t_ok = np.mean(np.diff(maybe_t.astype(np.float64)) >= 0) > 0.99 if len(maybe_t) > 1 else True
                if xy_ok and t_ok:
                    return {"x": 0, "y": 1, "t": 2, "p": 3}
        return BaseEventMultiViewDataset._infer_event_columns(sample)

    def _infer_intrinsics(self, h5_files, src_resolution):
        if self.user_intrinsics is not None:
            intrinsics = np.asarray(self.user_intrinsics, dtype=np.float32)
            if intrinsics.shape == (3, 3):
                return intrinsics
            if intrinsics.size >= 4:
                values = intrinsics.reshape(-1)
                mat = np.eye(3, dtype=np.float32)
                mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2] = values[:4]
                return mat

        attr_names = ("K", "camera_matrix", "intrinsics")
        dataset_names = (
            f"{self._base}/camera_matrix",
            f"{self._base}/intrinsics",
            f"{self._base}/K",
            "camera_matrix",
            "intrinsics",
            "K",
        )
        for h5_file in h5_files:
            for name in attr_names:
                if name in h5_file.attrs:
                    value = np.asarray(h5_file.attrs[name], dtype=np.float32)
                    if value.size >= 9:
                        return value.reshape(-1)[:9].reshape(3, 3)
                    if value.size >= 4:
                        mat = np.eye(3, dtype=np.float32)
                        mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2] = value.reshape(-1)[:4]
                        return mat
            for path in dataset_names:
                ds = _as_dataset(_h5_obj(h5_file, path))
                if ds is not None:
                    value = np.asarray(ds[:], dtype=np.float32)
                    if value.size >= 9:
                        return value.reshape(-1)[:9].reshape(3, 3)
                    if value.size >= 4:
                        mat = np.eye(3, dtype=np.float32)
                        mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2] = value.reshape(-1)[:4]
                        return mat

        width, height = src_resolution
        focal = float(max(width, height))
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = focal
        intrinsics[1, 1] = focal
        intrinsics[0, 2] = (float(width) - 1.0) * 0.5
        intrinsics[1, 2] = (float(height) - 1.0) * 0.5
        return intrinsics

    @staticmethod
    def _search_event_time_indices(events_ds, time_col, query_times, chunk_size=1_000_000):
        query_times = np.asarray(query_times, dtype=np.float64).reshape(-1)
        if query_times.size == 0:
            return np.zeros((0,), dtype=np.int64)

        total = int(events_ds.shape[0])
        order = np.argsort(query_times)
        sorted_queries = query_times[order]
        found = np.zeros_like(sorted_queries, dtype=np.int64)

        query_pos = 0
        for start in range(0, total, int(chunk_size)):
            end = min(start + int(chunk_size), total)
            times = np.asarray(events_ds[start:end, time_col], dtype=np.float64)
            if times.size == 0:
                continue

            finite = np.isfinite(times)
            if not finite.any():
                continue
            times = times[finite]
            last_t = float(times[-1])
            next_query_pos = int(np.searchsorted(sorted_queries, last_t, side="right"))
            if next_query_pos <= query_pos:
                continue

            chunk_queries = sorted_queries[query_pos:next_query_pos]
            rel = np.searchsorted(times, chunk_queries, side="left")
            found[query_pos:next_query_pos] = np.minimum(start + rel, total)
            query_pos = next_query_pos
            if query_pos >= sorted_queries.size:
                break

        if query_pos < sorted_queries.size:
            found[query_pos:] = total

        restored = np.zeros_like(found)
        restored[order] = found
        return restored.astype(np.int64)

    def _build_meta(self, name, data_path, gt_path):
        with h5py.File(data_path, "r") as data_h5, h5py.File(gt_path, "r") as gt_h5:
            event_path, events_ds = _find_dataset(data_h5, self._event_paths())
            image_path, image_ds = _find_dataset(data_h5, self._image_paths())
            event_inds_path, event_inds_ds = _find_dataset(data_h5, self._event_index_paths())
            depth_path, depth_ds = _find_dataset(gt_h5, self._depth_paths())
            pose_path, pose_ds = _find_dataset(gt_h5, self._pose_paths())

            if events_ds is None or image_ds is None or depth_ds is None:
                return None

            event_count = int(events_ds.shape[0])
            image_count = int(image_ds.shape[0])
            pose_count = int(pose_ds.shape[0]) if pose_ds is not None else 0
            image_ts = _find_timestamps(data_h5, [image_path] + self._image_paths())
            depth_ts = _find_timestamps(gt_h5, [depth_path] + self._depth_paths())
            pose_ts = _find_timestamps(gt_h5, [pose_path] + self._pose_paths()) if pose_path else None

            frame_count = int(depth_ds.shape[0])
            if depth_ts is None or len(depth_ts) != frame_count:
                depth_ts = np.arange(frame_count, dtype=np.float64) / max(float(self.fps), 1.0)
            else:
                depth_ts = np.asarray(depth_ts[:frame_count], dtype=np.float64)

            if image_ts is None:
                image_ts = depth_ts[:image_count] if image_count <= len(depth_ts) else np.arange(image_count)
            image_ts = np.asarray(image_ts, dtype=np.float64).reshape(-1)

            if pose_ds is not None:
                if pose_ts is None:
                    pose_ts = depth_ts[:pose_count] if pose_count <= len(depth_ts) else np.arange(pose_count)
                pose_ts = np.asarray(pose_ts, dtype=np.float64).reshape(-1)

            sample_count = min(event_count, 20000)
            sample = np.asarray(events_ds[:sample_count]) if sample_count > 0 else np.zeros((0, 4), dtype=np.float32)
            image_resolution = _image_to_rgb(image_ds[0]).size
            attrs = {key: BaseEventMultiViewDataset._decode_h5_attr(value) for key, value in events_ds.attrs.items()}
            columns = self._parse_event_columns(sample, attrs=attrs, image_resolution=image_resolution)
            image_for_depth = np.asarray(
                [_nearest_index(image_ts, float(timestamp)) for timestamp in depth_ts],
                dtype=np.int64,
            )

            image_event_inds = None
            if event_inds_ds is not None and int(event_inds_ds.shape[0]) > 0:
                image_event_inds = np.asarray(event_inds_ds[:], dtype=np.int64).reshape(-1)
                image_event_inds = np.clip(image_event_inds, 0, event_count)
                image_for_depth = np.clip(image_for_depth, 0, len(image_event_inds) - 1)
                depth_event_inds = image_event_inds[image_for_depth]
                event_index = np.zeros((frame_count, 2), dtype=np.int64)
                event_index[1:, 0] = depth_event_inds[:-1]
                event_index[1:, 1] = depth_event_inds[1:]
                bad = event_index[:, 1] < event_index[:, 0]
                event_index[bad, 1] = event_index[bad, 0]
                event_index_source = "image_raw_event_inds"
            else:
                query_times = np.stack([depth_ts[:-1], depth_ts[1:]], axis=1).reshape(-1)
                found = self._search_event_time_indices(events_ds, columns["t"], query_times)
                event_index = np.zeros((frame_count, 2), dtype=np.int64)
                if found.size > 0:
                    found = found.reshape(-1, 2)
                    event_index[1:, 0] = found[:, 0]
                    event_index[1:, 1] = found[:, 1]
                event_index_source = "event_timestamp_scan"

            intrinsics = self._infer_intrinsics((data_h5, gt_h5), image_resolution)

        return {
            "name": name,
            "data_path": data_path,
            "gt_path": gt_path,
            "event_path": event_path,
            "event_inds_path": event_inds_path,
            "image_path": image_path,
            "depth_path": depth_path,
            "pose_path": pose_path,
            "image_ts": image_ts.astype(np.float64),
            "depth_ts": depth_ts.astype(np.float64),
            "pose_ts": pose_ts.astype(np.float64) if pose_ts is not None else None,
            "image_for_depth": image_for_depth.astype(np.int64),
            "event_columns": columns,
            "event_index": event_index,
            "event_index_source": event_index_source,
            "frame_count": frame_count,
            "image_count": image_count,
            "pose_count": pose_count,
            "src_resolution": np.array(image_resolution, dtype=np.int32),
            "intrinsics": intrinsics.astype(np.float32),
        }

    def _split_start_ids(self, meta):
        frame_count = int(meta["frame_count"])
        min_start = 1
        max_start_exclusive = max(min_start, frame_count - self.num_views + 1)
        all_starts = list(range(min_start, max_start_exclusive))
        if not all_starts:
            return []

        test_count = min(max(self.test_frame_count, self.num_views), len(all_starts))
        if self.split == "test":
            return all_starts[-test_count:]
        if self.split == "train":
            return all_starts[:-test_count] or all_starts
        return all_starts

    def _discover_sequences(self):
        for name, data_path, gt_path in self._pair_files():
            meta = self._build_meta(name, data_path, gt_path)
            if meta is None:
                continue
            start_ids = self._split_start_ids(meta)
            if not start_ids:
                continue
            meta["start_ids"] = start_ids
            self.scene_data[name] = meta
            self.scenes.append(name)
            self.start_img_ids.extend((name, start_id) for start_id in start_ids)

        if not self.start_img_ids:
            print(
                f"MVSECEventDataset found no usable clips under {self.ROOT}. "
                f"camera={self.camera}, depth_key={self.depth_key}, split={self.split}"
            )

    def _read_event_slice(self, events_ds, columns, start_idx, end_idx, time_origin):
        if end_idx <= start_idx:
            return MyEventDataset._pack_event_data(
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        events = np.asarray(events_ds[int(start_idx) : int(end_idx)])
        if events.size == 0:
            return MyEventDataset._pack_event_data(
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        event_x = events[:, columns["x"]].astype(np.int32)
        event_y = events[:, columns["y"]].astype(np.int32)
        event_t = (events[:, columns["t"]].astype(np.float64) - float(time_origin)).astype(np.float32)
        event_p = events[:, columns["p"]].astype(np.float32)
        event_p[event_p == 0] = -1.0
        return MyEventDataset._pack_event_data(
            np.stack([event_x, event_y], axis=-1).astype(np.int32),
            event_t,
            event_p,
        )

    def _load_pose(self, gt_h5, meta, timestamp):
        if not meta["pose_path"] or meta["pose_ts"] is None:
            return np.eye(4, dtype=np.float32)
        pose_ds = _as_dataset(_h5_obj(gt_h5, meta["pose_path"]))
        if pose_ds is None or len(pose_ds) == 0:
            return np.eye(4, dtype=np.float32)
        pose_idx = _nearest_index(meta["pose_ts"], timestamp)
        return _pose_to_matrix(pose_ds[pose_idx])

    def _get_views(self, idx, resolution, rng, num_views):
        scene_name, start_id = self.start_img_ids[idx]
        meta = self.scene_data[scene_name]
        frame_ids = list(range(start_id, start_id + num_views))
        views = []

        with h5py.File(meta["data_path"], "r") as data_h5, h5py.File(meta["gt_path"], "r") as gt_h5:
            events_ds = _as_dataset(_h5_obj(data_h5, meta["event_path"]))
            image_ds = _as_dataset(_h5_obj(data_h5, meta["image_path"]))
            depth_ds = _as_dataset(_h5_obj(gt_h5, meta["depth_path"]))

            for local_idx, frame_idx in enumerate(frame_ids):
                timestamp = float(meta["depth_ts"][frame_idx])
                prev_timestamp = float(meta["depth_ts"][frame_idx - 1]) if frame_idx > 0 else timestamp

                if frame_idx < len(meta["image_for_depth"]):
                    image_idx = int(meta["image_for_depth"][frame_idx])
                else:
                    image_idx = _nearest_index(meta["image_ts"], timestamp)
                image = _image_to_rgb(image_ds[image_idx])
                src_width, src_height = image.size
                depth, mask = _clean_depth(depth_ds[frame_idx])

                intrinsics = meta["intrinsics"].copy()
                image = self.resize_image(image, resolution)
                depth = self.resize_hw_map(depth, resolution, mode="bilinear", mask=mask)
                mask = self.resize_hw_map(mask.astype(np.float32), resolution, mode="bilinear") > 0.5
                intrinsics = self.scale_intrinsics(intrinsics, (src_width, src_height), resolution)
                width, height = image.size

                event_start, event_end = meta["event_index"][frame_idx]
                event_data = self._read_event_slice(
                    events_ds,
                    meta["event_columns"],
                    event_start,
                    event_end,
                    time_origin=prev_timestamp,
                )
                event_data = MyEventDataset._resize_event_data(
                    event_data,
                    src_resolution=meta["src_resolution"],
                    dst_resolution=np.array([width, height], dtype=np.int32),
                    spatial_transform=self.spatial_transform,
                    resize_method=self.event_resize_method,
                    resize_bins=self.event_resize_bins,
                )
                if event_data.get("event_voxel") is not None:
                    event_data["event_voxel"] = event_data["event_voxel"] * mask[None].astype(np.float32, copy=False)

                pose = self._load_pose(gt_h5, meta, timestamp)
                view = dict(
                    img=image,
                    depthmap=depth.astype(np.float32),
                    camera_pose=pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    mask=mask.astype(bool),
                    event_xy=event_data["event_xy"],
                    event_t=event_data["event_t"],
                    event_p=event_data["event_p"],
                    event_voxel=event_data.get("event_voxel", np.zeros((0, height, width), dtype=np.float32)),
                    event_time_range=np.array([prev_timestamp, timestamp], dtype=np.float32),
                    event_resolution=np.array([width, height], dtype=np.int32),
                    event_source_resolution=meta["src_resolution"].astype(np.int32),
                    event_spatial_transform=self.spatial_transform,
                    event_y_flip=np.array(self.spatial_transform == "vflip", dtype=bool),
                    has_event=np.array(event_end > event_start, dtype=bool),
                    dataset="mvsec_event_dataset",
                    label=f"{scene_name}_{self.camera}_{frame_idx:06d}",
                    instance=f"{scene_name}_{frame_idx:06d}",
                    is_metric=self.is_metric,
                    is_video=True,
                    img_mask=np.array(True, dtype=bool),
                    reset=np.array(local_idx == 0, dtype=bool),
                )
                if self.return_debug_event_fields:
                    view["events"] = event_data["events"]
                views.append(view)

        return views


def get_mvsec_dataset(
    root,
    *,
    num_views=6,
    resolution=(346, 260),
    fps=20,
    seed=0,
    split="train",
    sequence_names=None,
    camera="left",
    test_frame_count=20,
    event_resize_method="voxel_antialias",
    event_resize_bins=10,
    **kwargs,
):
    return MVSECEventDataset(
        ROOT=root,
        num_views=num_views,
        split=split,
        resolution=resolution,
        fps=fps,
        seed=seed,
        sequence_names=sequence_names,
        camera=camera,
        test_frame_count=test_frame_count,
        event_resize_method=event_resize_method,
        event_resize_bins=event_resize_bins,
        **kwargs,
    )


__all__ = ["MVSECEventDataset", "get_mvsec_dataset", "event_multiview_collate"]
