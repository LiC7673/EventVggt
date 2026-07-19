"""Convert the MVSEC ROS1 bags used by MVSECEventDataset into paired HDF5 files."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _time_seconds(value):
    """Read both native ROS1 and rosbags normalized time objects."""
    if hasattr(value, "to_sec"):
        return float(value.to_sec())
    sec = getattr(value, "sec", getattr(value, "secs", 0))
    nsec = getattr(value, "nanosec", getattr(value, "nsecs", 0))
    return float(sec) + float(nsec) * 1e-9


def stamp(msg):
    return _time_seconds(msg.header.stamp)


def iter_bag(path, topics):
    """Yield deserialized ROS1 messages without requiring a ROS installation."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "Package 'rosbags' is required; install it with: pip install rosbags"
        ) from exc
    with AnyReader([Path(path)]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topics]
        missing = sorted(set(topics) - {connection.topic for connection in connections})
        if missing:
            print(f"[rosbags] {Path(path).name}: absent optional topics={missing}", flush=True)
        for connection, _, rawdata in reader.messages(connections=connections):
            yield connection.topic, reader.deserialize(rawdata, connection.msgtype)


def image_array(msg):
    dtype = np.float32 if "32F" in msg.encoding else np.uint16 if "16" in msg.encoding else np.uint8
    raw = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
    arr = np.frombuffer(raw, dtype=dtype)
    channels = max(int(msg.step) // (int(msg.width) * np.dtype(dtype).itemsize), 1)
    arr = arr.reshape(int(msg.height), int(msg.width), channels)
    return arr[..., 0] if channels == 1 else arr[..., :3]


def append(ds, values):
    old = ds.shape[0]; ds.resize(old + len(values), axis=0); ds[old:] = values


def convert(data_bag, gt_bag, output_dir):
    name = Path(data_bag).stem.removesuffix("_data"); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    data_out, gt_out = output_dir / f"{name}_data.hdf5", output_dir / f"{name}_gt.hdf5"
    with h5py.File(data_out, "w") as h5:
        g = h5.require_group("davis/left"); event_ds = g.create_dataset("events", (0, 4), maxshape=(None, 4), dtype="f8", chunks=(262144, 4))
        image_ds = image_ts = image_inds = None; event_count = 0
        topics = ["/davis/left/events", "/davis/left/image_raw", "/davis/left/camera_info"]
        for topic, msg in iter_bag(data_bag, topics):
            if topic.endswith("/events"):
                values = np.asarray(
                    [[e.x, e.y, _time_seconds(e.ts), 1 if e.polarity else -1] for e in msg.events],
                    dtype=np.float64,
                )
                if len(values): append(event_ds, values); event_count += len(values)
            elif topic.endswith("/camera_info"):
                if "camera_matrix" not in g:
                    matrix = getattr(msg, "k", getattr(msg, "K", None))
                    if matrix is not None:
                        g.create_dataset("camera_matrix", data=np.asarray(matrix, dtype=np.float32).reshape(3, 3))
            else:
                image = image_array(msg)
                if image_ds is None:
                    image_ds = g.create_dataset("image_raw", (0,) + image.shape, maxshape=(None,) + image.shape, dtype=image.dtype, chunks=(1,) + image.shape)
                    image_ts = g.create_dataset("image_raw_ts", (0,), maxshape=(None,), dtype="f8")
                    image_inds = g.create_dataset("image_raw_event_inds", (0,), maxshape=(None,), dtype="i8")
                append(image_ds, image[None]); append(image_ts, [stamp(msg)]); append(image_inds, [event_count])
    with h5py.File(gt_out, "w") as h5:
        g = h5.require_group("davis/left"); depth_ds = depth_ts = pose_ds = pose_ts = None
        topics = ["/davis/left/depth_image_rect", "/davis/left/pose"]
        for topic, msg in iter_bag(gt_bag, topics):
            if topic.endswith("depth_image_rect"):
                depth = image_array(msg).astype(np.float32)
                if depth_ds is None:
                    depth_ds = g.create_dataset("depth_image_rect", (0,) + depth.shape, maxshape=(None,) + depth.shape, dtype="f4", chunks=(1,) + depth.shape)
                    depth_ts = g.create_dataset("depth_image_rect_ts", (0,), maxshape=(None,), dtype="f8")
                append(depth_ds, depth[None]); append(depth_ts, [stamp(msg)])
            else:
                pose = getattr(msg, "pose", None)
                if pose is not None:
                    p, q = pose.position, pose.orientation
                else:
                    transform = msg.transform; p, q = transform.translation, transform.rotation
                value = np.asarray([[p.x, p.y, p.z, q.x, q.y, q.z, q.w]], dtype=np.float32)
                if pose_ds is None:
                    pose_ds = g.create_dataset("pose", (0, 7), maxshape=(None, 7), dtype="f4")
                    pose_ts = g.create_dataset("pose_ts", (0,), maxshape=(None,), dtype="f8")
                append(pose_ds, value); append(pose_ts, [stamp(msg)])
    print(f"converted {name}: {data_out}, {gt_out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    p.add_argument(
        "--sequences", nargs="+",
        default=["outdoor_day1", "outdoor_day2", "outdoor_night1", "outdoor_night2", "outdoor_night3"],
        help="MVSEC sequence stems to convert (only the requested bags are required)",
    )
    a = p.parse_args(); root = Path(a.root)
    for name in a.sequences:
        folder = root / ("outdoor_day" if "day" in name else "outdoor_night")
        data, gt = folder / f"{name}_data.bag", folder / f"{name}_gt.bag"
        if not data.is_file() or not gt.is_file(): raise FileNotFoundError(f"missing pair: {data}, {gt}")
        convert(data, gt, a.output)


if __name__ == "__main__": main()
