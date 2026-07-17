"""Convert the MVSEC ROS1 bags used by MVSECEventDataset into paired HDF5 files."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def stamp(msg):
    s = msg.header.stamp
    return float(s.secs) + float(s.nsecs) * 1e-9


def image_array(msg):
    dtype = np.float32 if "32F" in msg.encoding else np.uint16 if "16" in msg.encoding else np.uint8
    arr = np.frombuffer(msg.data, dtype=dtype)
    channels = max(int(msg.step) // (int(msg.width) * np.dtype(dtype).itemsize), 1)
    arr = arr.reshape(int(msg.height), int(msg.width), channels)
    return arr[..., 0] if channels == 1 else arr[..., :3]


def append(ds, values):
    old = ds.shape[0]; ds.resize(old + len(values), axis=0); ds[old:] = values


def convert(data_bag, gt_bag, output_dir):
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("ROS1 Python package 'rosbag' is required. Run this converter inside the MVSEC/ROS environment.") from exc
    name = Path(data_bag).stem.removesuffix("_data"); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    data_out, gt_out = output_dir / f"{name}_data.hdf5", output_dir / f"{name}_gt.hdf5"
    with h5py.File(data_out, "w") as h5:
        g = h5.require_group("davis/left"); event_ds = g.create_dataset("events", (0, 4), maxshape=(None, 4), dtype="f8", chunks=(262144, 4))
        image_ds = image_ts = image_inds = None; event_count = 0
        with rosbag.Bag(str(data_bag)) as bag:
            for topic, msg, _ in bag.read_messages(topics=["/davis/left/events", "/davis/left/image_raw", "/davis/left/camera_info"]):
                if topic.endswith("/events"):
                    values = np.asarray([[e.x, e.y, e.ts.to_sec(), 1 if e.polarity else -1] for e in msg.events], dtype=np.float64)
                    if len(values): append(event_ds, values); event_count += len(values)
                elif topic.endswith("/camera_info"):
                    if "camera_matrix" not in g:
                        g.create_dataset("camera_matrix", data=np.asarray(msg.K, dtype=np.float32).reshape(3, 3))
                else:
                    image = image_array(msg)
                    if image_ds is None:
                        image_ds = g.create_dataset("image_raw", (0,) + image.shape, maxshape=(None,) + image.shape, dtype=image.dtype, chunks=(1,) + image.shape)
                        image_ts = g.create_dataset("image_raw_ts", (0,), maxshape=(None,), dtype="f8")
                        image_inds = g.create_dataset("image_raw_event_inds", (0,), maxshape=(None,), dtype="i8")
                    append(image_ds, image[None]); append(image_ts, [stamp(msg)]); append(image_inds, [event_count])
    with h5py.File(gt_out, "w") as h5:
        g = h5.require_group("davis/left"); depth_ds = depth_ts = pose_ds = pose_ts = None
        with rosbag.Bag(str(gt_bag)) as bag:
            for topic, msg, _ in bag.read_messages(topics=["/davis/left/depth_image_rect", "/davis/left/pose"]):
                if topic.endswith("depth_image_rect"):
                    depth = image_array(msg).astype(np.float32)
                    if depth_ds is None:
                        depth_ds = g.create_dataset("depth_image_rect", (0,) + depth.shape, maxshape=(None,) + depth.shape, dtype="f4", chunks=(1,) + depth.shape)
                        depth_ts = g.create_dataset("depth_image_rect_ts", (0,), maxshape=(None,), dtype="f8")
                    append(depth_ds, depth[None]); append(depth_ts, [stamp(msg)])
                else:
                    p, q = msg.pose.position, msg.pose.orientation
                    value = np.asarray([[p.x, p.y, p.z, q.x, q.y, q.z, q.w]], dtype=np.float32)
                    if pose_ds is None:
                        pose_ds = g.create_dataset("pose", (0, 7), maxshape=(None, 7), dtype="f4")
                        pose_ts = g.create_dataset("pose_ts", (0,), maxshape=(None,), dtype="f8")
                    append(pose_ds, value); append(pose_ts, [stamp(msg)])
    print(f"converted {name}: {data_out}, {gt_out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); root = Path(a.root)
    names = ["outdoor_day2", "outdoor_night1", "outdoor_night2", "outdoor_night3"]
    for name in names:
        folder = root / ("outdoor_day" if "day" in name else "outdoor_night")
        data, gt = folder / f"{name}_data.bag", folder / f"{name}_gt.bag"
        if not data.is_file() or not gt.is_file(): raise FileNotFoundError(f"missing pair: {data}, {gt}")
        convert(data, gt, a.output)


if __name__ == "__main__": main()
