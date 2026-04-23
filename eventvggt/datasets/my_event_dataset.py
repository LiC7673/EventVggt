import json
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from eventvggt.datasets.base.base_event_dataset import BaseEventMultiViewDataset


def _numeric_stem_key(path):
    stem = osp.splitext(osp.basename(path))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _list_files(folder, suffixes):
    if not osp.isdir(folder):
        return []
    return sorted(
        [
            osp.join(folder, name)
            for name in os.listdir(folder)
            if osp.splitext(name)[1].lower() in suffixes
        ],
        key=_numeric_stem_key,
    )


class MyEventDataset(BaseEventMultiViewDataset):
    """Event/RGB sequential dataset with lazy per-scene loading."""

    def __init__(
        self,
        *args,
        ROOT,
        scene_names=None,
        initial_scene_idx=0,
        active_scene_count=1,
        test_frame_count=10,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.scene_names = scene_names
        self.current_scene_index = initial_scene_idx
        self.active_scene_count = active_scene_count
        self.test_frame_count = test_frame_count
        self.is_metric = False
        self.video = True
        # Extract split from kwargs, default to 'train'
        self.split = kwargs.get('split', 'train')
        super().__init__(*args, **kwargs)
        self._discover_scenes()

    def _load_camera_matrices(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = data["fl_x"]
        intrinsics[1, 1] = data["fl_y"]
        intrinsics[0, 2] = data["cx"]
        intrinsics[1, 2] = data["cy"]

        poses = np.stack(
            [np.asarray(frame["transform_matrix"], dtype=np.float32) for frame in data["frames"]],
            axis=0,
        )
        poses = self.blender_to_opencv(poses).astype(np.float32)
        intrinsics = np.repeat(intrinsics[None], len(poses), axis=0).astype(np.float32)
        return intrinsics, poses

    def _probe_scene(self, scene_name):
        scene_dir = osp.join(self.ROOT, scene_name)
        image_paths = _list_files(osp.join(scene_dir, "LDR","ev_5"), {".png", ".jpg", ".jpeg"})
        event_path = osp.join(scene_dir, "event", "v2e-dvs-events.h5")
        pose_json = osp.join(scene_dir, "transforms.json")

        if not image_paths or not osp.isfile(event_path) or not osp.isfile(pose_json):
            return None

        intrinsics, poses = self._load_camera_matrices(pose_json)
        frame_count = min(len(image_paths), len(intrinsics), len(poses))
        if frame_count < self.num_views:
            return None

        return {
            "scene": scene_name,
            "scene_dir": scene_dir,
            "frame_count": frame_count,
            "event_h5": event_path,
            "pose_json": pose_json,
        }

    def _discover_scenes(self):
        if self.scene_names is None:
            raw_scenes = sorted(
                [name for name in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, name))]
            )
        else:
            raw_scenes = [
                name for name in self.scene_names if osp.isdir(osp.join(self.ROOT, name))
            ]

        self.scenes = []
        self.scene_records = {}
        self.scene_data_cache = {}
        self.active_scenes = []
        self.active_scene_data = {}
        self.start_img_ids = []

        for scene_name in raw_scenes:
            record = self._probe_scene(scene_name)
            if record is None:
                continue
            self.scenes.append(scene_name)
            self.scene_records[scene_name] = record

        if not self.scenes:
            raise RuntimeError(f"No valid scenes found under {self.ROOT}")

        self.current_scene_index %= len(self.scenes)
        initial_indices = [
            (self.current_scene_index + offset) % len(self.scenes)
            for offset in range(max(1, self.active_scene_count))
        ]
        self.set_active_scenes_by_indices(initial_indices)

    def _read_scene_metadata(self, scene_name):
        record = self.scene_records[scene_name]
        scene_dir = record["scene_dir"]
        # image_paths = _list_files(osp.join(scene_dir, "HDR"), {".png", ".jpg", ".jpeg"})
        image_paths = _list_files(osp.join(scene_dir, "LDR","ev_2"), {".png", ".jpg", ".jpeg"})
        normal_paths = _list_files(osp.join(scene_dir, "Normal"), {".png", ".jpg", ".jpeg"})
        mask_paths = _list_files(osp.join(scene_dir, "Mask"), {".png", ".jpg", ".jpeg"})
        # depth_paths = _list_files(
        #     osp.join(scene_dir, "depth"), {".png ".exr", ".jpg", ".jpeg", ".npy", ".npz"}
        # )
        depth_paths = _list_files(
            osp.join(scene_dir, "depth"), { ".exr"}
        )
        # depth_paths = _list_files(
        #     osp.join(scene_dir, "depth"), { ".png"}
        # )
        # print(depth_paths)
        intrinsics, poses = self._load_camera_matrices(record["pose_json"])
        frame_count = min(record["frame_count"], len(image_paths), len(intrinsics), len(poses))
        image_paths = image_paths[:frame_count]
        normal_paths = normal_paths[:frame_count] if normal_paths else []
        mask_paths = mask_paths[:frame_count] if mask_paths else []
        depth_paths = depth_paths[:frame_count] if depth_paths else []

        frame_event_index = self.build_frame_event_index(record["event_h5"], frame_count)

        # Split dataset: last test_frame_count frames for test, rest for train
        train_frame_count = frame_count - self.test_frame_count
        
        if self.split == 'train':
            # For train split: all frames must be in training range [0, train_frame_count)
            # So start_id + num_views <= train_frame_count
            # Which means start_id <= train_frame_count - num_views
            max_start_id = max(0, train_frame_count - self.num_views + 1)
            start_ids = list(range(max_start_id))
        elif self.split == 'test':
            # For test split: at least one frame must be in test range [train_frame_count, frame_count)
            # So start_id + num_views > train_frame_count
            # Which means start_id >= train_frame_count - num_views + 1
            min_start_id = max(0, train_frame_count - self.num_views + 1)
            start_ids = list(range(min_start_id, frame_count - self.num_views + 1))
        else:
            # Default: use all start_ids (for backward compatibility)
            start_ids = list(range(frame_count - self.num_views + 1))

        return {
            "scene": scene_name,
            "scene_dir": scene_dir,
            "image_paths": image_paths,
            "normal_paths": normal_paths,
            "mask_paths": mask_paths,
            "depth_paths": depth_paths,
            "intrinsics": intrinsics[:frame_count],
            "poses": poses[:frame_count],
            "event_h5": record["event_h5"],
            "frame_count": frame_count,
            "frame_event_index": frame_event_index,
            "start_ids": start_ids,
        }

    def _ensure_scene_loaded(self, scene_name):
        if scene_name not in self.scene_data_cache:
            self.scene_data_cache[scene_name] = self._read_scene_metadata(scene_name)
        return self.scene_data_cache[scene_name]

    def _rebuild_start_img_ids(self):
        self.start_img_ids = []
        for scene_name in self.active_scenes:
            scene_meta = self.active_scene_data[scene_name]
            self.start_img_ids.extend(
                (scene_name, start_id) for start_id in scene_meta["start_ids"]
            )

    def set_active_scenes(self, scene_names):
        if isinstance(scene_names, str):
            scene_names = [scene_names]
        self.active_scenes = list(scene_names)
        self.active_scene_data = {
            scene_name: self._ensure_scene_loaded(scene_name) for scene_name in self.active_scenes
        }
        self._rebuild_start_img_ids()

    def set_active_scenes_by_indices(self, scene_indices):
        if isinstance(scene_indices, int):
            scene_indices = [scene_indices]
        scene_names = [self.scenes[idx % len(self.scenes)] for idx in scene_indices]
        self.set_active_scenes(scene_names)
        self.current_scene_index = scene_indices[0] % len(self.scenes)

    def randomize_active_scenes(self, count=None, rng=None):
        count = self.active_scene_count if count is None else count
        count = max(1, min(count, len(self.scenes)))
        rng = np.random.default_rng() if rng is None else rng
        chosen = rng.choice(self.scenes, size=count, replace=False).tolist()
        self.set_active_scenes(chosen)
        self.current_scene_index = self.scenes.index(chosen[0])
        return chosen

    def get_current_scene(self):
        return self.active_scenes[0] if self.active_scenes else None

    def get_active_scenes(self):
        return list(self.active_scenes)

    def change_current_scene(self, scene_idx):
        count = self.active_scene_count
        indices = [(scene_idx + offset) % len(self.scenes) for offset in range(count)]
        self.set_active_scenes_by_indices(indices)

    def load_new_scene(self):
        self.change_current_scene((self.current_scene_index + 1) % len(self.scenes))

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return sum(scene["frame_count"] for scene in self.active_scene_data.values())

    @staticmethod
    def _load_rgb(path):
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_optional(paths, frame_idx, loader):
        if not paths or frame_idx >= len(paths) or not osp.isfile(paths[frame_idx]):
            return None
        return loader(paths[frame_idx])

    def _load_view_data(self, scene_meta, frame_idx, resolution,normalize=True):
        image = self._load_rgb(scene_meta["image_paths"][frame_idx])
        src_width, src_height = image.size
        mask = self._load_optional(scene_meta["mask_paths"], frame_idx, self.load_mask)
        normal = self._load_optional(
            scene_meta["normal_paths"], frame_idx, lambda path: self.load_normal_map(path, None)
        )

        depth_shape = image.size[1], image.size[0]
      
        depth_path = (
            scene_meta["depth_paths"][frame_idx]
            if scene_meta["depth_paths"] and frame_idx < len(scene_meta["depth_paths"])
            else None
        )
        depthmap = self.load_depth_any(depth_path, fallback_shape=depth_shape)

        if mask is not None:
            # Apply mask to image
            img_np = np.array(image)
            img_np[~mask] = 0  # Set invalid regions to black
            image = Image.fromarray(img_np)

        if mask is not None and normal is not None:
            normal = normal.astype(np.float32)
            normal[~mask] = 0.0
        if mask is not None and depthmap is not None:
            depthmap[~mask] = 0.0
            depthmap[depthmap > 250] = 0.0  
        if normalize:
            # Handle invalid values (inf, nan)
            valid_mask = np.isfinite(depthmap)
            
            if not valid_mask.any():
                # All invalid, return zeros
                return np.zeros_like(depthmap, dtype=np.float32)
            
            # Get min and max from valid values
            valid_depth = depthmap[valid_mask]
            depth_min = valid_depth.min()
            depth_max = valid_depth.max()
            
            # # Avoid division by zero
            # if depth_max > depth_min:
            #     depth = (depthmap - depth_min) / (depth_max - depth_min)
            # else:
            #     depth = np.zeros_like(depthmap, dtype=np.float32)
            depth = depthmap.astype(np.float32)/255.0
            # Clip to ensure all values are in [0, 1]
            depth = np.clip(depth, 0.0, 1.0)
        intrinsics = scene_meta["intrinsics"][frame_idx]
        image, depthmap, intrinsics = self._crop_resize_if_necessary(
            image, depthmap, intrinsics, resolution, rng=self._rng, info=frame_idx
        )
        resized = {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_intrinsics": intrinsics.astype(np.float32),
            "src_resolution": np.array([src_width, src_height], dtype=np.int32),
            "dst_resolution": np.array([image.size[0], image.size[1]], dtype=np.int32),
        }
        if normal is not None:
            resized["normal"] = self.resize_hw_map(normal, resolution, mode="bilinear")
        if mask is not None:
            resized["mask"] = self.resize_hw_map(mask, resolution,)
        return resized

    @staticmethod
    def _resize_event_data(event_data, src_resolution, dst_resolution):
        src_width, src_height = int(src_resolution[0]), int(src_resolution[1])
        dst_width, dst_height = int(dst_resolution[0]), int(dst_resolution[1])

        event_xy = event_data["event_xy"]
        event_t = event_data["event_t"]
        event_p = event_data["event_p"]

        if event_xy.size == 0:
            return {
                "event_xy": event_xy.astype(np.int32, copy=False),
                "event_t": event_t.astype(np.float32, copy=False),
                "event_p": event_p.astype(np.float32, copy=False),
                "events": np.zeros((0, 4), dtype=np.float32),
            }

        sx = dst_width / max(src_width, 1)
        sy = dst_height / max(src_height, 1)

        resized_xy = event_xy.astype(np.float32, copy=True)
        resized_xy[:, 0] = np.floor(resized_xy[:, 0] * sx)
        resized_xy[:, 1] = np.floor(resized_xy[:, 1] * sy)
        resized_xy = resized_xy.astype(np.int32)

        valid = (
            (resized_xy[:, 0] >= 0)
            & (resized_xy[:, 0] < dst_width)
            & (resized_xy[:, 1] >= 0)
            & (resized_xy[:, 1] < dst_height)
        )
        resized_xy = resized_xy[valid]
        event_t = event_t[valid].astype(np.float32, copy=False)
        event_p = event_p[valid].astype(np.float32, copy=False)

        return {
            "event_xy": resized_xy,
            "event_t": event_t,
            "event_p": event_p,
            "events": np.stack(
                [
                    resized_xy[:, 0].astype(np.float32, copy=False),
                    resized_xy[:, 1].astype(np.float32, copy=False),
                    event_t,
                    event_p,
                ],
                axis=-1,
            ).astype(np.float32, copy=False),
        }

    def _get_views(self, idx, resolution, rng, num_views):
        scene_name, start_id = self.start_img_ids[idx]
        scene_meta = self.active_scene_data[scene_name]

        frame_ids = list(range(start_id, start_id + num_views))
        views = []
        for frame_idx in frame_ids:
            resized = self._load_view_data(scene_meta, frame_idx, resolution)
            width, height = resized["img"].size
            event_start, event_end = scene_meta["frame_event_index"][frame_idx]
            event_data = self.load_event_slice(scene_meta["event_h5"], event_start, event_end)
            event_data = self._resize_event_data(
                event_data,
                src_resolution=resized["src_resolution"],
                dst_resolution=resized["dst_resolution"],
            )

            # Apply mask to events if mask is available
            if "mask" in resized and event_data["event_xy"].size > 0:
                mask = resized["mask"]
                valid_events = mask[event_data["event_xy"][:, 1], event_data["event_xy"][:, 0]]
                event_data["event_xy"] = event_data["event_xy"][valid_events]
                event_data["event_t"] = event_data["event_t"][valid_events]
                event_data["event_p"] = event_data["event_p"][valid_events]
                event_data["events"] = event_data["events"][valid_events]

            if frame_idx == 0:
                time_range = np.array([0.0, 0.0], dtype=np.float32)
            else:
                time_range = np.array(
                    [(frame_idx - 1) * self.dt_us, frame_idx * self.dt_us], dtype=np.float32
                )

            basename = osp.splitext(osp.basename(scene_meta["image_paths"][frame_idx]))[0]
            view = dict(
                img=resized["img"],
                depthmap=resized["depthmap"].astype(np.float32),
                camera_pose=scene_meta["poses"][frame_idx].astype(np.float32),
                camera_intrinsics=resized["camera_intrinsics"].astype(np.float32),
                normal_gt=(
                    resized["normal"].astype(np.float32)
                    if "normal" in resized
                    else np.zeros((height, width, 3), dtype=np.float32)
                ),
                mask=(
                    resized["mask"].astype(bool)
                    if "mask" in resized
                    else np.ones((height, width), dtype=bool)
                ),
                event_xy=event_data["event_xy"],
                event_t=event_data["event_t"],
                event_p=event_data["event_p"],
                events=event_data["events"],
                event_time_range=time_range,
                event_resolution=np.array([width, height], dtype=np.int32),
                has_event=np.array(frame_idx > 0, dtype=bool),
                dataset="my_event_dataset",
                label=f"{scene_name}_{basename}",
                instance=f"{scene_name}_{frame_idx}",
                is_metric=self.is_metric,
                is_video=True,
                img_mask=np.array(True, dtype=bool),
                reset=np.array(frame_idx == start_id, dtype=bool),
            )
            views.append(view)

        return views


def get_combined_dataset(
    root,
    *,
    num_views=2,
    resolution=(640, 480),
    fps=120,
    seed=0,
    scene_names=None,
    initial_scene_idx=0,
    active_scene_count=1,
    split="train",
    test_frame_count=10,
):
    return MyEventDataset(
        ROOT=root,
        num_views=num_views,
        split=split,
        resolution=resolution,
        fps=fps,
        seed=seed,
        allow_repeat=False,
        scene_names=scene_names,
        initial_scene_idx=initial_scene_idx,
        active_scene_count=active_scene_count,
        test_frame_count=test_frame_count,
        # normalize=True
    )


def event_multiview_collate(batch):
    if not batch:
        return batch

    num_views = len(batch[0])
    collated_views = []
    variable_length_keys = {"events", "event_xy", "event_t", "event_p"}

    for view_idx in range(num_views):
        per_view_samples = [sample[view_idx] for sample in batch]
        keys = per_view_samples[0].keys()
        collated_view = {}

        for key in keys:
            values = [sample[key] for sample in per_view_samples]
            if key in variable_length_keys:
                collated_view[key] = [
                    torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    for value in values
                ]
            else:
                collated_view[key] = default_collate(values)

        collated_views.append(collated_view)

    return collated_views


if __name__ == "__main__":
    root = "E:/dataSet/myblendevent/final"

    # Create training dataset (first 90% of frames, i.e., all except last 10)
    train_dataset = get_combined_dataset(root, active_scene_count=1, split='train', test_frame_count=10)
    print(
        f"\n=== TRAINING DATASET ===\n"
        f"Active scenes: {train_dataset.get_active_scenes()}, "
        f"samples: {len(train_dataset)}, total scenes: {len(train_dataset.scenes)}"
    )

    # Create test dataset (last 10 frames)
    test_dataset = get_combined_dataset(root, active_scene_count=1, split='test', test_frame_count=10)
    print(
        f"\n=== TEST DATASET ===\n"
        f"Active scenes: {test_dataset.get_active_scenes()}, "
        f"samples: {len(test_dataset)}, total scenes: {len(test_dataset.scenes)}"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=event_multiview_collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=event_multiview_collate,
    )

    import time

    # Test training dataloader
    print("\n=== LOADING TRAINING BATCHES ===")
    start = time.time()
    for i, views in enumerate(train_loader):
        first_view = views[0]
        print(f"Train batch {i} processed. Time: {time.time() - start:.4f}s")
        start = time.time()
        if i > 2:
            break

    # Test test dataloader
    print("\n=== LOADING TEST BATCHES ===")
    start = time.time()
    for i, views in enumerate(test_loader):
        first_view = views[0]
        print(f"Test batch {i} processed. Time: {time.time() - start:.4f}s")
        start = time.time()
        if i > 2:
            break
