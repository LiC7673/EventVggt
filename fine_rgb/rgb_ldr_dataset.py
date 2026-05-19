import os.path as osp

import numpy as np

from eventvggt.datasets.my_event_dataset import MyEventDataset


class PureRgbLdrDataset(MyEventDataset):
    """MyEventDataset variant that never loads event slices.

    It keeps the same scene/LDR discovery logic, depth/pose/intrinsics, and view
    sampling, but returns RGB-only views. This keeps pure RGB ablations honest
    and avoids the memory/time cost of event voxel construction.
    """

    def _get_views(self, idx, resolution, rng, num_views):
        scene_name, start_id = self.start_img_ids[idx]
        scene_meta = self.active_scene_data[scene_name]
        extra_index = getattr(self, "_sample_extra_index", ())
        requested_ldr_event = extra_index[0] if extra_index else None
        ldr_event_id = self._select_ldr_event(scene_meta, rng, requested_ldr_event)
        frame_ids = list(range(start_id, start_id + num_views))

        views = []
        for frame_idx in frame_ids:
            resized = self._load_view_data(scene_meta, frame_idx, resolution, ldr_event_id=ldr_event_id)
            width, height = resized["img"].size
            image_path = scene_meta["image_paths_by_ldr"][ldr_event_id][frame_idx]
            basename = osp.splitext(osp.basename(image_path))[0]

            view = dict(
                img=resized["img"],
                depthmap=resized["depthmap"].astype(np.float32),
                camera_pose=scene_meta["poses"][frame_idx].astype(np.float32),
                camera_intrinsics=resized["camera_intrinsics"].astype(np.float32),
                mask=(
                    resized["mask"].astype(bool)
                    if "mask" in resized
                    else np.ones((height, width), dtype=bool)
                ),
                dataset="pure_rgb_ldr_dataset",
                label=f"{scene_name}_{ldr_event_id}_{basename}",
                instance=f"{scene_name}_{frame_idx}",
                ldr_event_id=ldr_event_id,
                is_metric=self.is_metric,
                is_video=True,
                img_mask=np.array(True, dtype=bool),
                reset=np.array(frame_idx == start_id, dtype=bool),
            )
            if self.return_normal_gt:
                view["normal_gt"] = (
                    resized["normal"].astype(np.float32)
                    if "normal" in resized
                    else np.zeros((height, width, 3), dtype=np.float32)
                )
            views.append(view)

        return views


def get_rgb_ldr_dataset(
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
    ldr_event_id="auto",
    return_normal_gt=False,
):
    return PureRgbLdrDataset(
        ROOT=root,
        num_views=num_views,
        split=split,
        resolution=resolution,
        fps=fps,
        seed=seed,
        scene_names=scene_names,
        initial_scene_idx=initial_scene_idx,
        active_scene_count=active_scene_count,
        test_frame_count=test_frame_count,
        ldr_event_id=ldr_event_id,
        return_normal_gt=return_normal_gt,
    )
