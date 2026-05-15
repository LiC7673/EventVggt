import os.path as osp
import random
import itertools

import cv2
import h5py
import numpy as np
import PIL
import torch
import torch.nn.functional as F

from eventvggt.datasets.base.easy_dataset import EasyDataset
from eventvggt.datasets.utils.transforms import ImgNorm, SeqColorJitter
from eventvggt.utils.geometry import depthmap_to_absolute_camera_coordinates


class BaseEventMultiViewDataset(EasyDataset):
    """Base dataset for event-camera sequences.

    Unlike BaseMultiViewDataset, this class does not impose geometry-specific
    processing such as ray-map construction or depth-to-3D conversion. A sample
    is still represented as a list of per-frame `view` dicts.
    """

    def __init__(
        self,
        *,
        num_views=None,
        split=None,
        resolution=None,
        transform=ImgNorm,
        aug_crop=False,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
        fps=120,
    ):
        assert num_views is not None, "undefined num_views"
        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.fps = fps
        self.dt_us = int(1e6 / fps)
        print(f"Initialized {type(self).__name__} with {self.get_stats()} and resolutions {self._resolutions},views={self.num_views}, split={self.split}, seed={self.seed}, transform={transform}, aug_crop={aug_crop}, seq_aug_crop={seq_aug_crop}")
        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            transform = eval(transform)
        if transform == SeqColorJitter:
            transform = SeqColorJitter()
            self.is_seq_color_jitter = True
        self.transform = transform

        self.aug_crop = aug_crop
        self.seq_aug_crop = seq_aug_crop

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} groups of event views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        extra_index = ()
        if isinstance(idx, (tuple, list, np.ndarray)):
            raw_idx = idx
            if len(raw_idx) < 3:
                raise ValueError(f"Expected at least (idx, ar_idx, nview), got {raw_idx}")
            idx, ar_idx, nview = raw_idx[:3]
            extra_index = tuple(raw_idx[3:])
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            nview = self.num_views

        assert 1 <= nview <= self.num_views
        self._sample_extra_index = extra_index

        if self.seed is not None:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        resolution = self._resolutions[ar_idx]
        views = self._get_views(idx, resolution, self._rng, nview)
        assert len(views) == nview

        transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

        for v, view in enumerate(views):
            view["idx"] = (idx, ar_idx, v)
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = transform(view["img"])

            if "depthmap" in view:
                view["sky_mask"] = view["depthmap"] < 0

            assert "camera_intrinsics" in view, f"Missing camera_intrinsics for view {view_name(view)}"
            if "camera_pose" not in view:
                view["camera_pose"] = np.ones((4, 4), dtype=np.float32)
            else:
                assert np.isfinite(
                    view["camera_pose"]
                ).all(), f"NaN in camera pose for view {view_name(view)}"

            if "depthmap" in view:
                assert np.isfinite(
                    view["depthmap"]
                ).all(), f"NaN in depthmap for view {view_name(view)}"
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                    depthmap=view["depthmap"],
                    camera_intrinsics=view["camera_intrinsics"],
                    camera_pose=view["camera_pose"],
                )
                view["pts3d"] = pts3d
                view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

        for view in views:
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"
        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(height, int), f"Bad type for {height=} {type(height)=}, should be int"
            self._resolutions.append((width, height))

    @staticmethod
    def efficient_random_intervals(
        start,
        num_elements,
        interval_range,
        fixed_interval_prob=0.8,
        weights=None,
        seed=42,
    ):
        if random.random() < fixed_interval_prob:
            intervals = random.choices(interval_range, weights=weights) * (
                num_elements - 1
            )
        else:
            intervals = [
                random.choices(interval_range, weights=weights)[0]
                for _ in range(num_elements - 1)
            ]
        return list(itertools.accumulate([start] + intervals))

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        assert block_shuffle > 0
        blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
        shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
        return [item for block in shuffled_blocks for item in block]

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            if rng.random() < video_prob:
                if rng.random() < fix_interval_prob:
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = pos + rng.choice(
                pos_candidates, num_views - len(pos), replace=False
            ).tolist()
            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = pos * num_full_repeat + pos[: num_views - len(pos) * num_full_repeat]
            elif revisit_random < 0.9:
                pos = rng.choice(pos, num_views, replace=True)
            else:
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video

    @staticmethod
    def blender_to_opencv(poses):
        convert = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim == 2:
            return poses @ convert
        if poses.ndim == 3:
            return poses @ convert[None]
        raise ValueError(f"Unsupported pose shape: {poses.shape}")

    @staticmethod
    def load_depth_any(path, fallback_shape=None):
        if path is None or not osp.isfile(path):
            if fallback_shape is None:
                raise FileNotFoundError(f"Missing depth file: {path}")
            return np.ones(fallback_shape, dtype=np.float32)

        suffix = osp.splitext(path)[1].lower()
        if suffix == ".npy":
            depth = np.load(path)
        elif suffix == ".npz":
            with np.load(path) as data:
                if "depth" in data:
                    depth = data["depth"]
                else:
                    depth = data[list(data.keys())[0]]
        elif suffix == ".exr":
            # depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            # print("load exr")
            import os
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到深度图: {path}")
            depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if len(depth.shape) == 3:
                depth=depth[..., 0]  # 如果是 RGBA，只取第一通道
            # print(f"Loaded EXR depth from {path} with shape {depth.shape} and dtype {depth.dtype}") 
            if depth is None:
                raise ValueError(
                    f"Failed to read EXR depth file: {path}. "
                    "If OpenCV reports EXR support is disabled, set OPENCV_IO_ENABLE_OPENEXR=1."
                )
        else:
            depth = np.array(PIL.Image.open(path))

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth

    @staticmethod
    def load_depth_map(path, fallback_shape=None, normalize=True,smooth=True):
        """Load depth map and optionally normalize to [0, 1] range.
        
        Args:
            path: Path to depth file (.npy, .npz, .exr, or image format)
            fallback_shape: Shape to use if file is missing
            normalize: If True, normalize depth values to [0, 1] range
            
        Returns:
            Normalized depth map as float32 array in [0, 1] range if normalize=True
        """
        depth = BaseEventMultiViewDataset.load_depth_any(path, fallback_shape)
        
        if normalize:
            # Handle invalid values (inf, nan)
            valid_mask = np.isfinite(depth)
            
            if not valid_mask.any():
                # All invalid, return zeros
                return np.zeros_like(depth, dtype=np.float32)
            
            # Get min and max from valid values
            valid_depth = depth[valid_mask]
            depth_min = valid_depth.min()
            depth_max = valid_depth.max()
            
            # Avoid division by zero
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth, dtype=np.float32)
            
            # Clip to ensure all values are in [0, 1]
            depth = np.clip(depth, 0.0, 1.0)
        # if smooth:
        #     depth = cv2.GaussianBlur(depth, (3, 3), 0)
        return depth.astype(np.float32)

    @staticmethod
    def load_mask(path):
        mask = np.array(PIL.Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask > 254).astype(bool)

    @staticmethod
    def ensure_hwc3(array, name="normal"):
        array = np.asarray(array)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        elif array.ndim == 3:
            if array.shape[-1] in (3, 4):
                array = array[..., :3]
            elif array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            elif array.shape[0] in (3, 4):
                array = np.moveaxis(array[:3], 0, -1)
            elif array.shape[0] == 1:
                array = np.repeat(np.moveaxis(array, 0, -1), 3, axis=-1)
            else:
                raise ValueError(
                    f"Expected {name} map with shape [H,W,3] or [3,H,W], got {array.shape}"
                )
        else:
            raise ValueError(
                f"Expected {name} map with shape [H,W,3] or [3,H,W], got {array.shape}"
            )
        return array.astype(np.float32, copy=False)

    @staticmethod
    def load_normal_map(path, mask_path=None):
        normal = np.array(PIL.Image.open(path)).astype(np.float32)
        normal = BaseEventMultiViewDataset.ensure_hwc3(normal, name=f"normal map {path}")
 
        # if mask_path is not None and osp.isfile(mask_path):
        #     mask = BaseEventMultiViewDataset.load_mask(mask_path)
        #     normal[~mask] = 0.0
        # import cv2
        # cv2.imwrite(path.replace('.png', '_normal_vis.png').replace('.jpg', '_normal_vis.png'), (normal / normal.max() * 255).astype(np.uint8))
        # print(f"Loaded normal map from {path} with shape {normal.shape} and dtype {normal.dtype}")
        return normal

    # @staticmethod
    # def resize_hw_map(array, resolution, mode="bilinear"):
    #     target_w, target_h = resolution
    #     tensor = torch.as_tensor(array)
    #     original_dtype = tensor.dtype

    #     if tensor.ndim == 2:
    #         tensor = tensor.unsqueeze(0).unsqueeze(0).float()
    #     elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
    #         tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
    #     elif tensor.ndim == 3:
    #         tensor = tensor.unsqueeze(0).float()
    #     else:
    #         raise ValueError(f"Unsupported array shape: {tuple(tensor.shape)}")

    #     resized = F.interpolate(
    #         tensor,
    #         size=(target_h, target_w),
    #         mode=mode,
    #         align_corners=False if mode != "nearest" else None,
    #     ).squeeze(0)

    #     if resized.ndim == 3 and resized.shape[0] in (1, 3):
    #         resized = resized.permute(1, 2, 0)
    #         if resized.shape[-1] == 1:
    #             resized = resized[..., 0]

    #     if original_dtype == torch.bool:
    #         return (resized > 0.5).cpu().numpy().astype(bool)
    #     return resized.cpu().numpy().astype(np.float32)
    @staticmethod
    def resize_hw_map(array, resolution, mode="bicubic", mask=None):
        """
        改进版插值：
        1. 使用 bicubic 代替 bilinear 减少棋盘格纹理。
        2. 如果提供 mask，通过归一化插值消除边缘 0 值污染导致的飞点。
        """
        target_w, target_h = resolution
        tensor = torch.as_tensor(array)
        original_dtype = tensor.dtype

        # 统一维度到 [B, C, H, W]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0).float()
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).float()
        
        # 核心修改 1: 准备 Mask 进行归一化插值
        # 如果没传 mask，我们假设所有大于 0 的点都是有效的
        if mask is None:
            mask = (tensor > 0).float()
        else:
            mask = torch.as_tensor(mask).float()
            # 确保 mask 维度与 tensor 一致
            if mask.ndim == 2: mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[-1] == 1: mask = mask.permute(2,0,1).unsqueeze(0)

        # 屏蔽无效值
        masked_tensor = tensor * mask

        # 核心修改 2: 对数据和掩码同时进行高精度插值
        # 使用 bicubic 能够产生比 bilinear 更平滑的梯度，有效抑制棋盘纹
        # 注意：bicubic 必须配合 align_corners=True/False，通常建议 True 来减少偏移
        interp_mode = mode if mode != "area" else "bilinear" # area 不支持 float 掩码平滑
        
        resized_data = F.interpolate(
            masked_tensor,
            size=(target_h, target_w),
            mode=interp_mode,
            align_corners=True if interp_mode == "bicubic" else False
        )
        
        resized_mask = F.interpolate(
            mask,
            size=(target_h, target_w),
            mode=interp_mode,
            align_corners=True if interp_mode == "bicubic" else False
        )

        # 核心修改 3: 归一化修正 (消除边缘被 0 污染导致的深度下降)
        # 通过除以插值后的掩码，补偿了边缘像素因靠近背景 0 值而损失的权重
        resized = resized_data / resized_mask.clamp(min=1e-8)

        # 核心修改 4: 重新设置硬掩码，彻底切断飞点
        # 只有当插值后的权重足够高（说明周围大多是有效点）时才保留
        valid_threshold_mask = resized_mask > 0.5
        resized = resized * valid_threshold_mask

        # 恢复维度
        resized = resized.squeeze(0)
        if resized.ndim == 3 and resized.shape[0] in (1, 3):
            resized = resized.permute(1, 2, 0)
            if resized.shape[-1] == 1:
                resized = resized[..., 0]

        if original_dtype == torch.bool:
            return (resized > 0.5).cpu().numpy().astype(bool)
        return resized.cpu().numpy().astype(np.float32)
    @staticmethod
    def resize_image(image, resolution):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(np.asarray(image))
        return image.resize(tuple(resolution), PIL.Image.Resampling.BILINEAR)

    @staticmethod
    def scale_intrinsics(intrinsics, src_size, dst_size):
        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        sx = dst_w / src_w
        sy = dst_h / src_h
        intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
        intrinsics[0, 0] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[0, 2] *= sx
        intrinsics[1, 2] *= sy
        return intrinsics

    def resize_modalities(
        self,
        image,
        depthmap,
        intrinsics,
        resolution,
        *,
        normal=None,
        mask=None,
    ):
        src_w, src_h = image.size if isinstance(image, PIL.Image.Image) else image.shape[1::-1]
        image = self.resize_image(image, resolution)
        depthmap = self.resize_hw_map(depthmap, resolution, )
        intrinsics = self.scale_intrinsics(intrinsics, (src_w, src_h), resolution)

        resized = {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_intrinsics": intrinsics.astype(np.float32),
        }
        if normal is not None:
            normal = self.ensure_hwc3(normal, name="normal")
            resized["normal"] = self.resize_hw_map(normal, resolution,)
        if mask is not None:
            resized["mask"] = self.resize_hw_map(mask, resolution, mode="nearest")
        return resized

    @staticmethod
    def get_h5_length(path):
        with h5py.File(path, "r") as h5_file:
            if "events" not in h5_file:
                raise ValueError(f"Unsupported event format: {path}")
            return len(h5_file["events"])

    @staticmethod
    def _sample_h5_events(events_ds, max_count=200_000):
        total_events = len(events_ds)
        if total_events <= 0:
            return np.zeros((0, 4), dtype=np.float64)
        head_count = min(max_count // 2, total_events)
        tail_count = min(max_count - head_count, max(0, total_events - head_count))
        chunks = [events_ds[:head_count]]
        if tail_count > 0:
            chunks.append(events_ds[total_events - tail_count : total_events])
        return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)

    @staticmethod
    def _decode_h5_attr(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray) and value.size == 1:
            return BaseEventMultiViewDataset._decode_h5_attr(value.reshape(-1)[0])
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _normalize_time_unit(value):
        if value is None:
            return None
        unit = str(BaseEventMultiViewDataset._decode_h5_attr(value)).strip().lower()
        if unit in {"sec", "s", "second", "seconds"}:
            return "seconds"
        if unit in {"ms", "millisecond", "milliseconds"}:
            return "milliseconds"
        if unit in {"us", "microsecond", "microseconds"}:
            return "microseconds"
        if unit in {"ns", "nanosecond", "nanoseconds"}:
            return "nanoseconds"
        return None

    @staticmethod
    def _dt_from_time_unit(unit, fps):
        if unit == "seconds":
            return 1.0 / float(fps)
        if unit == "milliseconds":
            return 1_000.0 / float(fps)
        if unit == "microseconds":
            return 1_000_000.0 / float(fps)
        if unit == "nanoseconds":
            return 1_000_000_000.0 / float(fps)
        return None

    @staticmethod
    def _event_columns_from_attrs(attrs):
        columns_attr = attrs.get("columns", None)
        if columns_attr is not None:
            columns = str(BaseEventMultiViewDataset._decode_h5_attr(columns_attr)).lower().replace(" ", "")
            if columns in {"t,x,y,p", "[t,x,y,p]"}:
                return {"t": 0, "x": 1, "y": 2, "p": 3, "format": "t,x,y,p"}
            if columns in {"x,y,t,p", "[x,y,t,p]"}:
                return {"t": 2, "x": 0, "y": 1, "p": 3, "format": "x,y,t,p"}

        format_attr = attrs.get("format", None)
        if format_attr is not None:
            fmt = str(BaseEventMultiViewDataset._decode_h5_attr(format_attr)).lower().replace(" ", "")
            if "[t,x,y,p]" in fmt or "= [t,x,y,p]".replace(" ", "") in fmt:
                return {"t": 0, "x": 1, "y": 2, "p": 3, "format": "t,x,y,p"}
            if "[x,y,t,p]" in fmt:
                return {"t": 2, "x": 0, "y": 1, "p": 3, "format": "x,y,t,p"}

        return None

    @staticmethod
    def _infer_event_columns(sample):
        if sample.size == 0 or sample.ndim != 2 or sample.shape[1] < 4:
            return {"t": 0, "x": 1, "y": 2, "p": 3, "format": "t,x,y,p"}

        scores = []
        for col in range(sample.shape[1]):
            values = sample[:, col].astype(np.float64, copy=False)
            finite = np.isfinite(values)
            if finite.sum() < 2:
                scores.append((-np.inf, col))
                continue
            values = values[finite]
            diffs = np.diff(values)
            monotonic = float((diffs >= 0).mean()) if diffs.size else 0.0
            value_range = float(values.max() - values.min())
            unique_count = len(np.unique(values[: min(values.size, 5000)]))
            binary_like = unique_count <= 4 and values.min() >= -1.0 and values.max() <= 1.0
            score = monotonic * 50.0 + np.log1p(max(value_range, 0.0))
            if binary_like:
                score -= 20.0
            scores.append((score, col))

        time_col = max(scores)[1]
        remaining = [col for col in range(sample.shape[1]) if col != time_col]

        polarity_scores = []
        for col in remaining:
            values = sample[:, col].astype(np.float64, copy=False)
            finite = np.isfinite(values)
            values = values[finite]
            if values.size == 0:
                polarity_scores.append((-np.inf, col))
                continue
            rounded = np.unique(np.round(values[: min(values.size, 20000)], decimals=3))
            small_cardinality = len(rounded) <= 4
            signed_or_binary = values.min() >= -1.0 and values.max() <= 1.0
            integer_like = float(np.mean(np.isclose(values, np.round(values))))
            score = 0.0
            score += 8.0 if small_cardinality else 0.0
            score += 8.0 if signed_or_binary else 0.0
            score += integer_like
            polarity_scores.append((score, col))
        polarity_col = max(polarity_scores)[1] if polarity_scores else 3

        coord_cols = [col for col in range(sample.shape[1]) if col not in (time_col, polarity_col)]
        coord_cols = coord_cols[:2]
        if len(coord_cols) < 2:
            coord_cols = [1, 2]

        return {
            "t": int(time_col),
            "x": int(coord_cols[0]),
            "y": int(coord_cols[1]),
            "p": int(polarity_col),
            "format": f"col{time_col},col{coord_cols[0]},col{coord_cols[1]},col{polarity_col}",
        }

    def _infer_event_timing(self, sample, columns, frame_count, time_unit=None):
        if sample.size == 0:
            return {
                "unit": "microseconds",
                "dt": float(self.dt_us),
                "origin": 0.0,
                "first_t": 0.0,
                "last_t": 0.0,
            }

        t_values = sample[:, columns["t"]].astype(np.float64, copy=False)
        t_values = t_values[np.isfinite(t_values)]
        if t_values.size == 0:
            first_t = last_t = 0.0
        else:
            first_t = float(t_values.min())
            last_t = float(t_values.max())

        normalized_unit = self._normalize_time_unit(time_unit)
        if normalized_unit is not None:
            return {
                "unit": normalized_unit,
                "dt": float(self._dt_from_time_unit(normalized_unit, self.fps)),
                "origin": 0.0,
                "first_t": float(first_t),
                "last_t": float(last_t),
            }

        duration = max(last_t - first_t, 0.0)
        expected_steps = max(frame_count - 1, 1)
        candidates = [
            ("seconds", 1.0 / float(self.fps)),
            ("milliseconds", 1_000.0 / float(self.fps)),
            ("microseconds", 1_000_000.0 / float(self.fps)),
            ("nanoseconds", 1_000_000_000.0 / float(self.fps)),
        ]

        best_unit, best_dt, best_score = "microseconds", float(self.dt_us), float("inf")
        for unit, dt in candidates:
            expected_duration = expected_steps * dt
            score = abs(np.log((duration + 1e-9) / (expected_duration + 1e-9)))
            if score < best_score:
                best_unit, best_dt, best_score = unit, dt, score

        expected_total = max(frame_count - 1, 1) * float(best_dt)
        if first_t >= 0.0 and last_t <= expected_total + 2.0 * float(best_dt):
            origin = 0.0
        else:
            origin = 0.0 if abs(first_t) <= best_dt else first_t
        return {
            "unit": best_unit,
            "dt": float(best_dt),
            "origin": float(origin),
            "first_t": float(first_t),
            "last_t": float(last_t),
        }

    def build_frame_event_index(
        self,
        h5_path,
        frame_count,
        chunk_size=1_000_000,
        *,
        return_time_info=False,
    ):
        with h5py.File(h5_path, "r") as h5_file:
            if "events" not in h5_file:
                raise ValueError(f"Unsupported event format: {h5_path}")
            events_ds = h5_file["events"]
            attrs = {key: self._decode_h5_attr(value) for key, value in h5_file.attrs.items()}
            attrs.update({key: self._decode_h5_attr(value) for key, value in events_ds.attrs.items()})
            sample = self._sample_h5_events(events_ds)
            columns = self._event_columns_from_attrs(attrs) or self._infer_event_columns(sample)
            timing = self._infer_event_timing(sample, columns, frame_count, time_unit=attrs.get("time_unit"))

        raw_boundaries = timing["origin"] + np.arange(frame_count, dtype=np.float64) * float(timing["dt"])
        relative_boundaries = (raw_boundaries - float(timing["origin"])).astype(np.float32)
        boundaries = raw_boundaries
        boundary_indices = np.empty(len(boundaries), dtype=np.int64)
        boundary_pos = 0
        global_offset = 0

        with h5py.File(h5_path, "r") as h5_file:
            events_ds = h5_file["events"]
            total_events = len(events_ds)

            while global_offset < total_events and boundary_pos < len(boundaries):
                chunk_end = min(global_offset + chunk_size, total_events)
                timestamps = events_ds[global_offset:chunk_end, columns["t"]].astype(np.float64)
                if timestamps.size == 0:
                    break
                chunk_last = timestamps[-1]

                while boundary_pos < len(boundaries) and boundaries[boundary_pos] <= chunk_last:
                    local_idx = np.searchsorted(timestamps, boundaries[boundary_pos], side="left")
                    boundary_indices[boundary_pos] = global_offset + local_idx
                    boundary_pos += 1

                global_offset = chunk_end

        if boundary_pos < len(boundaries):
            boundary_indices[boundary_pos:] = self.get_h5_length(h5_path)

        frame_event_index = np.zeros((frame_count, 2), dtype=np.int64)
        for frame_idx in range(1, frame_count):
            frame_event_index[frame_idx, 0] = boundary_indices[frame_idx - 1]
            frame_event_index[frame_idx, 1] = boundary_indices[frame_idx]
        if not return_time_info:
            return frame_event_index

        time_info = {
            **timing,
            "columns": columns,
            "h5_attrs": attrs,
            "event_width": int(attrs["width"]) if "width" in attrs else None,
            "event_height": int(attrs["height"]) if "height" in attrs else None,
            "raw_first_boundary": float(raw_boundaries[0]) if raw_boundaries.size else 0.0,
            "raw_last_boundary": float(raw_boundaries[-1]) if raw_boundaries.size else 0.0,
        }
        return frame_event_index, relative_boundaries, time_info

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None
    ):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        src_size = image.size
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )

        image = self.resize_image(image, tuple(target_resolution))
        depthmap = self.resize_hw_map(depthmap, tuple(target_resolution), mode="bilinear")
        intrinsics = self.scale_intrinsics(intrinsics, src_size, tuple(target_resolution))

        if tuple(target_resolution) != tuple(resolution):
            image = self.resize_image(image, resolution)
            depthmap = self.resize_hw_map(depthmap, resolution, mode="bilinear")
            intrinsics = self.scale_intrinsics(
                intrinsics, tuple(target_resolution), tuple(resolution)
            )

        return image, depthmap, intrinsics

    @staticmethod
    def load_event_slice(path, start_idx, end_idx, *, event_columns=None, time_origin=0.0):
        with h5py.File(path, "r") as h5_file:
            if "events" not in h5_file:
                raise ValueError(f"Unsupported event format: {path}")
            events = h5_file["events"][start_idx:end_idx]

        if len(events) == 0:
            return {
                "event_xy": np.zeros((0, 2), dtype=np.int32),
                "event_t": np.zeros((0,), dtype=np.float32),
                "event_p": np.zeros((0,), dtype=np.float32),
                "events": np.zeros((0, 4), dtype=np.float32),
            }

        if event_columns is None:
            event_columns = {"t": 0, "x": 1, "y": 2, "p": 3}

        event_t = (events[:, event_columns["t"]].astype(np.float64) - float(time_origin)).astype(np.float32)
        event_x = events[:, event_columns["x"]].astype(np.int32)
        event_y = events[:, event_columns["y"]].astype(np.int32)
        event_p = events[:, event_columns["p"]].astype(np.float32)
        event_p[event_p == 0] = -1.0
        return {
            "event_xy": np.stack([event_x, event_y], axis=-1).astype(np.int32),
            "event_t": event_t,
            "event_p": event_p,
            "events": np.stack(
                [
                    event_x.astype(np.float32),
                    event_y.astype(np.float32),
                    event_t,
                    event_p,
                ],
                axis=-1,
            ).astype(np.float32),
        }


def is_good_type(key, v):
    if isinstance(v, (str, int, tuple)):
        return True, None
    if not hasattr(v, "dtype"):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"
