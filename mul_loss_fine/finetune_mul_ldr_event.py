from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from mul_loss_fine.event_supported_mv_loss import _make_event_support, _weighted_mean


def _format_ldr(value):
    value = str(value)
    return value if value.startswith("ev_") else f"ev_{value}"


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return list(value)


class MultiLdrBatchSampler:
    """Repeat each base sample with several LDR levels in the same batch."""

    def __init__(
        self,
        dataset,
        *,
        scenes_per_batch,
        ldr_event_ids,
        num_views,
        exposures_per_sample=2,
        shuffle=True,
        drop_last=True,
        seed=0,
    ):
        self.dataset = dataset
        self.scenes_per_batch = max(int(scenes_per_batch), 1)
        self.ldr_event_ids = [_format_ldr(x) for x in ldr_event_ids]
        self.num_views = int(num_views)
        self.exposures_per_sample = max(int(exposures_per_sample), 1)
        # Accelerate needs a declared, fixed batch size to shard a custom
        # batch sampler across ranks while preserving each exposure pair.
        self.batch_size = self.scenes_per_batch * self.exposures_per_sample
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.scenes_per_batch
        return (len(self.dataset) + self.scenes_per_batch - 1) // self.scenes_per_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        sample_indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng.shuffle(sample_indices)

        for batch_idx in range(len(self)):
            start = batch_idx * self.scenes_per_batch
            end = start + self.scenes_per_batch
            base_indices = sample_indices[start:end]
            if len(base_indices) < self.scenes_per_batch and self.drop_last:
                continue

            batch = []
            for sample_idx in base_indices:
                replace = self.exposures_per_sample > len(self.ldr_event_ids)
                chosen = rng.choice(self.ldr_event_ids, size=self.exposures_per_sample, replace=replace)
                for ldr_event_id in chosen:
                    batch.append((int(sample_idx), 0, self.num_views, str(ldr_event_id)))
            yield batch

        self.epoch += 1


def _build_dataset(cfg, split, ldr_event_id):
    return get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=cfg.data.scene_names if cfg.data.scene_names else None,
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=ldr_event_id,
        event_y_flip=getattr(cfg.data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(cfg.data, "event_spatial_transform", "auto"),
        event_resize_method=getattr(cfg.data, "event_resize_method", "voxel_antialias"),
        event_resize_bins=getattr(cfg.data, "event_resize_bins", 10),
        return_normal_gt=getattr(cfg.data, "return_normal_gt", False),
        return_debug_event_fields=getattr(cfg.data, "return_debug_event_fields", False),
    )


def build_mul_ldr_loader(cfg, split="train"):
    if split == "train":
        dataset = _build_dataset(cfg, split, getattr(cfg.data, "ldr_event_id", "random"))
    else:
        dataset = _build_dataset(cfg, split, getattr(cfg.data, "eval_ldr_event_id", "auto"))

    if len(dataset) <= 0:
        raise ValueError(
            f"Dataset has no valid samples under {cfg.data.root}. "
            f"num_views={cfg.data.num_views}, active_scenes={dataset.get_active_scenes()}"
        )

    if split == "train":
        requested = [_format_ldr(x) for x in _to_list(getattr(cfg.data, "mul_ldr_train_ids", []))]
        available = dataset.get_active_ldr_events(common=True)
        ldr_event_ids = requested or available
        missing = [ldr for ldr in ldr_event_ids if ldr not in available]
        if missing:
            raise ValueError(f"Requested LDR ids {missing} are not common in active scenes. Available: {available}")

        sampler = MultiLdrBatchSampler(
            dataset,
            scenes_per_batch=getattr(cfg.data, "mul_ldr_scenes_per_batch", cfg.batch_size),
            ldr_event_ids=ldr_event_ids,
            num_views=cfg.data.num_views,
            exposures_per_sample=getattr(cfg.data, "mul_ldr_exposures_per_sample", 2),
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=event_multiview_collate,
        )
        print(
            f"Mul-LDR train loader: scenes_per_batch={sampler.scenes_per_batch}, "
            f"actual_batch={sampler.batch_size}, "
            f"ldr_event_ids={ldr_event_ids}"
        )
        return loader

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    print(
        f"Mul-LDR {split} loader: eval_ldr={getattr(cfg.data, 'eval_ldr_event_id', 'auto')}, "
        f"active_scenes={dataset.get_active_scenes()}, samples={len(dataset)}"
    )
    return loader


def _batch_group_keys(views: List[Dict[str, torch.Tensor]], batch_size: int):
    keys = []
    for batch_idx in range(batch_size):
        instances = []
        for view in views:
            instance = view.get("instance", "")
            if isinstance(instance, (list, tuple)):
                instances.append(str(instance[batch_idx]))
            else:
                instances.append(str(instance))
        keys.append(tuple(instances))
    return keys


def _image_saturation_weight(views, *, threshold, height, width, device, dtype):
    images = fe.stack_view_field(views, "img").to(device=device, dtype=dtype)
    if images.min() < -0.05:
        images = images * 0.5 + 0.5
    sat = images.amax(dim=2) > float(threshold)
    if sat.shape[-2:] != (height, width):
        sat = F.interpolate(sat.flatten(0, 1).float().unsqueeze(1), size=(height, width), mode="nearest")
        sat = sat.squeeze(1).view(images.shape[0], images.shape[1], height, width).bool()
    return sat


class MultiLdrExposureLoss(fe.EventSupervisedLoss):
    def __init__(
        self,
        *args,
        exp_depth_weight=0.3,
        exp_normal_weight=0.2,
        exp_sat_boost=1.0,
        exp_event_boost=0.5,
        exp_base_weight=0.1,
        exp_sat_threshold=0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exp_depth_weight = float(exp_depth_weight)
        self.exp_normal_weight = float(exp_normal_weight)
        self.exp_sat_boost = float(exp_sat_boost)
        self.exp_event_boost = float(exp_event_boost)
        self.exp_base_weight = float(exp_base_weight)
        self.exp_sat_threshold = float(exp_sat_threshold)

    def forward(self, model_output, views):
        base_loss, details, aux = super().forward(model_output, views)
        if self.exp_depth_weight <= 0.0 and self.exp_normal_weight <= 0.0:
            details.update({"ldr_exp_loss": 0.0, "ldr_exp_depth_loss": 0.0, "ldr_exp_normal_loss": 0.0})
            return base_loss, details, aux

        pred = model_output.ress
        depth_pred = torch.stack([res["depth"] for res in pred], dim=1).squeeze(-1)
        batch, seq_len, height, width = depth_pred.shape
        if batch <= 1:
            details.update({"ldr_exp_loss": 0.0, "ldr_exp_depth_loss": 0.0, "ldr_exp_normal_loss": 0.0})
            return base_loss, details, aux

        intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(device=depth_pred.device, dtype=depth_pred.dtype)
        depth_gt = fe.stack_view_field(views, "depthmap").to(device=depth_pred.device, dtype=depth_pred.dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)
        pred_normals = fe.depth_to_normals(depth_pred.clamp_min(self.depth_min), intrinsics)

        sat = _image_saturation_weight(
            views,
            threshold=self.exp_sat_threshold,
            height=height,
            width=width,
            device=depth_pred.device,
            dtype=depth_pred.dtype,
        )
        event_support = _make_event_support(
            views,
            height=height,
            width=width,
            device=depth_pred.device,
            dtype=depth_pred.dtype,
            blur_kernel=5,
            dilate_kernel=3,
            threshold=0.02,
        ).detach()

        groups = {}
        for batch_idx, key in enumerate(_batch_group_keys(views, batch)):
            groups.setdefault(key, []).append(batch_idx)

        depth_terms = []
        normal_terms = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            ref = indices[0]
            for other in indices[1:]:
                pair_valid = (valid_mask[ref] & valid_mask[other]).unsqueeze(1).to(dtype=depth_pred.dtype)
                sat_union = (sat[ref] | sat[other]).unsqueeze(1).to(dtype=depth_pred.dtype)
                event_pair = torch.maximum(event_support[ref], event_support[other]).unsqueeze(1)
                weight = pair_valid * (
                    self.exp_base_weight
                    + self.exp_sat_boost * sat_union
                    + self.exp_event_boost * event_pair
                )

                if self.exp_depth_weight > 0.0:
                    depth_a = torch.log(depth_pred[ref].clamp_min(self.depth_min))
                    depth_b = torch.log(depth_pred[other].clamp_min(self.depth_min))
                    depth_terms.append(_weighted_mean((depth_a - depth_b).abs().unsqueeze(1), weight))

                if self.exp_normal_weight > 0.0:
                    normal_a = F.normalize(pred_normals[ref], dim=-1, eps=1e-6)
                    normal_b = F.normalize(pred_normals[other], dim=-1, eps=1e-6)
                    normal_loss = 1.0 - (normal_a * normal_b).sum(dim=-1).clamp(-1.0, 1.0)
                    normal_terms.append(_weighted_mean(normal_loss.unsqueeze(1), weight))

        zero = depth_pred.new_tensor(0.0)
        depth_loss = torch.stack(depth_terms).mean() if depth_terms else zero
        normal_loss = torch.stack(normal_terms).mean() if normal_terms else zero
        exp_loss = self.exp_depth_weight * depth_loss + self.exp_normal_weight * normal_loss

        details.update(
            {
                "ldr_exp_loss": float(exp_loss.detach()),
                "ldr_exp_depth_loss": float(depth_loss.detach()),
                "ldr_exp_normal_loss": float(normal_loss.detach()),
            }
        )
        return base_loss + exp_loss, details, aux


def make_configured_loss(cfg):
    class ConfiguredMultiLdrLoss(MultiLdrExposureLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                exp_depth_weight=float(getattr(cfg.loss, "ldr_exp_depth_weight", 0.3)),
                exp_normal_weight=float(getattr(cfg.loss, "ldr_exp_normal_weight", 0.2)),
                exp_sat_boost=float(getattr(cfg.loss, "ldr_exp_sat_boost", 1.0)),
                exp_event_boost=float(getattr(cfg.loss, "ldr_exp_event_boost", 0.5)),
                exp_base_weight=float(getattr(cfg.loss, "ldr_exp_base_weight", 0.1)),
                exp_sat_threshold=float(getattr(cfg.loss, "ldr_exp_sat_threshold", 0.95)),
                **kwargs,
            )

    return ConfiguredMultiLdrLoss


def _ensure_defaults(cfg):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    OmegaConf.set_struct(cfg.loss, False)
    OmegaConf.set_struct(cfg.vis, False)
    cfg.data.ldr_event_id = getattr(cfg.data, "ldr_event_id", "random")
    cfg.data.eval_ldr_event_id = getattr(cfg.data, "eval_ldr_event_id", "auto")
    cfg.data.mul_ldr_train_ids = getattr(cfg.data, "mul_ldr_train_ids", ["ev_2", "ev_5", "ev_10"])
    cfg.data.mul_ldr_exposures_per_sample = int(getattr(cfg.data, "mul_ldr_exposures_per_sample", 2))
    cfg.data.mul_ldr_scenes_per_batch = int(getattr(cfg.data, "mul_ldr_scenes_per_batch", cfg.batch_size))
    cfg.loss.ldr_exp_depth_weight = float(getattr(cfg.loss, "ldr_exp_depth_weight", 0.3))
    cfg.loss.ldr_exp_normal_weight = float(getattr(cfg.loss, "ldr_exp_normal_weight", 0.2))
    cfg.loss.ldr_exp_sat_boost = float(getattr(cfg.loss, "ldr_exp_sat_boost", 1.0))
    cfg.loss.ldr_exp_event_boost = float(getattr(cfg.loss, "ldr_exp_event_boost", 0.5))
    cfg.loss.ldr_exp_base_weight = float(getattr(cfg.loss, "ldr_exp_base_weight", 0.1))
    cfg.loss.ldr_exp_sat_threshold = float(getattr(cfg.loss, "ldr_exp_sat_threshold", 0.95))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "test_max_batches", 2))
    cfg.vis.test_num_views = int(getattr(cfg.vis, "test_num_views", 6))
    if str(getattr(cfg, "exp_name", "")) == "event_finetune_LDR5":
        cfg.exp_name = "mul_ldr_event"
    OmegaConf.resolve(cfg)


def launch(cfg):
    _ensure_defaults(cfg)
    fe.build_event_loader = build_mul_ldr_loader
    fe.EventSupervisedLoss = make_configured_loss(cfg)
    print(
        "Mul-LDR training: "
        f"train_ids={_to_list(cfg.data.mul_ldr_train_ids)}, "
        f"exposures_per_sample={cfg.data.mul_ldr_exposures_per_sample}, "
        f"eval_ldr={cfg.data.eval_ldr_event_id}"
    )
    fe.train(cfg)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    launch(cfg)


if __name__ == "__main__":
    run()
