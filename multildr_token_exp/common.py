"""Shared configuration for controlled Multi-LDR strategy comparisons."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from accelerate import Accelerator as HFAccelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_event as fe
from ablation.finetune_paper_ablation import FULL_RELIABILITY_WEIGHTS
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from fine_event.finetune_event_random_ldr import RandomLdrBatchSampler
from mul_loss_fine.finetune_mul_ldr_event import (
    MultiLdrBatchSampler,
    MultiLdrExposureLoss,
    _format_ldr,
    _to_list,
)
from mul_loss_fine.image_guided_event_reliability_loss import (
    make_configured_image_guided_event_reliability_loss,
)
from mul_loss_fine.launcher import configure_mul_loss_cfg
from multildr_token_exp.token_loss import wrap_token_consistency
from multildr_token_exp.token_model import StreamVGGT as TokenAlignedStreamVGGT


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("random_ldr_full", "paired_output_full", "paired_token_full")


class UnevenBatchAccelerator(HFAccelerator):
    def __init__(self, *args, **kwargs):
        signature = inspect.signature(HFAccelerator.__init__)
        if "even_batches" in signature.parameters:
            kwargs.setdefault("even_batches", False)
        super().__init__(*args, **kwargs)
        try:
            self.even_batches = False
        except Exception:
            pass
        config = getattr(self, "dataloader_config", None)
        if config is not None and hasattr(config, "even_batches"):
            config.even_batches = False


def _scene_names(cfg):
    path = Path(str(cfg.data.multildr_scene_manifest))
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scenes = list(manifest.get("training_scenes", []))
    if len(scenes) != 12:
        raise RuntimeError(f"Expected exactly 12 shared training scenes in {path}, got {scenes}")
    return scenes


def _dataset(cfg, split: str, ldr_event_id: str):
    scenes = _scene_names(cfg)
    dataset = get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=scenes,
        initial_scene_idx=0,
        active_scene_count=len(scenes),
        split=split,
        test_frame_count=cfg.data.test_frame_count,
        ldr_event_id=ldr_event_id,
        event_y_flip=getattr(cfg.data, "event_y_flip", "auto"),
        event_spatial_transform=getattr(cfg.data, "event_spatial_transform", "auto"),
        event_resize_method=getattr(cfg.data, "event_resize_method", "voxel_antialias"),
        event_resize_bins=getattr(cfg.data, "event_resize_bins", 10),
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    missing = sorted(set(scenes) - set(dataset.get_active_scenes()))
    if missing:
        raise RuntimeError(f"Shared Multi-LDR scenes are unavailable: {missing}")
    return dataset


def build_strategy_loader(cfg, split="train"):
    strategy = str(cfg.multildr_strategy)
    if split != "train":
        dataset = _dataset(cfg, split, str(cfg.data.eval_ldr_event_id))
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            drop_last=False,
            collate_fn=event_multiview_collate,
        )

    dataset = _dataset(cfg, split, "random")
    requested = [_format_ldr(value) for value in _to_list(cfg.data.multildr_train_ids)]
    available = dataset.get_active_ldr_events(common=True)
    missing = [value for value in requested if value not in available]
    if missing:
        raise ValueError(f"Requested LDR levels {missing} are unavailable; common={available}")

    if strategy == "random_ldr_full":
        sampler = RandomLdrBatchSampler(
            dataset,
            batch_size=cfg.batch_size,
            ldr_event_ids=requested,
            num_views=cfg.data.num_views,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
    else:
        sampler = MultiLdrBatchSampler(
            dataset,
            scenes_per_batch=1,
            ldr_event_ids=requested,
            num_views=cfg.data.num_views,
            exposures_per_sample=2,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        collate_fn=event_multiview_collate,
    )


def _build_token_model(cfg):
    return TokenAlignedStreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=cfg.model.event_hidden_dim,
        head_frames_chunk_size=cfg.model.head_frames_chunk_size,
        event_num_bins=cfg.model.event_num_bins,
        event_count_cmax=cfg.model.event_count_cmax,
        residual_scale=cfg.model.refiner_residual_scale,
        residual_highpass_kernel=cfg.model.event_delta_highpass_kernel,
        residual_patch_zero_mean=cfg.model.event_delta_patch_zero_mean,
        residual_patch_size=cfg.model.event_delta_patch_size,
        residual_abs_limit=cfg.model.event_delta_abs_limit,
        reliability_gate_enabled=True,
        reliability_gate_floor=cfg.model.event_reliability_gate_floor,
        reliability_init_bias=cfg.model.event_reliability_init_bias,
        refine_points=True,
        use_checkpoint=cfg.model.refiner_use_checkpoint,
        token_adapter_hidden_dim=cfg.model.token_adapter_hidden_dim,
        token_adapter_max_scale=cfg.model.token_adapter_max_scale,
    )


def configure_trainable(model, cfg):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "event_detail_refiner" in name:
            parameter.requires_grad = True
        if str(cfg.multildr_strategy) == "paired_token_full" and "exposure_token_adapter" in name:
            parameter.requires_grad = True
    for module_name in ("depth_head", "point_head"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(True)


def _paired_output_loss(base_loss, cfg):
    class PairedOutputLoss(MultiLdrExposureLoss, base_loss):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                exp_depth_weight=float(cfg.loss.ldr_output_depth_weight),
                exp_normal_weight=float(cfg.loss.ldr_output_normal_weight),
                exp_sat_boost=float(cfg.loss.ldr_output_sat_boost),
                exp_event_boost=float(cfg.loss.ldr_output_event_boost),
                exp_base_weight=float(cfg.loss.ldr_output_base_weight),
                exp_sat_threshold=float(cfg.loss.ldr_output_sat_threshold),
                **kwargs,
            )

    return PairedOutputLoss


def prepare_cfg(cfg, strategy: str):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown Multi-LDR strategy: {strategy}")
    OmegaConf.set_struct(cfg, False)
    for branch_name in ("model", "train", "loss", "data", "vis"):
        OmegaConf.set_struct(getattr(cfg, branch_name), False)

    cfg.multildr_strategy = strategy
    cfg.model.variant = "temporal_detail"
    cfg.model.event_num_bins = int(cfg.data.event_resize_bins)
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.035
    cfg.model.event_delta_highpass_kernel = 9
    cfg.model.event_delta_patch_zero_mean = True
    cfg.model.event_delta_patch_size = 14
    cfg.model.event_delta_abs_limit = 0.025
    cfg.model.event_reliability_gate_enabled = True
    cfg.model.event_reliability_gate_floor = 0.20
    cfg.model.event_reliability_init_bias = 0.0
    cfg.model.token_adapter_hidden_dim = int(getattr(cfg.model, "token_adapter_hidden_dim", 256))
    cfg.model.token_adapter_max_scale = float(getattr(cfg.model, "token_adapter_max_scale", 0.10))

    cfg.data.ldr_event_id = "random"
    cfg.data.eval_ldr_event_id = str(getattr(cfg.data, "eval_ldr_event_id", "ev_5"))
    cfg.data.multildr_train_ids = getattr(
        cfg.data, "multildr_train_ids", ["ev_1", "ev_2", "ev_5", "ev_10"]
    )
    cfg.data.return_normal_gt = True
    cfg.data.return_debug_event_fields = False
    cfg.data.active_scene_count = 12
    cfg.data.initial_scene_idx = 0

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    cfg.lr = min(float(cfg.lr), 4.0e-5)
    cfg.epochs = max(int(cfg.epochs), 20)
    cfg.eval_every_steps = 0

    if str(getattr(cfg, "pretrained", "") or "") in {"", "./ckpt/model.pt", "ckpt/model.pt"}:
        cfg.pretrained = str(ROOT / "ckpt" / "model.pt")

    output_root = Path(str(getattr(cfg, "strategy_output_root", ROOT / "abl_event_exp/multildr_token_strategy")))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cfg.exp_name = strategy
    cfg.save_dir = str(output_root)
    cfg.output_dir = str(output_root / strategy)
    cfg.logdir = str(Path(cfg.output_dir) / "logs")

    cfg.loss.ldr_output_depth_weight = 0.15
    cfg.loss.ldr_output_normal_weight = 0.10
    cfg.loss.ldr_output_sat_boost = 1.0
    cfg.loss.ldr_output_event_boost = 0.50
    cfg.loss.ldr_output_base_weight = 0.10
    cfg.loss.ldr_output_sat_threshold = 0.95
    cfg.loss.ldr_token_weight = float(getattr(cfg.loss, "ldr_token_weight", 0.05))
    configure_mul_loss_cfg(cfg, weights=FULL_RELIABILITY_WEIGHTS, exp_name=strategy)
    return cfg


def launch(cfg, strategy: str):
    cfg = prepare_cfg(cfg, strategy)
    base_loss = make_configured_image_guided_event_reliability_loss(cfg)
    if strategy == "paired_output_full":
        loss_class = _paired_output_loss(base_loss, cfg)
    elif strategy == "paired_token_full":
        loss_class = wrap_token_consistency(base_loss, cfg)
    else:
        loss_class = base_loss

    fe.Accelerator = UnevenBatchAccelerator
    fe.build_event_loader = build_strategy_loader
    fe.configure_trainable_params = configure_trainable
    fe.EventSupervisedLoss = loss_class
    if strategy == "paired_token_full":
        fe.build_event_model = _build_token_model

    print(
        f"[Multi-LDR strategy] {strategy}: scenes=12, "
        f"train_ids={list(cfg.data.multildr_train_ids)}, output={cfg.output_dir}"
    )
    fe.train(cfg)


__all__ = ["launch"]
