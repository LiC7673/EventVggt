import datetime
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from eventvggt.models.streamvggt import StreamVGGT as EventStreamVGGT
from eventvggt.utils.pose_enc import pose_encoding_to_extri_intri

import finetune_event as fe

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
printer = get_logger(__name__, log_level="INFO")


def ensure_homogeneous(c2w: torch.Tensor) -> torch.Tensor:
    if c2w.shape[-2:] == (3, 4):
        bottom_row = torch.tensor([0, 0, 0, 1], device=c2w.device, dtype=c2w.dtype)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(*c2w.shape[:-2], 1, 4)
        c2w = torch.cat([c2w, bottom_row], dim=-2)
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected pose shape [...,4,4] or [...,3,4], got {c2w.shape}")
    return c2w


def pose_encoding_to_c2w(pose_encoding: torch.Tensor, image_size_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    w2c_pred, intrinsics_pred = pose_encoding_to_extri_intri(
        pose_encoding,
        image_size_hw=image_size_hw,
    )
    B, S = w2c_pred.shape[:2]
    bottom_row = torch.tensor([0, 0, 0, 1], device=w2c_pred.device, dtype=w2c_pred.dtype)
    bottom_row = bottom_row.view(1, 1, 1, 4).expand(B, S, 1, 4)
    w2c_full = torch.cat([w2c_pred, bottom_row], dim=-2)
    c2w_pred = torch.linalg.inv(w2c_full)
    return c2w_pred, intrinsics_pred


def align_c2w_by_first_frame(pred_c2w: torch.Tensor, gt_c2w: torch.Tensor) -> torch.Tensor:
    pred_c2w = ensure_homogeneous(pred_c2w)
    gt_c2w = ensure_homogeneous(gt_c2w)

    first_pred = pred_c2w[..., 0, :, :]
    first_gt = gt_c2w[..., 0, :, :]
    alignment = torch.matmul(first_gt, torch.linalg.inv(first_pred))
    alignment = alignment.unsqueeze(1)
    aligned_c2w = torch.matmul(alignment, pred_c2w)
    return aligned_c2w


def c2w_to_pose_encoding(c2w: torch.Tensor, intrinsics: torch.Tensor, image_size_hw: Tuple[int, int]) -> torch.Tensor:
    c2w = ensure_homogeneous(c2w)
    w2c = torch.linalg.inv(c2w)
    return fe.extri_intri_to_pose_encoding(
        w2c[..., :3, :],
        intrinsics,
        image_size_hw=image_size_hw,
    )


class AlignedEventSupervisedLoss(fe.EventSupervisedLoss):
    def forward(self, model_output, views: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        pred = model_output.ress

        depth_pred = torch.stack([res["depth"] for res in pred], dim=1).squeeze(-1)
        points_pred = torch.stack([res["pts3d_in_other_view"] for res in pred], dim=1)
        pose_pred = torch.stack([res["camera_pose"] for res in pred], dim=1)

        depth_gt = fe.stack_view_field(views, "depthmap").to(device=depth_pred.device, dtype=depth_pred.dtype)
        intrinsics_gt = fe.stack_view_field(views, "camera_intrinsics").to(device=depth_pred.device, dtype=depth_pred.dtype)
        pose_matrix_gt = fe.stack_view_field(views, "camera_pose").to(device=depth_pred.device, dtype=depth_pred.dtype)
        valid_mask = fe.build_valid_mask(views, depth_gt, depth_min=self.depth_min, depth_max=self.depth_max)

        if self.align_depth_scale_enabled:
            depth_gt_aligned, depth_scales = fe.align_depth_scale(depth_pred.detach(), depth_gt, valid_mask)
        else:
            depth_gt_aligned = depth_gt
            depth_scales = depth_gt.new_ones(depth_gt.shape[:2])

        points_gt_aligned = fe.depth_to_world_points(depth_gt_aligned, intrinsics_gt, pose_matrix_gt)
        points_gt = points_gt_aligned
        points_mask = valid_mask.unsqueeze(-1).expand_as(points_gt)

        normal_mask = valid_mask.clone()
        normal_mask[..., 0, :] = False
        normal_mask[..., -1, :] = False
        normal_mask[..., :, 0] = False
        normal_mask[..., :, -1] = False

        if all("normal" in v for v in views):
            gt_normals = fe.stack_view_field(views, "normal")
            if gt_normals.ndim == 5 and gt_normals.shape[2] == 3:
                gt_normals = gt_normals.permute(0, 1, 3, 4, 2)
        else:
            gt_normals = fe.depth_to_normals(depth_gt, intrinsics_gt)
        pred_normals = fe.depth_to_normals(depth_pred, intrinsics_gt)
        normal_loss = fe.masked_cosine_loss(pred_normals, gt_normals, normal_mask)

        height, width = depth_gt.shape[-2:]
        pose_gt = fe.camera_pose_to_pose_encoding(
            pose_matrix_gt,
            intrinsics_gt,
            image_size_hw=(height, width),
        ).to(device=pose_pred.device, dtype=pose_pred.dtype)

        pred_c2w, pred_intrinsics = pose_encoding_to_c2w(pose_pred, image_size_hw=(height, width))
        pred_c2w_aligned = align_c2w_by_first_frame(pred_c2w, pose_matrix_gt)
        pose_pred_aligned = c2w_to_pose_encoding(pred_c2w_aligned, pred_intrinsics, image_size_hw=(height, width)).to(
            device=pose_pred.device,
            dtype=pose_pred.dtype,
        )

        if self.align_depth_scale_enabled:
            pose_pred_aligned[..., :3] = pose_pred_aligned[..., :3] * depth_scales.unsqueeze(-1)

        depth_loss = fe.masked_l1(depth_pred, depth_gt_aligned, valid_mask)
        points_loss = fe.masked_chamfer_distance(points_pred, points_gt, valid_mask)
        pose_loss = F.smooth_l1_loss(pose_pred_aligned, pose_gt)

        total_loss = (
            self.pose_weight * pose_loss
            + self.depth_weight * depth_loss
            + self.points_weight * points_loss
        )

        loss_details = {
            "pose_loss": float(pose_loss.detach()),
            "depth_loss": float(depth_loss.detach()),
            "points_loss": float(points_loss.detach()),
            "normal_loss": float(normal_loss.detach()) if self.normal_weight > 0.0 else 0.0,
        }

        aux = {
            "depth_pred": depth_pred,
            "depth_gt": depth_gt,
            "valid_mask": valid_mask,
            "points_pred": points_pred,
            "points_gt": points_gt,
        }

        return total_loss, loss_details, aux


def train(cfg):
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accum_iter,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
        dst_dir = fe.save_current_code(cfg.output_dir)
        printer.info("Saved current code snapshot to %s", dst_dir)

    seed = cfg.seed + accelerator.process_index
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = cfg.benchmark

    data_loader_train = fe.build_event_loader(cfg, split="train")
    data_loader_test = fe.build_event_loader(cfg, split="test")

    test_samples_count = len(data_loader_test) if data_loader_test else 0
    train_samples_count = len(data_loader_train)
    printer.info("Train dataset: %d samples, Test dataset: %d samples", train_samples_count, test_samples_count)

    model = EventStreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=cfg.model.event_hidden_dim,
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
    )

    if cfg.pretrained:
        printer.info("Loading model init weights from %s", cfg.pretrained)
        ckpt = fe.unwrap_state_dict(torch.load(cfg.pretrained, map_location="cpu"))
        msg = model.load_state_dict(ckpt, strict=False)
        printer.info("Checkpoint load result: %s", msg)

    fe.configure_trainable_params(model, cfg)
    fe.log_trainable_params(model)

    criterion = AlignedEventSupervisedLoss(
        pose_weight=cfg.loss.pose_weight,
        depth_weight=cfg.loss.depth_weight,
        points_weight=cfg.loss.points_weight,
        normal_weight=float(getattr(cfg.loss, "normal_weight", 0.0)),
        depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
        depth_max=(float(cfg.loss.depth_max) if getattr(cfg.loss, "depth_max", None) is not None else None),
        align_depth_scale_enabled=bool(getattr(cfg.loss, "align_depth_scale", True)),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were enabled. Check configure_trainable_params().")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    loss_scaler = NativeScaler(accelerator=accelerator)

    model, optimizer, data_loader_train, data_loader_test = accelerator.prepare(
        model,
        optimizer,
        data_loader_train,
        data_loader_test,
    )
    criterion = criterion.to(accelerator.device)

    log_writer = SummaryWriter(log_dir=cfg.logdir) if accelerator.is_main_process else None

    best_loss = float("inf")
    global_step = 0
    start_time = time.time()
    eval_every_steps = getattr(cfg, "eval_every_steps", max(cfg.save_every_steps, 500))

    for epoch in range(cfg.start_epoch, cfg.epochs):
        model.train()
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"

        for data_iter_step, views in enumerate(metric_logger.log_every(data_loader_train, cfg.print_freq, accelerator, header)):
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                views = fe.maybe_denormalize_views(views)

                if global_step % cfg.accum_iter == 0:
                    epoch_f = epoch + data_iter_step / max(len(data_loader_train), 1)
                    misc.adjust_learning_rate(optimizer, epoch_f, cfg)

                model_output = model(views)
                loss, loss_details, aux = criterion(model_output, views)

                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Non-finite loss detected: {loss_value}")

                loss_scaler(
                    loss,
                    optimizer,
                    parameters=[p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad],
                    update_grad=True,
                    clip_grad=cfg.clip_grad,
                )

                metric_logger.update(loss=loss_value)
                metric_logger.update(**loss_details)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                if accelerator.is_main_process and log_writer is not None and global_step % cfg.log_freq == 0:
                    log_writer.add_scalar("train/loss", loss_value, global_step)
                    log_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    for key, value in loss_details.items():
                        log_writer.add_scalar(f"train/{key}", value, global_step)

                if accelerator.is_main_process:
                    fe.save_training_visuals(cfg, views, aux, global_step)

                if (
                    test_samples_count > 0
                    and eval_every_steps > 0
                    and global_step % eval_every_steps == 0
                    and global_step > 0
                ):
                    if accelerator.is_main_process:
                        printer.info("Running test evaluation at step %d", global_step)
                    test_stats, _ = fe.evaluate_on_test_set(
                        model,
                        data_loader_test,
                        criterion,
                        accelerator,
                        cfg,
                        global_step,
                        log_writer=log_writer,
                    )
                    if accelerator.is_main_process:
                        fe.save_test_summary(cfg, epoch, global_step, test_stats)
                        fe.save_metrics_json(cfg, epoch, global_step, {}, test_stats)
                    model.train()

                if accelerator.is_main_process and global_step % cfg.save_every_steps == 0 and global_step > 0:
                    fe.save_checkpoint(accelerator, model, optimizer, loss_scaler, cfg, epoch, global_step, best_loss)

                best_loss = min(best_loss, loss_value)
                global_step += 1

        metric_logger.synchronize_between_processes(accelerator)
        epoch_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        printer.info("Epoch %d stats: %s", epoch, epoch_stats)

        if accelerator.is_main_process:
            with open(Path(cfg.output_dir) / "log.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "step": global_step, **epoch_stats}) + "\n")
            fe.save_metrics_json(cfg, epoch, global_step, epoch_stats, test_stats=None)

        fe.save_checkpoint(accelerator, model, optimizer, loss_scaler, cfg, epoch, global_step, best_loss)

    total_time = time.time() - start_time
    printer.info("Training finished in %.2f minutes", total_time / 60.0)

    if test_samples_count > 0:
        if accelerator.is_main_process:
            printer.info("Running final test evaluation")
        test_stats, _ = fe.evaluate_on_test_set(
            model,
            data_loader_test,
            criterion,
            accelerator,
            cfg,
            global_step,
            log_writer=log_writer,
        )
        if accelerator.is_main_process:
            fe.save_test_summary(cfg, epoch, global_step, test_stats)
            fe.save_metrics_json(cfg, epoch, global_step, {}, test_stats)
    if accelerator.is_main_process:
        fe.generate_loss_plots(cfg)

    if log_writer is not None:
        log_writer.close()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    train(cfg)


if __name__ == "__main__":
    run()
