import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from eventvggt.models.streamvggt import StreamVGGT as EventStreamVGGT


def get_args_parser():
    parser = argparse.ArgumentParser("Brief EventVGGT evaluation")
    parser.add_argument("--config", default="config/finetune_event.yaml", help="Path to finetune_event yaml")
    parser.add_argument("--weights", required=True, help="Checkpoint path. Supports raw state_dict or finetune checkpoint dict")
    parser.add_argument("--output_dir", default="eval_outputs/eventvggt", help="Directory for metrics and visualizations")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"], help="Dataset split to evaluate")
    parser.add_argument("--max_batches", default=0, type=int, help="0 means evaluate all batches")
    parser.add_argument("--num_workers", default=None, type=int, help="Override dataloader workers")
    parser.add_argument("--batch_size", default=None, type=int, help="Override eval batch size")
    parser.add_argument("--save_vis_batches", default=1, type=int, help="Number of first batches to save visual panels for")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading")
    parser.add_argument("--data_root", default=None, help="Override cfg.data.root")
    parser.add_argument("--num_views", default=None, type=int, help="Override cfg.data.num_views")
    parser.add_argument("--ldr_event_id", default=None, help="Override cfg.data.ldr_event_id, e.g. auto, 5, ev_10")
    return parser


def update_cfg_from_args(cfg, args):
    if args.data_root is not None:
        cfg.data.root = args.data_root
    if args.num_views is not None:
        cfg.data.num_views = args.num_views
    if args.ldr_event_id is not None:
        cfg.data.ldr_event_id = args.ldr_event_id
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    OmegaConf.resolve(cfg)
    return cfg


def build_eval_loader(cfg, split):
    dataset = get_combined_dataset(
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
        ldr_event_id=getattr(cfg.data, "ldr_event_id", "auto"),
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )


def move_views_to_device(views, device, dtype):
    for view in views:
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                view[key] = value.to(device=device, dtype=dtype if key != "events" else dtype)
    return views


def load_model(args, cfg, device):
    model = EventStreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=cfg.model.event_hidden_dim,
    )
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = fe.unwrap_state_dict(ckpt)
    msg = model.load_state_dict(state_dict, strict=args.strict)
    print(f"Loaded checkpoint: {args.weights}")
    print(f"load_state_dict: {msg}")
    model = model.to(device)
    model.eval()
    return model


def build_criterion(cfg):
    return fe.EventSupervisedLoss(
        pose_weight=cfg.loss.pose_weight,
        depth_weight=cfg.loss.depth_weight,
        points_weight=cfg.loss.points_weight,
        normal_weight=float(getattr(cfg.loss, "normal_weight", 0.0)),
        depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
        depth_max=(float(cfg.loss.depth_max) if getattr(cfg.loss, "depth_max", None) is not None else None),
        align_depth_scale_enabled=bool(getattr(cfg.loss, "align_depth_scale", True)),
        points_loss_type=str(getattr(cfg.loss, "points_loss_type", "l1")),
    )


def depth_metrics(depth_pred, depth_gt, valid_mask):
    pred = depth_pred.detach().float()
    gt = depth_gt.detach().float()
    mask = valid_mask.bool() & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
    if mask.sum() == 0:
        return {}

    pred = pred[mask].clamp_min(1e-6)
    gt = gt[mask].clamp_min(1e-6)
    abs_err = (pred - gt).abs()
    sq_err = (pred - gt) ** 2
    ratio = torch.maximum(pred / gt, gt / pred)
    return {
        "depth_mae": float(abs_err.mean().cpu()),
        "depth_rmse": float(torch.sqrt(sq_err.mean()).cpu()),
        "depth_abs_rel": float((abs_err / gt).mean().cpu()),
        "depth_delta1": float((ratio < 1.25).float().mean().cpu()),
        "valid_pixels": int(mask.sum().cpu()),
    }


def weighted_average(metric_rows):
    totals = {}
    weights = {}
    for row in metric_rows:
        weight = float(row.get("valid_pixels", 1))
        for key, value in row.items():
            if key == "valid_pixels":
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * weight
            weights[key] = weights.get(key, 0.0) + weight
    averaged = {key: totals[key] / max(weights[key], 1.0) for key in sorted(totals)}
    averaged["valid_pixels"] = int(sum(row.get("valid_pixels", 0) for row in metric_rows))
    return averaged


def save_batch_visuals(cfg, views, aux, output_dir, global_step):
    old_output_dir = cfg.output_dir
    old_vis_every = getattr(cfg.vis, "save_every_steps", 200)
    cfg.output_dir = str(output_dir)
    cfg.vis.save_every_steps = 1
    try:
        max_frames = min(len(views), 4)
        for frame_idx in range(max_frames):
            fe.save_training_visuals(cfg, views, aux, global_step, frame_idx=frame_idx, sample_idx=0)
    finally:
        cfg.output_dir = old_output_dir
        cfg.vis.save_every_steps = old_vis_every


def main(args):
    cfg = OmegaConf.load(args.config)
    cfg = update_cfg_from_args(cfg, args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args, cfg, device)
    criterion = build_criterion(cfg).to(device)
    loader = build_eval_loader(cfg, args.split)

    loss_rows = []
    depth_rows = []
    print(f"Evaluating split={args.split}, batches={len(loader)}, output_dir={output_dir}")

    with torch.no_grad():
        for batch_idx, views in enumerate(tqdm(loader)):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            views = fe.maybe_denormalize_views(views)
            dtype = next(model.parameters()).dtype
            views = move_views_to_device(views, device, dtype)

            model_output = model(views)
            loss, loss_details, aux = criterion(model_output, views)
            loss_row = {"loss": float(loss.detach().cpu()), **loss_details}
            loss_rows.append(loss_row)

            depth_rows.append(depth_metrics(aux["depth_pred"], aux["depth_gt"], aux["valid_mask"]))

            if batch_idx < args.save_vis_batches:
                save_batch_visuals(cfg, views, aux, output_dir, batch_idx)

    depth_rows = [row for row in depth_rows if row]
    summary = {}
    if loss_rows:
        for key in sorted(loss_rows[0].keys()):
            summary[key] = float(np.mean([row[key] for row in loss_rows if key in row]))
    if depth_rows:
        summary.update(weighted_average(depth_rows))

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": args.weights,
                "config": args.config,
                "split": args.split,
                "num_batches": len(loss_rows),
                "summary": summary,
            },
            f,
            indent=2,
        )

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
