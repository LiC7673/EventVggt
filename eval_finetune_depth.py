import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset


DEFAULT_EXPERIMENTS = {
    "event": {
        "config": "config/finetune_event.yaml",
        "checkpoint": "checkpoints/event_finetune_LDR5/checkpoint-last.pth",
        "model_type": "event",
    },
    "normal_consistency": {
        "config": "config/finetune_event_normal_consistency.yaml",
        "checkpoint": "checkpoints/event_finetune_normal_consistency/checkpoint-last.pth",
        "model_type": "event",
    },
    "global_token": {
        "config": "config/finetune_event_global_token.yaml",
        "checkpoint": "checkpoints/event_finetune_global_token/checkpoint-last.pth",
        "model_type": "global_token",
    },
    "global_local": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/event_finetune_global_local/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_rgb_coarse": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_rgb_coarse/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_global_only": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_global_only/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_local_only": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_local_only/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_global_local": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_global_local/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_tokens_4": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_tokens_4/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_tokens_16": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_tokens_16/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_tokens_64": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_tokens_64/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_tokens_256": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_tokens_256/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_event_h16": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_event_h16/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_event_h8": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_event_h8/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_event_h4": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_event_h4/checkpoint-last.pth",
        "model_type": "global_local",
    },
    "ablation_event_h2": {
        "config": "config/finetune_event_global_local.yaml",
        "checkpoint": "checkpoints/ablation_event_h2/checkpoint-last.pth",
        "model_type": "global_local",
    },
}


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def infer_model_type(name: str, config_path: Path, cfg) -> str:
    text = f"{name} {config_path}".lower()
    if "global_local" in text or "ablation_" in text or hasattr(cfg.model, "branch_mode"):
        return "global_local"
    if "global_token" in text:
        return "global_token"
    return "event"


def build_model(cfg, model_type: str):
    model_kwargs = dict(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=cfg.model.event_hidden_dim,
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
    )

    if model_type == "global_local":
        from eventvggt.models.streamvggt_global_local import StreamVGGT

        model_kwargs.update(
            branch_mode=str(getattr(cfg.model, "branch_mode", "global_local")),
            num_global_tokens=int(getattr(cfg.model, "num_global_tokens", 16)),
            event_downsample=int(getattr(cfg.model, "event_downsample", 4)),
            global_num_heads=int(getattr(cfg.model, "global_num_heads", 8)),
            global_inject_layers=list(getattr(cfg.model, "global_inject_layers", [23])),
            detail_hidden_dim=int(getattr(cfg.model, "detail_hidden_dim", 128)),
            residual_scale=float(getattr(cfg.model, "residual_scale", 0.1)),
        )
        return StreamVGGT(**model_kwargs)

    if model_type == "global_token":
        from eventvggt.models.streamvggt_global_token import StreamVGGT

        return StreamVGGT(**model_kwargs)

    from eventvggt.models.streamvggt import StreamVGGT

    return StreamVGGT(**model_kwargs)


def build_test_loader(cfg, args):
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.batch_size)
    num_workers = int(args.num_workers)
    pin_memory = bool(args.pin_memory)
    dataset = get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=cfg.data.scene_names if cfg.data.scene_names else None,
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split="test",
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=getattr(cfg.data, "ldr_event_id", "auto"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )


def move_views_to_device(views: List[Dict], device: torch.device, dtype: torch.dtype):
    for view in views:
        for key, value in list(view.items()):
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    view[key] = value.to(device=device, dtype=dtype, non_blocking=True)
                else:
                    view[key] = value.to(device=device, non_blocking=True)
            elif isinstance(value, list):
                moved = []
                for item in value:
                    if torch.is_tensor(item):
                        moved.append(item.to(device=device, non_blocking=True))
                    else:
                        moved.append(item)
                view[key] = moved
    return views


def stack_depth_field(model_output, key: str) -> Optional[torch.Tensor]:
    if not getattr(model_output, "ress", None):
        return None
    if not all(key in res for res in model_output.ress):
        return None
    value = torch.stack([res[key] for res in model_output.ress], dim=1)
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    return value


def build_valid_mask(views, depth_gt: torch.Tensor, depth_min: float, depth_max: Optional[float]) -> torch.Tensor:
    if "valid_mask" in views[0]:
        mask = torch.stack([view["valid_mask"] for view in views], dim=1)
    elif "mask" in views[0]:
        mask = torch.stack([view["mask"] for view in views], dim=1)
    else:
        mask = depth_gt > 0

    mask = mask.bool() & torch.isfinite(depth_gt) & (depth_gt > depth_min)
    if depth_max is not None and depth_max > 0:
        mask = mask & (depth_gt < depth_max)
    return mask


def median_align_prediction(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    aligned = pred.clone()
    flat_pred = aligned.reshape(-1, *aligned.shape[-2:])
    flat_gt = gt.reshape(-1, *gt.shape[-2:])
    flat_valid = valid.reshape(-1, *valid.shape[-2:])
    scales = []
    for idx in range(flat_pred.shape[0]):
        mask = flat_valid[idx] & torch.isfinite(flat_pred[idx]) & (flat_pred[idx] > eps)
        if mask.sum() < 16:
            scale = pred.new_tensor(1.0)
        else:
            scale = flat_gt[idx][mask].median() / flat_pred[idx][mask].median().clamp_min(eps)
            scale = scale.clamp(1e-3, 1e3)
            flat_pred[idx] = flat_pred[idx] * scale
        scales.append(scale)
    return aligned, torch.stack(scales).mean()


class DepthMetricAccumulator:
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.abs_rel = 0.0
        self.sq_rel = 0.0
        self.mae = 0.0
        self.rmse_sq = 0.0
        self.rmse_log_sq = 0.0
        self.log_diff = 0.0
        self.log_diff_sq = 0.0
        self.delta1 = 0.0
        self.delta2 = 0.0
        self.delta3 = 0.0
        self.scale_sum = 0.0
        self.scale_count = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, *, median_align: bool = False):
        eps = 1e-6
        valid = mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > eps) & (pred > eps)
        if median_align:
            pred, scale = median_align_prediction(pred, gt, valid, eps=eps)
            self.scale_sum += float(scale.detach().cpu())
            self.scale_count += 1
            valid = mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > eps) & (pred > eps)

        if valid.sum() == 0:
            return

        pred_v = pred[valid].float().clamp_min(eps)
        gt_v = gt[valid].float().clamp_min(eps)
        diff = pred_v - gt_v
        abs_diff = diff.abs()
        sq_diff = diff.square()
        log_diff = torch.log(pred_v) - torch.log(gt_v)
        ratio = torch.maximum(pred_v / gt_v, gt_v / pred_v)
        n = int(pred_v.numel())

        self.count += n
        self.abs_rel += float((abs_diff / gt_v).sum().detach().cpu())
        self.sq_rel += float((sq_diff / gt_v).sum().detach().cpu())
        self.mae += float(abs_diff.sum().detach().cpu())
        self.rmse_sq += float(sq_diff.sum().detach().cpu())
        self.rmse_log_sq += float(log_diff.square().sum().detach().cpu())
        self.log_diff += float(log_diff.sum().detach().cpu())
        self.log_diff_sq += float(log_diff.square().sum().detach().cpu())
        self.delta1 += float((ratio < 1.25).sum().detach().cpu())
        self.delta2 += float((ratio < 1.25**2).sum().detach().cpu())
        self.delta3 += float((ratio < 1.25**3).sum().detach().cpu())

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"prediction": self.name, "pixels": 0}

        count = float(self.count)
        mean_log = self.log_diff / count
        silog_var = max(self.log_diff_sq / count - mean_log * mean_log, 0.0)
        result = {
            "prediction": self.name,
            "pixels": self.count,
            "abs_rel": self.abs_rel / count,
            "sq_rel": self.sq_rel / count,
            "mae": self.mae / count,
            "rmse": math.sqrt(self.rmse_sq / count),
            "rmse_log": math.sqrt(self.rmse_log_sq / count),
            "silog": math.sqrt(silog_var) * 100.0,
            "delta1": self.delta1 / count,
            "delta2": self.delta2 / count,
            "delta3": self.delta3 / count,
        }
        if self.scale_count > 0:
            result["median_scale"] = self.scale_sum / self.scale_count
        return result


def save_depth_visuals(out_dir: Path, experiment: str, batch_idx: int, views, predictions, depth_gt, valid_mask, limit: int):
    if limit <= 0 or batch_idx >= limit:
        return
    vis_dir = out_dir / "vis" / experiment
    vis_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = 0
    frame_count = min(depth_gt.shape[1], 4)
    for frame_idx in range(frame_count):
        rgb = fe.tensor_rgb_to_uint8(views[frame_idx]["img"][sample_idx], valid_mask[sample_idx, frame_idx])
        panels = [
            fe.make_labeled_panel("rgb", rgb),
            fe.make_labeled_panel("gt_depth", fe.depth_to_uint8(depth_gt[sample_idx, frame_idx], valid_mask[sample_idx, frame_idx])),
        ]
        for pred_name, pred in predictions.items():
            panels.append(
                fe.make_labeled_panel(
                    pred_name,
                    fe.depth_to_uint8(pred[sample_idx, frame_idx], valid_mask[sample_idx, frame_idx]),
                )
            )

        total_width = sum(panel.width for panel in panels)
        max_height = max(panel.height for panel in panels)
        from PIL import Image

        canvas = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
        x_offset = 0
        for panel in panels:
            canvas.paste(panel, (x_offset, 0))
            x_offset += panel.width
        canvas.save(vis_dir / f"batch_{batch_idx:04d}_frame_{frame_idx:02d}.png")


def load_config_and_checkpoint(config_path: Path, checkpoint_path: Path, overrides, use_checkpoint_cfg: bool):
    cfg = OmegaConf.load(config_path)
    ckpt = torch_load(checkpoint_path)
    if use_checkpoint_cfg and isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(ckpt["cfg"]))
    if isinstance(ckpt, dict):
        ckpt.pop("optimizer", None)
        ckpt.pop("loss_scaler", None)
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.resolve(cfg)
    return cfg, ckpt


def evaluate_one(name: str, spec: Dict, args, overrides, out_dir: Path) -> List[Dict]:
    config_path = Path(spec["config"])
    checkpoint_path = Path(spec["checkpoint"])
    if not checkpoint_path.exists():
        message = f"[skip] {name}: checkpoint not found: {checkpoint_path}"
        if args.strict_missing:
            raise FileNotFoundError(message)
        print(message)
        return []
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found for {name}: {config_path}")

    cfg, ckpt = load_config_and_checkpoint(
        config_path,
        checkpoint_path,
        overrides,
        use_checkpoint_cfg=not args.ignore_checkpoint_cfg,
    )
    model_type = spec.get("model_type") or infer_model_type(name, config_path, cfg)
    model_type = infer_model_type(name, config_path, cfg) if model_type == "auto" else model_type

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch.float32
    model = build_model(cfg, model_type)
    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))
    load_msg = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    data_loader = build_test_loader(cfg, args)
    print(
        f"[eval] {name}: model={model_type}, checkpoint={checkpoint_path}, "
        f"test_batches={len(data_loader)}, num_views={cfg.data.num_views}, load={load_msg}"
    )

    metric_sets = {
        "final": DepthMetricAccumulator("final"),
    }
    if args.median_align:
        metric_sets["final_median_aligned"] = DepthMetricAccumulator("final_median_aligned")

    residual_abs_sum = 0.0
    residual_abs_count = 0
    start_time = time.time()

    for batch_idx, views in enumerate(data_loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        views = fe.maybe_denormalize_views(views)
        views = move_views_to_device(views, device, dtype)

        with torch.no_grad():
            use_amp = args.amp != "none" and device.type == "cuda"
            amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                model_output = model(views)

        depth_pred = stack_depth_field(model_output, "depth")
        depth_coarse = stack_depth_field(model_output, "depth_coarse")
        depth_residual = stack_depth_field(model_output, "depth_residual")
        if depth_pred is None:
            raise RuntimeError(f"{name} did not return depth predictions")

        depth_gt = fe.stack_view_field(views, "depthmap").to(device=device, dtype=depth_pred.dtype)
        depth_max = getattr(cfg.loss, "depth_max", None)
        depth_max = float(depth_max) if depth_max is not None else None
        valid_mask = build_valid_mask(
            views,
            depth_gt,
            depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
            depth_max=depth_max,
        )

        predictions = {"final": depth_pred}
        if depth_coarse is not None:
            predictions["coarse"] = depth_coarse
            metric_sets.setdefault("coarse", DepthMetricAccumulator("coarse"))
            if args.median_align:
                metric_sets.setdefault("coarse_median_aligned", DepthMetricAccumulator("coarse_median_aligned"))
        if depth_residual is not None:
            valid_float = valid_mask.to(dtype=depth_residual.dtype)
            residual_abs_sum += float((depth_residual.abs() * valid_float).sum().detach().cpu())
            residual_abs_count += int(valid_mask.sum().detach().cpu())

        for pred_name, pred in predictions.items():
            metric_sets[pred_name].update(pred, depth_gt, valid_mask, median_align=False)
            aligned_name = f"{pred_name}_median_aligned"
            if args.median_align and aligned_name in metric_sets:
                metric_sets[aligned_name].update(pred, depth_gt, valid_mask, median_align=True)

        save_depth_visuals(out_dir, name, batch_idx, views, predictions, depth_gt, valid_mask, args.save_vis)

        if (batch_idx + 1) % args.print_freq == 0:
            elapsed = time.time() - start_time
            print(f"[eval] {name}: batch {batch_idx + 1}/{len(data_loader)} elapsed={elapsed:.1f}s")

    rows = []
    residual_abs = residual_abs_sum / max(residual_abs_count, 1)
    for metric in metric_sets.values():
        row = metric.compute()
        row.update(
            {
                "experiment": name,
                "model_type": model_type,
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "depth_residual_abs": residual_abs if residual_abs_count > 0 else "",
            }
        )
        rows.append(row)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def write_results(rows: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "experiment",
        "prediction",
        "model_type",
        "pixels",
        "abs_rel",
        "sq_rel",
        "mae",
        "rmse",
        "rmse_log",
        "silog",
        "delta1",
        "delta2",
        "delta3",
        "median_scale",
        "depth_residual_abs",
        "checkpoint",
        "config",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved results:\n  {csv_path}\n  {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned EventVGGT variants on the test split with depth metrics."
    )
    parser.add_argument(
        "--experiments",
        default="event,normal_consistency,global_token,global_local",
        help="Comma-separated experiment names from DEFAULT_EXPERIMENTS, or 'all'.",
    )
    parser.add_argument("--config", default=None, help="Single-run config path.")
    parser.add_argument("--checkpoint", default=None, help="Single-run checkpoint path.")
    parser.add_argument("--model-type", default="auto", choices=["auto", "event", "global_token", "global_local"])
    parser.add_argument("--name", default="custom", help="Single-run experiment name.")
    parser.add_argument("--out-dir", default=None, help="Output directory for summary files.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--median-align", action="store_true", help="Also report per-frame median scale aligned metrics.")
    parser.add_argument("--save-vis", type=int, default=0, help="Save visualizations for first N batches per experiment.")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--ignore-checkpoint-cfg", action="store_true")
    args, dotlist = parser.parse_known_args()
    dotlist = [item for item in dotlist if item != "--"]
    overrides = OmegaConf.from_dotlist(dotlist) if dotlist else None
    return args, overrides


def main():
    args, overrides = parse_args()
    if args.out_dir is None:
        args.out_dir = f"depth_eval_results/{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)

    if args.checkpoint is not None:
        if args.config is None:
            raise ValueError("--config is required when --checkpoint is provided")
        specs = {
            args.name: {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "model_type": args.model_type,
            }
        }
    else:
        if args.experiments == "all":
            names = list(DEFAULT_EXPERIMENTS.keys())
        else:
            names = [name.strip() for name in args.experiments.split(",") if name.strip()]
        unknown = [name for name in names if name not in DEFAULT_EXPERIMENTS]
        if unknown:
            raise KeyError(f"Unknown experiments: {unknown}. Available: {sorted(DEFAULT_EXPERIMENTS)}")
        specs = {name: DEFAULT_EXPERIMENTS[name] for name in names}

    rows = []
    for name, spec in specs.items():
        rows.extend(evaluate_one(name, spec, args, overrides, out_dir))

    if not rows:
        raise RuntimeError("No evaluation rows were produced. Check checkpoint paths or use --strict-missing.")
    write_results(rows, out_dir)

    print("\nDepth metric summary:")
    for row in rows:
        print(
            f"{row['experiment']:28s} {row['prediction']:22s} "
            f"AbsRel={row.get('abs_rel', float('nan')):.6f} "
            f"RMSE={row.get('rmse', float('nan')):.6f} "
            f"d1={row.get('delta1', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
