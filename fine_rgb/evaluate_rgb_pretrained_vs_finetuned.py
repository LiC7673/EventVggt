"""Compare untouched RGB weights with an RGB-finetuned checkpoint.

The evaluator is deliberately event-free.  It reports scale-sensitive depth
metrics, median-aligned depth metrics, and depth-derived surface-normal metrics
for every requested scene and for the pixel-weighted union of all scenes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_no_event as rgb
from fine_rgb.launcher import normalize_ldr_id
from fine_rgb.rgb_ldr_dataset import get_rgb_ldr_dataset


DEFAULT_SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_dict_from(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def config_for_checkpoint(base_config: Path, checkpoint) -> OmegaConf:
    cfg = OmegaConf.load(base_config)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("cfg"), dict):
        cfg = OmegaConf.merge(cfg, OmegaConf.create(checkpoint["cfg"]))
    OmegaConf.resolve(cfg)
    return cfg


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path)
    cfg = config_for_checkpoint(config_path, checkpoint)
    model = rgb.build_rgb_model(cfg)
    message = model.load_state_dict(state_dict_from(checkpoint), strict=False)
    essential_missing = [
        key for key in message.missing_keys
        if key.startswith(("aggregator.", "depth_head."))
    ]
    if essential_missing:
        raise RuntimeError(
            f"{checkpoint_path} is not a compatible RGB checkpoint; "
            f"missing essential keys: {essential_missing[:12]}"
        )
    model.to(device).eval()
    print(
        f"[load] {checkpoint_path} variant={getattr(cfg.model, 'variant', 'base')} "
        f"missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}"
    )
    return model, cfg


def detect_common_ldr_event_ids(args) -> List[str]:
    dataset = get_rgb_ldr_dataset(
        root=args.data_root,
        num_views=args.num_views,
        resolution=(args.width, args.height),
        fps=args.fps,
        seed=args.seed,
        scene_names=list(args.scenes),
        initial_scene_idx=0,
        active_scene_count=len(args.scenes),
        split="test",
        test_frame_count=args.test_frame_count,
        ldr_event_id="random",
        return_normal_gt=False,
    )
    active = list(dataset.get_active_scenes())
    if set(active) != set(args.scenes):
        raise RuntimeError(
            f"requested scenes={args.scenes!r}, but detector selected {active!r}"
        )
    values = list(dataset.get_active_ldr_events(common=True))
    if not values:
        raise RuntimeError(
            "The selected test scenes do not share any common ev_* RGB level."
        )
    return values


def resolve_ldr_event_ids(args) -> List[str]:
    # Keep the old singular option as an explicit compatibility override.
    if args.ldr_event_id:
        return [normalize_ldr_id(args.ldr_event_id)]
    raw = str(args.ldr_event_ids).strip()
    if raw.lower() in {"auto", "common", "all"}:
        return detect_common_ldr_event_ids(args)
    values = [normalize_ldr_id(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--ldr-event-ids must be 'auto' or a comma-separated ev_* list")
    # Stable de-duplication retains the user-specified evaluation order.
    return list(dict.fromkeys(values))


def make_loader(args, scene: str, ldr_event_id: str) -> DataLoader:
    dataset = get_rgb_ldr_dataset(
        root=args.data_root,
        num_views=args.num_views,
        resolution=(args.width, args.height),
        fps=args.fps,
        seed=args.seed,
        scene_names=[scene],
        initial_scene_idx=0,
        active_scene_count=1,
        split="test",
        test_frame_count=args.test_frame_count,
        ldr_event_id=ldr_event_id,
        return_normal_gt=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"test scene has no samples: {scene}")
    active = list(dataset.get_active_scenes())
    if active != [scene]:
        raise RuntimeError(f"requested scene {scene!r}, but dataset selected {active!r}")
    print(f"[data] scene={scene} samples={len(dataset)} ldr={ldr_event_id}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=rgb.rgb_multiview_collate,
    )


def move_views(views, device):
    views = rgb.maybe_denormalize_views(views)
    views = rgb.drop_event_fields(views)
    for view in views:
        for key, value in list(view.items()):
            if torch.is_tensor(value):
                view[key] = value.to(device=device, non_blocking=True)
    return views


class DepthMetrics:
    def __init__(self):
        self.count = 0
        self.abs_rel = self.sq_rel = self.mae = 0.0
        self.squared = self.log_squared = self.log_sum = 0.0
        self.delta1 = self.delta2 = self.delta3 = 0.0
        self.scale_sum = 0.0
        self.scale_count = 0

    def update(self, pred, target, valid, median_align=False):
        pred = pred.float()
        target = target.float()
        valid = valid.bool() & torch.isfinite(pred) & torch.isfinite(target)
        valid = valid & (pred > 1e-6) & (target > 1e-6)
        if median_align:
            pred = pred.clone()
            for p, t, mask in zip(
                pred.flatten(0, 1), target.flatten(0, 1), valid.flatten(0, 1)
            ):
                if int(mask.sum()) < 16:
                    scale = p.new_tensor(1.0)
                else:
                    scale = (t[mask].median() / p[mask].median().clamp_min(1e-6)).clamp(1e-3, 1e3)
                    p.mul_(scale)
                self.scale_sum += float(scale)
                self.scale_count += 1
            valid = valid & torch.isfinite(pred) & (pred > 1e-6)
        p = pred[valid].clamp_min(1e-6)
        t = target[valid].clamp_min(1e-6)
        if p.numel() == 0:
            return
        difference = p - t
        log_difference = p.log() - t.log()
        ratio = torch.maximum(p / t, t / p)
        self.count += p.numel()
        self.abs_rel += float((difference.abs() / t).sum())
        self.sq_rel += float((difference.square() / t).sum())
        self.mae += float(difference.abs().sum())
        self.squared += float(difference.square().sum())
        self.log_squared += float(log_difference.square().sum())
        self.log_sum += float(log_difference.sum())
        self.delta1 += float((ratio < 1.25).sum())
        self.delta2 += float((ratio < 1.25 ** 2).sum())
        self.delta3 += float((ratio < 1.25 ** 3).sum())

    def result(self):
        if self.count == 0:
            return {
                "pixels": 0,
                "abs_rel": float("nan"),
                "sq_rel": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "rmse_log": float("nan"),
                "silog": float("nan"),
                "delta1": float("nan"),
                "delta2": float("nan"),
                "delta3": float("nan"),
            }
        n = max(float(self.count), 1.0)
        mean_log = self.log_sum / n
        silog_variance = max(self.log_squared / n - mean_log ** 2, 0.0)
        result = {
            "pixels": self.count,
            "abs_rel": self.abs_rel / n,
            "sq_rel": self.sq_rel / n,
            "mae": self.mae / n,
            "rmse": math.sqrt(self.squared / n),
            "rmse_log": math.sqrt(self.log_squared / n),
            "silog": 100.0 * math.sqrt(silog_variance),
            "delta1": self.delta1 / n,
            "delta2": self.delta2 / n,
            "delta3": self.delta3 / n,
        }
        if self.scale_count:
            result["median_scale"] = self.scale_sum / self.scale_count
        return result


class NormalMetrics:
    def __init__(self):
        self.count = 0
        self.degree_sum = self.degree_squared = 0.0
        self.lt_11 = self.lt_22 = self.lt_30 = 0.0
        self.histogram = torch.zeros(1801, dtype=torch.int64)

    def update(self, pred_depth, target_depth, intrinsics, valid):
        pred_normal = rgb.depth_to_normals(pred_depth.float(), intrinsics.float())
        target_normal = rgb.depth_to_normals(target_depth.float(), intrinsics.float())
        mask = rgb.normal_stencil_valid_mask(valid, pred_depth)
        mask &= rgb.normal_stencil_valid_mask(valid, target_depth)
        mask &= pred_normal.norm(dim=-1) > .5
        mask &= target_normal.norm(dim=-1) > .5
        cosine = (F.normalize(pred_normal, dim=-1, eps=1e-6) *
                  F.normalize(target_normal, dim=-1, eps=1e-6)).sum(-1).clamp(-1, 1)
        degrees = torch.rad2deg(torch.acos(cosine))[mask]
        if degrees.numel() == 0:
            return
        self.count += degrees.numel()
        self.degree_sum += float(degrees.sum())
        self.degree_squared += float(degrees.square().sum())
        self.lt_11 += float((degrees < 11.25).sum())
        self.lt_22 += float((degrees < 22.5).sum())
        self.lt_30 += float((degrees < 30.0).sum())
        bins = (degrees.detach().cpu() * 10).round().long().clamp(0, 1800)
        self.histogram += torch.bincount(bins, minlength=1801)

    def result(self):
        if self.count == 0:
            return {
                "normal_pixels": 0,
                "normal_mean_deg": float("nan"),
                "normal_median_deg": float("nan"),
                "normal_rmse_deg": float("nan"),
                "normal_11_25": float("nan"),
                "normal_22_5": float("nan"),
                "normal_30": float("nan"),
            }
        n = max(float(self.count), 1.0)
        cumulative = self.histogram.cumsum(0)
        middle = max((self.count + 1) // 2, 1)
        median_bin = int(torch.searchsorted(cumulative, torch.tensor(middle)))
        return {
            "normal_pixels": self.count,
            "normal_mean_deg": self.degree_sum / n,
            "normal_median_deg": median_bin / 10.0,
            "normal_rmse_deg": math.sqrt(self.degree_squared / n),
            "normal_11_25": self.lt_11 / n,
            "normal_22_5": self.lt_22 / n,
            "normal_30": self.lt_30 / n,
        }


class MetricBundle:
    def __init__(self):
        self.raw = DepthMetrics()
        self.aligned = DepthMetrics()
        self.normal = NormalMetrics()

    def update(self, pred, target, intrinsics, valid):
        self.raw.update(pred, target, valid, median_align=False)
        self.aligned.update(pred, target, valid, median_align=True)
        self.normal.update(pred, target, intrinsics, valid)

    def result(self):
        result = dict(self.raw.result())
        result.update({f"aligned_{key}": value for key, value in self.aligned.result().items()})
        result.update(self.normal.result())
        return result


def evaluate(model, loaders: Dict[str, DataLoader], device, amp, max_batches):
    scene_metrics = {scene: MetricBundle() for scene in loaders}
    overall = MetricBundle()
    amp_enabled = amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.inference_mode():
        for scene, loader in loaders.items():
            for batch_index, views in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                views = move_views(views, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    output = model(views)
                pred = torch.stack([item["depth"][..., 0] for item in output.ress], dim=1).float()
                target = rgb.stack_view_field(views, "depthmap").float()
                intrinsics = rgb.stack_view_field(views, "camera_intrinsics").float()
                valid = rgb.build_valid_mask(views, target)
                scene_metrics[scene].update(pred, target, intrinsics, valid)
                overall.update(pred, target, intrinsics, valid)
            print(f"[done] {scene}")
    return scene_metrics, overall


def write_results(rows: List[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config/finetune_no_event.yaml")
    parser.add_argument("--pretrained", default="ckpt/model.pt")
    parser.add_argument(
        "--finetuned", default=None,
        help="Optional single checkpoint override. By default, each ev_* level uses "
             "the matching path from --finetuned-template.",
    )
    parser.add_argument(
        "--finetuned-template",
        default="checkpoints/fine_rgb_{ldr_event_id}/checkpoint-last.pth",
        help="Per-level checkpoint template; supports {ldr_event_id} and {ldr}.",
    )
    parser.add_argument(
        "--skip-missing-finetuned", action="store_true",
        help="Still report pretrained rows when a matching fine-tuned checkpoint is absent.",
    )
    parser.add_argument("--output-dir", default="exp/rgb_all_ev_pretrained_vs_finetuned")
    parser.add_argument("--data-root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument(
        "--ldr-event-ids", default="auto",
        help="'auto' evaluates every ev_* level common to all selected scenes; "
             "otherwise provide a comma-separated list.",
    )
    parser.add_argument(
        "--ldr-event-id", default=None,
        help="Deprecated single-level override retained for compatibility.",
    )
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--test-frame-count", type=int, default=10)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config_path = Path(args.base_config)
    ldr_event_ids = resolve_ldr_event_ids(args)
    print(f"[LDR levels] {ldr_event_ids}")
    loaders_by_ldr = {
        ldr_event_id: {
            scene: make_loader(args, scene, ldr_event_id) for scene in args.scenes
        }
        for ldr_event_id in ldr_event_ids
    }
    if args.finetuned:
        finetuned_paths = {
            ldr_event_id: Path(args.finetuned) for ldr_event_id in ldr_event_ids
        }
    else:
        finetuned_paths = {
            ldr_event_id: Path(args.finetuned_template.format(
                ldr_event_id=ldr_event_id, ldr=ldr_event_id
            ))
            for ldr_event_id in ldr_event_ids
        }
    missing_finetuned = {
        level: path for level, path in finetuned_paths.items() if not path.is_file()
    }
    if missing_finetuned and not args.skip_missing_finetuned:
        formatted = "\n".join(
            f"  {level}: {path}" for level, path in missing_finetuned.items()
        )
        raise FileNotFoundError(
            "Missing matching RGB-finetuned checkpoints:\n" + formatted +
            "\nRun the per-EV finetuning jobs, set --finetuned-template, or use "
            "--skip-missing-finetuned."
        )

    def append_level_rows(name, checkpoint_path, ldr_event_id, per_scene, overall):
        checkpoint_text = str(checkpoint_path)
        for scene, metrics in per_scene.items():
            rows.append({
                "experiment": name,
                "ldr_event_id": ldr_event_id,
                "scene": scene,
                "checkpoint": checkpoint_text,
                **metrics.result(),
            })
        rows.append({
            "experiment": name,
            "ldr_event_id": ldr_event_id,
            "scene": "ALL_PIXEL_WEIGHTED",
            "checkpoint": checkpoint_text,
            **overall.result(),
        })

    rows = []
    pretrained_path = Path(args.pretrained)
    pretrained_model, _ = build_model(config_path, pretrained_path, device)
    for ldr_event_id, loaders in loaders_by_ldr.items():
        print(f"[eval level] experiment=rgb_pretrained_no_finetune ldr={ldr_event_id}")
        per_scene, overall = evaluate(
            pretrained_model, loaders, device, args.amp, args.max_batches
        )
        append_level_rows(
            "rgb_pretrained_no_finetune", pretrained_path,
            ldr_event_id, per_scene, overall,
        )
    del pretrained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for ldr_event_id, loaders in loaders_by_ldr.items():
        checkpoint_path = finetuned_paths[ldr_event_id]
        if not checkpoint_path.is_file():
            print(f"[skip] rgb_finetuned ldr={ldr_event_id}: {checkpoint_path}")
            continue
        print(
            f"[eval level] experiment=rgb_finetuned ldr={ldr_event_id} "
            f"checkpoint={checkpoint_path}"
        )
        model, _ = build_model(config_path, checkpoint_path, device)
        per_scene, overall = evaluate(
            model, loaders, device, args.amp, args.max_batches
        )
        append_level_rows(
            "rgb_finetuned", checkpoint_path, ldr_event_id, per_scene, overall
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output_dir = Path(args.output_dir)
    write_results(rows, output_dir)
    print("\nexperiment                  EV       scene                                      AbsRel  A-AbsRel Nmean  N<11.25")
    for row in rows:
        print(
            f"{row['experiment']:<27} {row['ldr_event_id']:<8} {row['scene']:<42} "
            f"{row['abs_rel']:.5f}  {row['aligned_abs_rel']:.5f}  "
            f"{row['normal_mean_deg']:.3f}  {row['normal_11_25']:.4f}"
        )
    print(f"Saved {output_dir / 'summary.csv'} and {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
