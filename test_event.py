import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

import croco.utils.misc as misc
import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from eventvggt.models.streamvggt import StreamVGGT as EventStreamVGGT

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

printer = get_logger(__name__, log_level="INFO")


def build_dataset_for_scenes(cfg, scene_names: List[str], split: str):
    return get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=scene_names,
        initial_scene_idx=0,
        active_scene_count=len(scene_names),
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=getattr(cfg.data, "ldr_event_id", "auto"),
    )


def discover_eval_scenes(cfg) -> Tuple[List[str], List[str], List[str]]:
    probe = get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=cfg.data.scene_names if cfg.data.scene_names else None,
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split="train",
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=getattr(cfg.data, "ldr_event_id", "auto"),
    )
    all_scenes = list(probe.scenes)
    active_scenes = probe.get_active_scenes()
    non_active_scenes = [scene for scene in all_scenes if scene not in active_scenes]
    non_active_scenes = non_active_scenes[: len(active_scenes)]
    return all_scenes, active_scenes, non_active_scenes


def build_loader(cfg, scene_names: List[str], split: str):
    dataset = build_dataset_for_scenes(cfg, scene_names, split)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(?:step[_-]?|checkpoint[_-]?)(\d+)", path.stem)
    if match:
        return int(match.group(1))
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "step" in ckpt:
            return int(ckpt["step"])
    except Exception:
        pass
    return -1


def find_checkpoints(cfg) -> List[Path]:
    output_dir = Path(cfg.output_dir)
    patterns = ["checkpoint*.pth", "*.ckpt"]
    paths = []
    for pattern in patterns:
        paths.extend(output_dir.glob(pattern))
    unique_paths = sorted(set(paths), key=lambda p: (checkpoint_step(p), p.name))
    if unique_paths:
        return unique_paths
    pretrained = getattr(cfg, "pretrained", None)
    return [Path(pretrained)] if pretrained else []


def load_model_for_checkpoint(cfg, checkpoint_path: Path, device: torch.device) -> EventStreamVGGT:
    model = EventStreamVGGT(
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        event_hidden_dim=cfg.model.event_hidden_dim,
        head_frames_chunk_size=int(getattr(cfg.model, "head_frames_chunk_size", 2)),
    ).to(device)
    if checkpoint_path and checkpoint_path.exists():
        ckpt = fe.unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        msg = model.load_state_dict(ckpt, strict=False)
        printer.info("Loaded checkpoint %s: %s", checkpoint_path, msg)
    else:
        printer.warning("Checkpoint %s does not exist; evaluating randomly initialized model", checkpoint_path)
    model.eval()
    return model


def move_views_to_device(views: List[Dict], device: torch.device, dtype: torch.dtype) -> List[Dict]:
    for view in views:
        for key, value in list(view.items()):
            if not torch.is_tensor(value):
                continue
            if value.dtype.is_floating_point:
                view[key] = value.to(device=device, dtype=dtype)
            else:
                view[key] = value.to(device=device)
    return views


def depth_error_to_uint8(error: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    error_np = error.detach().float().cpu().numpy()
    mask_np = mask.detach().bool().cpu().numpy()
    if mask_np.any():
        scale = np.percentile(error_np[mask_np], 95)
        scale = max(float(scale), 1e-6)
    else:
        scale = 1.0
    norm = np.clip(error_np / scale, 0.0, 1.0)
    img = np.zeros((*norm.shape, 3), dtype=np.uint8)
    img[..., 0] = (norm * 255).astype(np.uint8)
    img[..., 1] = ((1.0 - np.abs(norm - 0.5) * 2.0) * 180).astype(np.uint8)
    img[..., 2] = ((1.0 - norm) * 255).astype(np.uint8)
    img[~mask_np] = 0
    return img


def safe_name(value) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            value = value.item()
        else:
            value = "_".join(map(str, value.flatten()[:4].tolist()))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def view_label(views: List[Dict], frame_idx: int, sample_idx: int) -> str:
    view = views[frame_idx]
    if "instance" in view:
        value = view["instance"][sample_idx] if isinstance(view["instance"], (list, tuple)) else view["instance"][sample_idx]
        return safe_name(value)
    if "label" in view:
        value = view["label"][sample_idx] if isinstance(view["label"], (list, tuple)) else view["label"][sample_idx]
        return safe_name(value)
    return f"sample_{sample_idx:03d}_view_{frame_idx:03d}"


def save_depth_visuals(
    out_dir: Path,
    views: List[Dict],
    aux: Dict[str, torch.Tensor],
    batch_idx: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size, num_views = aux["depth_pred"].shape[:2]
    for sample_idx in range(batch_size):
        for frame_idx in range(num_views):
            valid_mask = aux["valid_mask"][sample_idx, frame_idx].detach().bool()
            depth_pred = aux["depth_pred"][sample_idx, frame_idx].detach()
            depth_gt = aux["depth_gt"][sample_idx, frame_idx].detach()
            error = (depth_pred - depth_gt).abs()
            label = view_label(views, frame_idx, sample_idx)
            prefix = f"batch_{batch_idx:05d}_{label}_view_{frame_idx:02d}"

            rgb = fe.tensor_rgb_to_uint8(views[frame_idx]["img"][sample_idx], valid_mask)
            gt = fe.depth_to_uint8(depth_gt, valid_mask)
            pred = fe.depth_to_uint8(depth_pred, valid_mask)
            error_map = depth_error_to_uint8(error, valid_mask)

            panels = [
                fe.make_labeled_panel("rgb", rgb),
                fe.make_labeled_panel("gt_depth", gt),
                fe.make_labeled_panel("pred_depth", pred),
                fe.make_labeled_panel("error_map", error_map),
            ]
            canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, 0))
                x += panel.width
            canvas.save(out_dir / f"{prefix}_panel.png")
            Image.fromarray(error_map).save(out_dir / f"{prefix}_error_map.png")


def scalarize(details: Dict[str, float]) -> Dict[str, float]:
    return {key: float(value) for key, value in details.items()}


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    criterion,
    accelerator: Accelerator,
    out_dir: Path,
    cfg,
) -> Dict[str, float]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for batch_idx, views in enumerate(metric_logger.log_every(loader, cfg.print_freq, accelerator, str(out_dir.name))):
        views = fe.maybe_denormalize_views(views)
        views = move_views_to_device(views, device=device, dtype=dtype)
        with accelerator.autocast():
            model_output = model(views)
            loss, loss_details, aux = criterion(model_output, views)

        metric_logger.update(loss=float(loss.detach()))
        metric_logger.update(**scalarize(loss_details))
        save_depth_visuals(out_dir, views, aux, batch_idx)

    metric_logger.synchronize_between_processes(accelerator)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def make_criterion(cfg) -> fe.EventSupervisedLoss:
    return fe.EventSupervisedLoss(
        pose_weight=cfg.loss.pose_weight,
        depth_weight=cfg.loss.depth_weight,
        points_weight=cfg.loss.points_weight,
        normal_weight=float(getattr(cfg.loss, "normal_weight", 0.0)),
        depth_min=float(getattr(cfg.loss, "depth_min", 1e-6)),
        depth_max=(float(cfg.loss.depth_max) if getattr(cfg.loss, "depth_max", None) is not None else None),
        align_depth_scale_enabled=bool(getattr(cfg.loss, "align_depth_scale", True)),
        points_loss_type=str(getattr(cfg.loss, "points_loss_type", "cd")),
    )


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_eval(cfg) -> None:
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    output_root = Path(cfg.output_dir) / "event_test_eval"
    output_root.mkdir(parents=True, exist_ok=True)

    all_scenes, active_scenes, non_active_scenes = discover_eval_scenes(cfg)
    scene_groups = {
        "active": active_scenes,
        "non_active": non_active_scenes,
    }
    write_json(
        output_root / "scene_split.json",
        {
            "all_scenes": all_scenes,
            "active": active_scenes,
            "non_active": non_active_scenes,
            "num_active": len(active_scenes),
            "num_non_active": len(non_active_scenes),
        },
    )

    loaders = {}
    for group_name, scenes in scene_groups.items():
        if not scenes:
            printer.warning("No scenes for group %s; skipping", group_name)
            continue
        for split in ("train", "test"):
            loader = build_loader(cfg, scenes, split)
            loaders[(group_name, split)] = loader
            printer.info("%s/%s: scenes=%s samples=%d", group_name, split, scenes, len(loader.dataset))

    checkpoint_paths = find_checkpoints(cfg)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {cfg.output_dir} and cfg.pretrained is empty")

    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "results": {},
    }

    for checkpoint_path in checkpoint_paths:
        ckpt_name = checkpoint_path.stem
        model = load_model_for_checkpoint(cfg, checkpoint_path, accelerator.device)
        criterion = make_criterion(cfg).to(accelerator.device)
        summary["results"][ckpt_name] = {}

        for (group_name, split), loader in loaders.items():
            eval_dir = output_root / ckpt_name / group_name / split
            stats = evaluate_loader(model, loader, criterion, accelerator, eval_dir, cfg)
            summary["results"][ckpt_name][f"{group_name}/{split}"] = stats
            write_json(eval_dir / "metrics.json", stats)
            printer.info("%s %s/%s stats: %s", ckpt_name, group_name, split, stats)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(output_root / "summary.json", summary)
    printer.info("Saved test evaluation to %s", output_root)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "config"),
    config_name="finetune_event.yaml",
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    run_eval(cfg)


if __name__ == "__main__":
    main()
