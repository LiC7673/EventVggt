"""Render Stage-1 reliability labels without touching held-out test scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventvggt.datasets.my_event_dataset import (  # noqa: E402
    event_multiview_collate,
    get_combined_dataset,
)
from paper_scale_training.scene_split_loader import load_scene_split  # noqa: E402
from real_reliability_stage.render_reliability_labels import (  # noqa: E402
    _make_target,
    _safe_name,
    _save_preview,
)


def _format_ldr(value: str) -> str:
    value = str(value)
    return value if value.startswith("ev_") else f"ev_{value}"


def _scalar_text(value, fallback: str) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    return str(value) if value is not None else fallback


def _build_loader(args, scene_names, ldr_id):
    requested_scenes = list(scene_names)
    dataset = get_combined_dataset(
        root=args.root,
        num_views=1,
        resolution=tuple(args.resolution),
        fps=args.fps,
        seed=args.seed,
        scene_names=list(scene_names),
        initial_scene_idx=0,
        active_scene_count=len(scene_names),
        split="all",
        test_frame_count=0,
        ldr_event_id=ldr_id,
        event_resize_method=args.event_resize_method,
        event_resize_bins=args.event_bins,
        return_normal_gt=True,
    )
    missing = sorted(set(requested_scenes) - set(dataset.scenes))
    if missing:
        raise ValueError(
            f"Stage-1 scenes unavailable at LDR={ldr_id}: {missing}. "
            "Use LDR levels shared by all assigned scenes."
        )
    dataset.set_active_scenes(requested_scenes)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    return dataset, loader


def _render_partition(args, partition, scenes_by_ldr, manifest_name):
    output_root = Path(args.output_dir)
    items = []
    previews_left = max(int(args.preview_count), 0)
    event_pixels = 0
    positive_pixels = 0
    valid_pixels = 0

    all_scenes = []
    for raw_ldr_id, scene_names in scenes_by_ldr.items():
        ldr_id = _format_ldr(raw_ldr_id)
        all_scenes.extend(scene_names)
        dataset, loader = _build_loader(args, scene_names, ldr_id)
        sample_dir = output_root / partition / ldr_id
        preview_dir = output_root / "preview" / partition / ldr_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx, views in enumerate(loader):
            view = views[0]
            sample = _make_target(view, args)
            label = _scalar_text(view.get("label"), f"{partition}_{batch_idx:06d}")
            instance = _scalar_text(view.get("instance"), label)
            filename = f"{batch_idx:06d}_{_safe_name(label)}.npz"
            path = sample_dir / filename
            np.savez_compressed(path, **sample)
            items.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "instance": instance,
                    "ldr_event_id": ldr_id,
                    "source_partition": partition,
                }
            )
            event = np.squeeze(sample["event_support"]) >= args.event_support_min
            target = np.squeeze(sample["target_reliability"])
            valid = np.squeeze(sample["mask"]) > 0
            event_valid = event & valid
            event_pixels += int(event_valid.sum())
            positive_pixels += int((target[event_valid] >= 0.5).sum())
            valid_pixels += int(valid.sum())
            if previews_left > 0:
                _save_preview(preview_dir / f"{Path(filename).stem}.png", sample)
                previews_left -= 1

        print(
            f"[stage1 labels] partition={partition} LDR={ldr_id} "
            f"scenes={len(dataset.get_active_scenes())} frames={len(dataset)}"
        )

    manifest = {
        "split": manifest_name,
        "source_partition": partition,
        "root": args.root,
        "scenes": sorted(set(all_scenes)),
        "ldr_event_ids": [_format_ldr(value) for value in scenes_by_ldr],
        "num_samples": len(items),
        "items": items,
    }
    manifest_path = output_root / f"manifest_{manifest_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    stats = {
        "partition": partition,
        "num_samples": len(items),
        "event_pixel_ratio": event_pixels / max(valid_pixels, 1),
        "positive_ratio_on_event": positive_pixels / max(event_pixels, 1),
        "event_pixels": event_pixels,
        "valid_pixels": valid_pixels,
    }
    (output_root / f"label_stats_{manifest_name}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"[stage1 labels] wrote {manifest_path} ({len(items)} samples)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ldr-ids", nargs="+", default=["ev_2", "ev_5", "ev_10"])
    parser.add_argument("--val-ldr-id", default="ev_5")
    parser.add_argument(
        "--repeat-all-train-ldrs",
        action="store_true",
        help="Render every train frame at every LDR instead of balanced scene-level assignment.",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--event-resize-method", default="voxel_antialias")
    parser.add_argument("--event-bins", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--event-bin-threshold", type=float, default=1.0e-5)
    parser.add_argument("--geometry-percentile", type=float, default=99.0)
    parser.add_argument("--geometry-interior-erode", type=int, default=3)
    parser.add_argument("--geometry-silhouette-weight", type=float, default=0.45)
    parser.add_argument("--geometry-dilate-kernel", type=int, default=5)
    parser.add_argument("--geometry-dilate-gain", type=float, default=0.25)
    parser.add_argument("--normal-detail-weight", type=float, default=0.7)
    parser.add_argument("--depth-detail-weight", type=float, default=0.3)
    parser.add_argument("--image-support-floor", type=float, default=0.70)
    parser.add_argument("--saturation-reject", type=float, default=0.0)
    parser.add_argument("--persistence-floor", type=float, default=0.50)
    parser.add_argument("--persistence-power", type=float, default=1.5)
    parser.add_argument("--polarity-floor", type=float, default=0.70)
    parser.add_argument("--event-support-min", type=float, default=0.01)
    parser.add_argument("--target-mode", choices=("geometry", "cue_modulated"), default="geometry")
    parser.add_argument("--cue-fusion", choices=("geometric", "product"), default="geometric")
    parser.add_argument("--empty-weight", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_scene_split(args.scene_manifest)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "render_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_scenes = list(manifest["splits"]["train"])
    if args.repeat_all_train_ldrs:
        train_groups = {ldr_id: train_scenes for ldr_id in args.train_ldr_ids}
    else:
        train_groups = {
            ldr_id: train_scenes[index :: len(args.train_ldr_ids)]
            for index, ldr_id in enumerate(args.train_ldr_ids)
        }
    _render_partition(args, "train", train_groups, "train")
    # RenderedReliabilityDataset calls this split "test" internally, but the
    # source scenes are strictly the 12 validation scenes from our manifest.
    _render_partition(
        args,
        "val",
        {args.val_ldr_id: list(manifest["splits"]["val"])},
        "test",
    )


if __name__ == "__main__":
    main()
