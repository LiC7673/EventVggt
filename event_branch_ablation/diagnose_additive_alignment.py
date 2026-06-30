"""Verify fixed-window additive event alignment and save branch visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from event_filter_two_stage.data import build_full_event_dataset


def event_rgb(voxel: np.ndarray) -> np.ndarray:
    bins = voxel.shape[0] // 2
    pos = np.log1p(voxel[:bins].sum(axis=0))
    neg = np.log1p(voxel[bins : 2 * bins].sum(axis=0))
    scale = max(float(np.percentile(np.concatenate([pos.ravel(), neg.ravel()]), 99.5)), 1e-6)
    image = np.zeros((*pos.shape, 3), dtype=np.float32)
    image[..., 0] = np.clip(pos / scale, 0, 1)
    image[..., 2] = np.clip(neg / scale, 0, 1)
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--output-dir", default="abl_event_exp/additive_alignment_debug")
    parser.add_argument("--ldr-event-id", default="ev_5")
    parser.add_argument("--initial-scene-idx", type=int, default=0)
    parser.add_argument("--active-scene-count", type=int, default=3)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--mask-dilate-kernel", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "batch_size": 1,
            "num_workers": 0,
            "pin_mem": False,
            "data": {
                "root": args.root,
                "num_views": args.num_views,
                "resolution": [518, 392],
                "fps": 120,
                "scene_names": None,
                "initial_scene_idx": args.initial_scene_idx,
                "active_scene_count": args.active_scene_count,
                "test_frame_count": 10,
                "event_resize_method": "voxel_antialias",
                "event_resize_bins": 10,
                "random_train_ldr": False,
                "eval_ldr_event_id": args.ldr_event_id,
                "additive_event_root": "events_additive",
                "additive_mask_dilate_kernel": args.mask_dilate_kernel,
            },
        }
    )
    dataset = build_full_event_dataset(cfg, split="train", attach_targets=True)
    records = []
    for sample_idx in range(min(args.max_samples, len(dataset))):
        views = dataset[sample_idx]
        for view_idx, view in enumerate(views):
            full = np.asarray(view["event_voxel"], dtype=np.float32)
            geo = np.asarray(view["event_geometry_voxel"], dtype=np.float32)
            mat = np.asarray(view["event_material_voxel"], dtype=np.float32)
            noise = np.asarray(view["event_noise_voxel"], dtype=np.float32)
            residual = np.abs(full - (geo + mat + noise))
            denominator = np.abs(full).sum() + 1e-6
            record = {
                "sample": sample_idx,
                "view": view_idx,
                "label": str(view["label"]),
                "full_energy": float(full.sum()),
                "geometry_energy": float(geo.sum()),
                "material_energy": float(mat.sum()),
                "noise_energy": float(noise.sum()),
                "geometry_ratio": float(geo.sum() / denominator),
                "material_ratio": float(mat.sum() / denominator),
                "noise_ratio": float(noise.sum() / denominator),
                "additive_relative_l1": float(residual.sum() / denominator),
                "full_nonzero_ratio": float((full > 0).mean()),
                "geometry_nonzero_ratio": float((geo > 0).mean()),
            }
            records.append(record)

            panels = [event_rgb(value) for value in (full, geo, mat, noise)]
            residual_map = residual.sum(axis=0)
            residual_map = residual_map / max(float(np.percentile(residual_map, 99.5)), 1e-6)
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for axis, panel, title in zip(
                axes[:4], panels, ("full", "geometry", "material", "noise")
            ):
                axis.imshow(panel)
                axis.set_title(title)
                axis.axis("off")
            axes[4].imshow(np.clip(residual_map, 0, 1), cmap="magma")
            axes[4].set_title(f"additive error={record['additive_relative_l1']:.4g}")
            axes[4].axis("off")
            fig.tight_layout()
            fig.savefig(output / f"sample_{sample_idx:03d}_view_{view_idx:02d}.png", dpi=140)
            plt.close(fig)

    summary = {
        "num_records": len(records),
        "mean_additive_relative_l1": float(np.mean([r["additive_relative_l1"] for r in records])),
        "mean_geometry_ratio": float(np.mean([r["geometry_ratio"] for r in records])),
        "mean_material_ratio": float(np.mean([r["material_ratio"] for r in records])),
        "mean_noise_ratio": float(np.mean([r["noise_ratio"] for r in records])),
        "records": records,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

