"""Evaluate a strict direct-full StreamVGGT checkpoint on one exposure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import finetune_event as fe  # noqa: E402
from ablation import eag3r_metrics_eval as metric  # noqa: E402
from eventvggt.datasets.my_event_dataset import get_combined_dataset  # noqa: E402

SCENES = (
    "Centaur_Anodized_Red",
    "Child_with_goose_Industrial_Plastic_Grey",
    "Colchester Sphinx_Old_Copper",
    "Cupid as Shepherd_100MB_Old_Copper",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    p.add_argument("--exposure", default="ev_5")
    p.add_argument("--scene-names", nargs="+", default=list(SCENES))
    p.add_argument("--test-frame-count", type=int, default=120)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--event-resize-bins", type=int, default=5)
    p.add_argument("--depth-scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    p.add_argument("--max-batches", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    raw = metric.torch_load(checkpoint)
    cfg = metric.cfg_from_checkpoint(raw, str(ROOT / "config/finetune_event.yaml"))
    OmegaConf.set_struct(cfg, False)
    cfg.model.variant = "base"
    device = torch.device(args.device)
    model = fe.build_event_model(cfg)
    result = model.load_state_dict(
        metric.strip_module_prefix(fe.unwrap_state_dict(raw)), strict=True
    )
    print(f"[load direct-full] {result}", flush=True)
    model = model.to(device).eval()

    rows = []
    for scene in args.scene_names:
        dataset = get_combined_dataset(
            root=args.root,
            num_views=args.num_views,
            resolution=tuple(cfg.data.resolution),
            fps=int(cfg.data.fps),
            seed=int(cfg.seed),
            scene_names=[scene],
            initial_scene_idx=0,
            active_scene_count=1,
            split="test",
            test_frame_count=args.test_frame_count,
            ldr_event_id=args.exposure,
            event_y_flip=getattr(cfg.data, "event_y_flip", "auto"),
            event_spatial_transform=getattr(cfg.data, "event_spatial_transform", "auto"),
            event_resize_method="voxel_linear_time",
            event_resize_bins=args.event_resize_bins,
            event_source_mode="decomposition_full",
            decomposition_supervision=False,
            decomposition_event_root="events_additive",
            decomposition_geo_branch="geometry_motion",
            decomposition_full_branch="full",
            return_normal_gt=True,
        )
        loader_args = SimpleNamespace(
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=False,
            max_batches=args.max_batches,
            amp=args.amp,
            event_resize_bins=args.event_resize_bins,
            event_support_mode="temporal_polarity",
            event_high_fraction=0.2,
            event_low_fraction=0.2,
            pose_scale_align=False,
            print_freq=20,
        )
        from torch.utils.data import DataLoader
        from eventvggt.datasets.my_event_dataset import event_multiview_collate

        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
            collate_fn=event_multiview_collate,
        )
        depth = metric.DepthMetrics()
        pose = {key: metric.MeanAccumulator() for key in ("ate", "rpe_trans", "rpe_rot_deg")}
        batches = 0
        for views in loader:
            if args.max_batches is not None and batches >= args.max_batches:
                break
            views = metric.move_views_to_device(fe.maybe_denormalize_views(views), device)
            with torch.inference_mode():
                enabled = args.amp != "none"
                dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                    output = model(views)
            pred = metric.stack_output(output, "depth").float() * args.depth_scale
            gt = fe.stack_view_field(views, "depthmap").float()
            valid = fe.build_valid_mask(views, gt)
            depth.update(pred, gt, valid, median_align=False)
            pose_enc = metric.stack_output(output, "camera_pose")
            if pose_enc is not None:
                pred_c2w, _ = fe.pose_encoding_to_c2w(
                    pose_enc.float(), image_size_hw=pred.shape[-2:]
                )
                gt_c2w = fe.stack_view_field(views, "camera_pose").float()
                values = metric.pose_errors(pred_c2w, gt_c2w, scale_align=False)
                for key, value in values.items():
                    pose[key].update(value)
            batches += 1
        row = {
            "scene": scene,
            "exposure": args.exposure,
            "condition": "rgb_plus_direct_full_event",
            "evaluated_batches": batches,
            "depth_scale": args.depth_scale,
            **depth.compute(),
            **{key: value.compute() for key, value in pose.items()},
        }
        rows.append(row)
        print(
            f"[{scene} {args.exposure}] AbsRel={row['abs_rel']:.5f} "
            f"RMSElog={row['rmse_log']:.5f} d1={row['delta1']:.4f}",
            flush=True,
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metric.write_outputs(
        [
            {
                "name": f"direct_full_{args.exposure}_{row['scene']}",
                "family": "event",
                "variant": "base",
                "checkpoint": str(checkpoint),
                "split": "test",
                "active_scenes": row["scene"],
                "num_samples": row["evaluated_batches"],
                **row,
            }
            for row in rows
        ],
        output,
    )


if __name__ == "__main__":
    main()
