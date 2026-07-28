"""Render ev_5 results from cur_event_clean_hf_residual_v2 with E_geo/E_full.

E_geo and E_full are displayed as signed red/blue event maps:
positive polarity is red, negative polarity is blue, and zero is white.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import finetune_event as fe
from ablation.eag3r_metrics_eval import move_views_to_device
from eventvggt.datasets.my_event_dataset import (
    event_multiview_collate,
    get_combined_dataset,
)
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import (
    build_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth",
    )
    parser.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument(
        "--output",
        default="exp_f/cur_event_clean_hf_residual_v2_gpu4/ev5_geo_full_visualization",
    )
    parser.add_argument("--scene", default="Bearded Man_Ceramic_Glazed_White")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--depth-scale", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def build_synced_loader(cfg, args, source_mode, decomposition_supervision):
    dataset = get_combined_dataset(
        root=args.root,
        num_views=args.num_views,
        resolution=(518, 392),
        fps=int(getattr(cfg.data, "fps", 120)),
        seed=int(getattr(cfg, "seed", 0)),
        scene_names=[args.scene],
        initial_scene_idx=0,
        active_scene_count=1,
        split="test",
        test_frame_count=max(args.frames * args.frame_stride + args.num_views, 16),
        ldr_event_id="ev_5",
        event_spatial_transform=str(
            getattr(cfg.data, "event_spatial_transform", "auto")
        ),
        event_resize_method="voxel_linear_time",
        event_resize_bins=5,
        event_source_mode=source_mode,
        decomposition_supervision=decomposition_supervision,
        decomposition_event_root=str(
            getattr(cfg.data, "decomposition_event_root", "events_additive")
        ),
        decomposition_geo_branch=str(
            getattr(cfg.data, "decomposition_geo_branch", "geometry_motion")
        ),
        decomposition_full_branch=str(
            getattr(cfg.data, "decomposition_full_branch", "full")
        ),
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    indices = list(range(0, len(dataset), max(args.frame_stride, 1)))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    return dataset, loader


def signed_event_image(voxel: torch.Tensor) -> np.ndarray:
    """Convert [2B,H,W] split-polarity voxel to a signed event map."""
    voxel = voxel.detach().float().cpu()
    if voxel.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {tuple(voxel.shape)}")
    if voxel.shape[0] % 2 == 0:
        bins = voxel.shape[0] // 2
        signed = voxel[:bins].sum(0) - voxel[bins:].sum(0)
    else:
        signed = voxel.sum(0)
    return signed.numpy()


def signed_limit(*images: np.ndarray) -> float:
    values = np.concatenate([np.abs(x).reshape(-1) for x in images])
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return 1.0
    return max(float(np.quantile(nonzero, 0.995)), 1.0e-6)


def normal_rgb(normal, valid):
    image = ((normal.detach().float().cpu() + 1.0) * 0.5).clamp(0, 1)
    return image * valid.detach().float().cpu().unsqueeze(-1)


def rgb_image(view):
    image = view["img"][0].detach().float().cpu()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    return image.clamp(0, 1)


def identity(view, fallback):
    value = view.get("instance", fallback)
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value)


@torch.inference_mode()
def main():
    args = parse_args()
    if not 0 <= args.view_index < args.num_views:
        raise ValueError("--view-index must be in [0, num_views)")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    checkpoint = Path(args.checkpoint)
    model, cfg = build_model(checkpoint, device, args.depth_scale)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)

    # The model sees its trained cur_event representation and obtains E_geo
    # from the decomposition-supervision field.
    cur_dataset, cur_loader = build_synced_loader(
        cfg, args, source_mode="cur_event", decomposition_supervision=True
    )
    # This synchronized read is visualization-only and strictly loads
    # events_additive/full as E_full.
    full_dataset, full_loader = build_synced_loader(
        cfg, args, source_mode="decomposition_full", decomposition_supervision=False
    )
    if len(cur_loader) != len(full_loader):
        raise RuntimeError(
            f"cur/full loader mismatch: {len(cur_loader)} vs {len(full_loader)}"
        )
    print(
        f"[ev5 visualize] scene={args.scene} cur={len(cur_dataset)} "
        f"full={len(full_dataset)} frames={args.frames}",
        flush=True,
    )

    saved = 0
    for cur_cpu, full_cpu in zip(cur_loader, full_loader):
        cur_views = move_views_to_device(fe.maybe_denormalize_views(cur_cpu), device)
        full_views = move_views_to_device(fe.maybe_denormalize_views(full_cpu), device)
        index = args.view_index
        cur_view, full_view = cur_views[index], full_views[index]
        cur_id = identity(cur_view, f"sample_{saved:04d}")
        full_id = identity(full_view, "")
        if cur_id != full_id:
            raise RuntimeError(f"unsynchronized samples: cur={cur_id}, full={full_id}")

        outputs = model(cur_views)
        result = outputs.ress[index]
        coarse = result["depth_coarse"][0, ..., 0].float()
        final = result["depth"][0, ..., 0].float()
        gt = cur_view["depthmap"][0].float()
        intrinsics = cur_view["camera_intrinsics"][0].float()
        valid = torch.isfinite(gt) & (gt > 1.0e-6)

        coarse_n = fe.depth_to_normals(
            coarse[None, None], intrinsics[None, None]
        )[0, 0]
        final_n = fe.depth_to_normals(
            final[None, None], intrinsics[None, None]
        )[0, 0]
        gt_n = fe.depth_to_normals(gt[None, None], intrinsics[None, None])[0, 0]

        if "geometry_event_voxel" not in cur_view:
            raise RuntimeError("E_geo is missing; decomposition supervision is required")
        e_geo = signed_event_image(cur_view["geometry_event_voxel"][0])
        e_full = signed_event_image(full_view["event_voxel"][0])
        e_cur = signed_event_image(cur_view["event_voxel"][0])
        event_vlim = signed_limit(e_geo, e_full, e_cur)

        depth_values = torch.cat((coarse[valid], final[valid], gt[valid]))
        depth_lo, depth_hi = float(depth_values.min()), float(depth_values.max())
        depth_error = ((final - gt).abs() * valid).detach().cpu()
        depth_update = ((final - coarse) * valid).detach().cpu()
        update_limit = max(
            float(torch.quantile(depth_update.abs().flatten(), 0.995)), 1.0e-6
        )

        panels = (
            (rgb_image(cur_view), "LDR RGB (ev5)", None, None, None),
            (e_geo, r"$E_{\mathrm{geo}}$ (+ red / - blue)", "seismic",
             -event_vlim, event_vlim),
            (e_full, r"$E_{\mathrm{full}}$ (+ red / - blue)", "seismic",
             -event_vlim, event_vlim),
            (e_cur, r"$E_{\mathrm{cur}}$ (+ red / - blue)", "seismic",
             -event_vlim, event_vlim),
            (coarse.cpu() * valid.cpu(), "coarse depth", "viridis",
             depth_lo, depth_hi),
            (final.cpu() * valid.cpu(), "final depth", "viridis",
             depth_lo, depth_hi),
            (gt.cpu() * valid.cpu(), "GT depth", "viridis",
             depth_lo, depth_hi),
            (depth_error, "|final - GT|", "magma", 0.0,
             max(float(depth_error.max()), 1.0e-6)),
            (normal_rgb(coarse_n, valid), "coarse normal", None, None, None),
            (normal_rgb(final_n, valid), "final normal", None, None, None),
            (normal_rgb(gt_n, valid), "GT normal", None, None, None),
            (depth_update, "depth update", "coolwarm",
             -update_limit, update_limit),
        )

        serial = saved + 1
        fig, axes = plt.subplots(3, 4, figsize=(20, 14))
        for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
            array = image.numpy() if torch.is_tensor(image) else image
            shown = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(f"[{serial:03d}] {title}")
            axis.axis("off")
            if cmap is not None:
                fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"#{serial:03d} | {args.scene} | ev_5 | {cur_id} | "
            "cur_event HF-residual V2",
            fontsize=16,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        path = output_dir / f"{serial:03d}_{cur_id}_ev5.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] {serial}/{args.frames}: {path}", flush=True)
        saved += 1
        if saved >= args.frames:
            break

    if saved < args.frames:
        raise RuntimeError(f"requested {args.frames} samples but only saved {saved}")
    print(f"Saved {saved} numbered ev5 visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
