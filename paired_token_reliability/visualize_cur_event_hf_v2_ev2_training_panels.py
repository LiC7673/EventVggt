"""Render 120 ev_2 samples with the diagnostics used during training.

This is intentionally a new entry point.  It does not modify the older ev_5
renderer.  Each output image contains RGB/events, both confidence maps,
alignment diagnostics, depth/normal results, and normal-derivative diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import finetune_event as fe
from ablation.eag3r_metrics_eval import move_views_to_device
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import build_model
from paired_token_reliability.visualize_cur_event_hf_v2_ev5_geo_full import (
    identity,
    normal_rgb,
    rgb_image,
    signed_event_image,
    signed_limit,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="exp_f/cur_event_clean_hf_residual_v2_gpu4/checkpoint-adapter-last.pth")
    p.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    p.add_argument("--output", default="exp_f/cur_event_clean_hf_residual_v2_gpu4/ev2_training_panels_120")
    p.add_argument("--scene", default="Bearded Man_Ceramic_Glazed_White")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--view-index", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--depth-scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dpi", type=int, default=130)
    return p.parse_args()


def make_loader(cfg, args, source_mode, decomposition_supervision):
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
        ldr_event_id="ev_2",
        event_spatial_transform=str(getattr(cfg.data, "event_spatial_transform", "auto")),
        event_resize_method="voxel_linear_time",
        event_resize_bins=5,
        event_source_mode=source_mode,
        decomposition_supervision=decomposition_supervision,
        decomposition_event_root=str(getattr(cfg.data, "decomposition_event_root", "events_additive")),
        decomposition_geo_branch=str(getattr(cfg.data, "decomposition_geo_branch", "geometry_motion")),
        decomposition_full_branch=str(getattr(cfg.data, "decomposition_full_branch", "full")),
        return_normal_gt=True,
        return_debug_event_fields=False,
    )
    ids = list(range(0, len(dataset), max(args.frame_stride, 1)))
    return dataset, DataLoader(
        Subset(dataset, ids), batch_size=1, shuffle=False, num_workers=0,
        pin_memory=False, drop_last=False, collate_fn=event_multiview_collate,
    )


def map2d(value, *, batch=True):
    """Convert an arbitrary diagnostic tensor to a displayable 2-D map."""
    if value is None:
        return None
    x = value.detach().float().cpu()
    if batch and x.ndim >= 3 and x.shape[0] == 1:
        x = x[0]
    while x.ndim > 2:
        # Preserve spatial dimensions and reduce feature/channel dimensions.
        channel_axis = 0 if x.shape[0] <= 64 and x.shape[-1] > 64 else -1
        x = x.abs().mean(channel_axis)
    return x.numpy()


def derivative_magnitude(normal):
    """Magnitude of the same forward-difference normal derivative used by training."""
    n = F.normalize(normal.detach().float(), dim=-1, eps=1.0e-6)
    dx, dy = torch.zeros_like(n), torch.zeros_like(n)
    dx[:, :-1] = n[:, 1:] - n[:, :-1]
    dy[:-1, :] = n[1:, :] - n[:-1, :]
    return torch.stack((dx, dy), dim=-2).square().sum((-1, -2)).sqrt()


def predicted_derivative_magnitude(result):
    value = result.get("event_normal_derivative_full")
    if value is None:
        value = result.get("event_normal_derivative")
    if value is None:
        return None
    x = value[0].detach().float().cpu()
    return x.reshape(*x.shape[:2], -1).square().sum(-1).sqrt().numpy()


def blank_like(height, width):
    return np.zeros((height, width), dtype=np.float32)


@torch.inference_mode()
def main():
    args = parse_args()
    if not 0 <= args.view_index < args.num_views:
        raise ValueError("--view-index must be in [0, num_views)")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    model, cfg = build_model(Path(args.checkpoint), device, args.depth_scale)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.data, False)
    cur_dataset, cur_loader = make_loader(cfg, args, "cur_event", True)
    full_dataset, full_loader = make_loader(cfg, args, "decomposition_full", False)
    if len(cur_loader) != len(full_loader):
        raise RuntimeError(f"cur/full loader mismatch: {len(cur_loader)} vs {len(full_loader)}")
    print(
        f"[ev2 training panels] scene={args.scene} cur={len(cur_dataset)} "
        f"full={len(full_dataset)} requested={args.frames}",
        flush=True,
    )

    saved = 0
    for cur_cpu, full_cpu in zip(cur_loader, full_loader):
        cur_views = move_views_to_device(fe.maybe_denormalize_views(cur_cpu), device)
        full_views = move_views_to_device(fe.maybe_denormalize_views(full_cpu), device)
        view = cur_views[args.view_index]
        full_view = full_views[args.view_index]
        sample_id = identity(view, f"sample_{saved:04d}")
        if sample_id != identity(full_view, ""):
            raise RuntimeError("cur_event and full-event samples are not synchronized")

        output = model(cur_views)
        result = output.ress[args.view_index]
        coarse = result["depth_coarse"][0, ..., 0].float()
        final = result["depth"][0, ..., 0].float()
        gt = view["depthmap"][0].float()
        intr = view["camera_intrinsics"][0].float()
        valid = torch.isfinite(gt) & (gt > 1.0e-6)
        h, w = gt.shape

        coarse_n = fe.depth_to_normals(coarse[None, None], intr[None, None])[0, 0]
        final_n = fe.depth_to_normals(final[None, None], intr[None, None])[0, 0]
        gt_n = fe.depth_to_normals(gt[None, None], intr[None, None])[0, 0]
        pred_dn = predicted_derivative_magnitude(result)
        gt_dn = derivative_magnitude(gt_n).cpu().numpy()
        final_dn = derivative_magnitude(final_n).cpu().numpy()
        if pred_dn is None:
            pred_dn = blank_like(h, w)
        dn_error = np.abs(pred_dn - gt_dn)

        if "geometry_event_voxel" not in view:
            raise RuntimeError("geometry_event_voxel is missing")
        e_geo = signed_event_image(view["geometry_event_voxel"][0])
        e_full = signed_event_image(full_view["event_voxel"][0])
        e_cur = signed_event_image(view["event_voxel"][0])
        ev_lim = signed_limit(e_geo, e_full, e_cur)
        support = map2d(result.get("event_normal_support"))
        if support is None:
            support = (np.abs(e_cur) > 0).astype(np.float32)

        c_fusion = map2d(result.get("event_contribution"))
        c_refine = map2d(result.get("normal_fusion_gate"))
        c_target = map2d(result.get("alignment_reliability_target"))
        feature_error = map2d(result.get("event_feature_alignment_error"))
        hdr_error = map2d(result.get("hdr_token_alignment_error"))
        fallback = blank_like(h, w)
        c_fusion = fallback if c_fusion is None else c_fusion
        c_refine = fallback if c_refine is None else c_refine
        c_target = fallback if c_target is None else c_target
        feature_error = fallback if feature_error is None else feature_error
        hdr_error = fallback if hdr_error is None else hdr_error

        values = torch.cat((coarse[valid], final[valid], gt[valid]))
        dmin, dmax = float(values.min()), float(values.max())
        depth_error = ((final - gt).abs() * valid).cpu()
        update = ((final - coarse) * valid).cpu()
        ulim = max(float(torch.quantile(update.abs().flatten(), .995)), 1.0e-6)

        panels = [
            (rgb_image(view), "LDR RGB (ev_2)", None, None, None),
            (e_geo, r"$E_{\rm geo}$", "seismic", -ev_lim, ev_lim),
            (e_full, r"$E_{\rm full}$", "seismic", -ev_lim, ev_lim),
            (e_cur, r"$E_{\rm cur}$", "seismic", -ev_lim, ev_lim),
            (support, "real event support", "gray", 0, 1),
            (c_fusion, r"$C_{\rm fusion}$", "magma", 0, 1),
            (c_refine, r"$C_{\rm refine}$", "magma", 0, 1),
            (c_target, r"$C$ target", "magma", 0, 1),
            (feature_error, "full-to-geo feature error", "magma", None, None),
            (hdr_error, "LDR+event to HDR token error", "magma", None, None),
            (coarse.cpu() * valid.cpu(), "coarse depth", "viridis", dmin, dmax),
            (final.cpu() * valid.cpu(), "final depth", "viridis", dmin, dmax),
            (gt.cpu() * valid.cpu(), "GT depth", "viridis", dmin, dmax),
            (depth_error, "|final - GT|", "magma", 0, max(float(depth_error.max()), 1e-6)),
            (update, "depth update", "coolwarm", -ulim, ulim),
            (normal_rgb(coarse_n, valid), "coarse normal", None, None, None),
            (normal_rgb(final_n, valid), "final normal", None, None, None),
            (normal_rgb(gt_n, valid), "GT normal", None, None, None),
            (final_dn, "final-depth |normal derivative|", "magma", 0, None),
            (pred_dn, "pred |event normal derivative|", "magma", 0, None),
            (gt_dn, "GT |normal derivative|", "magma", 0, None),
            (dn_error, "event derivative magnitude error", "magma", 0, None),
            (np.abs(final_dn - gt_dn), "final derivative magnitude error", "magma", 0, None),
            (np.abs(e_geo), r"$|E_{\rm geo}|$", "gray", 0, None),
            (np.abs(e_full - e_geo), r"$|E_{\rm full}-E_{\rm geo}|$", "gray", 0, None),
        ]

        serial = saved + 1
        fig, axes = plt.subplots(5, 5, figsize=(24, 22))
        for ax, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
            arr = image.numpy() if torch.is_tensor(image) else np.asarray(image)
            shown = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"[{serial:03d}] {title}", fontsize=10)
            ax.axis("off")
            if cmap is not None:
                fig.colorbar(shown, ax=ax, fraction=.046, pad=.025)
        fig.suptitle(
            f"#{serial:03d} | {args.scene} | ev_2 | {sample_id} | "
            "cur_event clean HF-residual V2",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, .975))
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        path = out_dir / f"{serial:03d}_{safe_id}_ev2_training_panel.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved += 1
        print(f"[visualize] {saved:03d}/{args.frames}: {path}", flush=True)
        if saved >= args.frames:
            break

    if saved < args.frames:
        raise RuntimeError(f"requested {args.frames} samples but only saved {saved}")
    print(f"Saved {saved} ev_2 training-style panels to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
