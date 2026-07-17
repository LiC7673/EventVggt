"""Visualize the first 10 Bearded-Man ev_2 frames with the HF-residual V2 model."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

import finetune_event as fe
import real_reliability_stage.evaluate_stage2_heldout as protocol
from ablation.eag3r_metrics_eval import move_views_to_device
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default="/data1/lzh/dataset/reflective_raw")
    p.add_argument("--output", default="exp_f/cur_event_clean_hf_residual_v2_gpu4/bearded_ev2_first10")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--depth-scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def normal_rgb(normal, valid):
    image = ((normal.detach().float().cpu() + 1.) * .5).clamp(0, 1)
    return image * valid.detach().float().cpu().unsqueeze(-1)


def identity(view, fallback):
    value = view.get("instance", fallback)
    if isinstance(value, (list, tuple)) and value: value = value[0]
    return str(value)


@torch.inference_mode()
def main():
    args = parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(args.checkpoint, device, args.depth_scale)
    OmegaConf.set_struct(cfg, False); OmegaConf.set_struct(cfg.data, False)
    cfg.data.event_source_mode = "cur_event"
    ns = SimpleNamespace(
        root=args.root, num_views=4, resolution=[518, 392],
        scene_names=["Bearded Man_Ceramic_Glazed_White"], initial_scene_idx=0,
        active_scene_count=1, test_frame_count=max(args.frames, 10),
        ldr_event_id="ev_2", event_resize_method="voxel_linear_time",
        event_resize_bins=5, window_stride=4, batch_size=1, num_workers=0,
        pin_memory=False, max_batches=None,
    )
    dataset, loader = protocol.build_loader(cfg, ns)
    print(f"[visualize] scenes={dataset.get_active_scenes()} source=cur_event ev_2", flush=True)
    saved = 0
    for cpu_views in loader:
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        output = model(views)
        for view_index, (view, result) in enumerate(zip(views, output.ress)):
            if saved >= args.frames: break
            coarse = result["depth_coarse"][0, ..., 0].float()
            final = result["depth"][0, ..., 0].float()
            gt = view["depthmap"][0].float()
            intrinsics = view["camera_intrinsics"][0].float()
            valid = torch.isfinite(gt) & (gt > 1e-6)
            coarse_n = fe.depth_to_normals(coarse[None, None], intrinsics[None, None])[0, 0]
            final_n = fe.depth_to_normals(final[None, None], intrinsics[None, None])[0, 0]
            gt_n = fe.depth_to_normals(gt[None, None], intrinsics[None, None])[0, 0]
            event = view["event_voxel"][0].detach().float().abs().sum(0).cpu()
            depth_values = torch.cat((coarse[valid], final[valid], gt[valid]))
            lo, hi = float(depth_values.min()), float(depth_values.max())
            panels = (
                (event, "cur_event voxel magnitude", "gray", None, None),
                (coarse.cpu() * valid.cpu(), "coarse depth", "viridis", lo, hi),
                (final.cpu() * valid.cpu(), "final depth", "viridis", lo, hi),
                (gt.cpu() * valid.cpu(), "GT depth", "viridis", lo, hi),
                (normal_rgb(coarse_n, valid), "coarse normal", None, None, None),
                (normal_rgb(final_n, valid), "final normal", None, None, None),
                (normal_rgb(gt_n, valid), "GT normal", None, None, None),
                (((final - gt).abs() * valid).cpu(), "|final-GT| depth", "magma", 0,
                 float(((final - gt).abs() * valid).max().clamp_min(1e-6))),
            )
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
                shown = axis.imshow(image.numpy() if torch.is_tensor(image) else image,
                                    cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_title(title); axis.axis("off")
                if cmap is not None: fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
            name = identity(view, f"frame_{saved:03d}")
            fig.suptitle(f"Bearded Man | ev_2 | {name} | cur_event HF residual V2")
            fig.tight_layout(); path = out / f"frame_{saved:03d}.png"
            fig.savefig(path, dpi=140); plt.close(fig)
            print(f"[visualize] {saved + 1}/{args.frames}: {path}", flush=True)
            saved += 1
        if saved >= args.frames: break
    if saved < args.frames:
        raise RuntimeError(f"requested {args.frames} frames but dataset yielded only {saved}")
    print(f"Saved {saved} visualizations to {out.resolve()}", flush=True)


if __name__ == "__main__": main()
