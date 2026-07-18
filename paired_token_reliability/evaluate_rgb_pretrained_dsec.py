"""Evaluate the original, non-finetuned RGB VGGT on the DSEC test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.dsec_event_dataset import get_dsec_dataset, event_multiview_collate
from paired_token_reliability.common import move_views_to_device, torch_load
from vggt.models.vggt import VGGT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="ckpt/model.pt")
    p.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    p.add_argument("--output", default="exp_f/dsec_rgb_pretrained_no_finetune")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--max-test-batches", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth-scale", type=float, default=1.0,
                   help="fixed scale applied to every prediction; never estimated from test GT")
    p.add_argument("--visualize-every", type=int, default=10)
    p.add_argument("--max-visualizations", type=int, default=40,
                   help="maximum saved view figures; 0 means unlimited")
    return p.parse_args()


def load_model(checkpoint, device):
    model = VGGT()
    raw = torch_load(checkpoint)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    # Official checkpoints may wrap the state dict once.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        print(f"[RGB checkpoint] load result: {loaded}", flush=True)
    return model.to(device).eval().requires_grad_(False)


def make_loader(args):
    dataset = get_dsec_dataset(
        args.root, split="test", num_views=args.num_views, resolution=(518, 392),
        seed=0, event_window_ms=50.0, event_resize_bins=5, clip_stride=4,
        allow_unaligned_rgb=False, depth_scale=1.0, max_depth=80.0,
    )
    print(f"[DSEC RGB] test: {dataset.get_stats()}", flush=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True,
                      collate_fn=event_multiview_collate, drop_last=False)


def evaluate_batch(output, views, scale):
    # This repository's VGGT returns per-view predictions through
    # VGGTOutput.ress rather than exposing ``depth`` at the top level.
    if getattr(output, "ress", None) is not None:
        pred = torch.stack([res["depth"][..., 0] for res in output.ress], dim=1).float()
    elif isinstance(output, dict) and "depth" in output:
        pred = output["depth"][..., 0].float()
    else:
        raise RuntimeError(f"VGGT output contains no depth prediction: {type(output)!r}")
    pred = pred * scale
    gt = fe.stack_view_field(views, "depthmap").float()
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > .1) & (gt < 80.) & (pred > 1e-6)
    pixels = valid.sum().clamp_min(1)
    difference = (pred - gt).abs()
    signed_log_error = torch.log(pred.clamp_min(1e-6)) - torch.log(gt.clamp_min(1e-6))
    mae = (difference * valid).sum() / pixels
    abs_rel = ((difference / gt.clamp_min(1e-6)) * valid).sum() / pixels
    rmse = torch.sqrt(((pred - gt).square() * valid).sum() / pixels)
    rmse_log = torch.sqrt((signed_log_error.square() * valid).sum() / pixels)
    ratio = torch.maximum(pred / gt.clamp_min(1e-6), gt / pred.clamp_min(1e-6))
    delta1 = ((ratio < 1.25) & valid).sum().float() / pixels
    delta2 = ((ratio < 1.25 ** 2) & valid).sum().float() / pixels
    delta3 = ((ratio < 1.25 ** 3) & valid).sum().float() / pixels
    pred_normal = F.normalize(fe.depth_to_normals(pred, intrinsics), dim=-1, eps=1e-6)
    gt_normal = F.normalize(fe.depth_to_normals(gt, intrinsics), dim=-1, eps=1e-6)
    cosine = (pred_normal * gt_normal).sum(-1).clamp(-1, 1)
    normal_valid = fe.normal_stencil_valid_mask(valid, gt, eps=1e-6)
    normal_pixels = normal_valid.sum().clamp_min(1)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6)))
    metrics = dict(
        MAE=float(mae), AbsRel=float(abs_rel), RMSE=float(rmse), RMSElog=float(rmse_log),
        delta1=float(delta1), delta2=float(delta2), delta3=float(delta3),
        Nmean=float((angle * normal_valid).sum() / normal_pixels),
        Nmedian=float(angle[normal_valid].median()) if normal_valid.any() else 0.0,
        N11_25=float(((angle < 11.25) & normal_valid).sum() / normal_pixels),
        N22_5=float(((angle < 22.5) & normal_valid).sum() / normal_pixels),
        N30=float(((angle < 30.0) & normal_valid).sum() / normal_pixels),
        pixels=float(pixels), normal_pixels=float(normal_pixels),
    )
    return metrics, (pred, gt, valid, pred_normal, gt_normal, normal_valid)


def normal_image(normal, mask):
    image = ((normal.detach().cpu().float() + 1.) * .5).clamp(0, 1)
    return image * mask.detach().cpu().float().unsqueeze(-1)


def save_visuals(root, batch_index, views, tensors):
    pred, gt, valid, pred_normal, gt_normal, normal_valid = tensors
    root.mkdir(parents=True, exist_ok=True)
    for view_index, view in enumerate(views):
        mask = valid[0, view_index]
        values = torch.cat((pred[0, view_index][mask], gt[0, view_index][mask]))
        if values.numel() == 0:
            continue
        vmin, vmax = float(values.min()), float(values.max())
        rgb = view["img"][0].detach().cpu().float().permute(1, 2, 0).clamp(0, 1)
        error = ((pred[0, view_index] - gt[0, view_index]).abs() * mask).detach().cpu()
        emax = float(torch.quantile(error.flatten(), .995).clamp_min(1e-6))
        panels = (
            (rgb, "RGB input", None, None, None),
            (pred[0, view_index].detach().cpu() * mask.cpu(), "RGB predicted depth", "viridis", vmin, vmax),
            (gt[0, view_index].detach().cpu() * mask.cpu(), "GT depth", "viridis", vmin, vmax),
            (error, "|RGB depth - GT|", "magma", 0, emax),
            (normal_image(pred_normal[0, view_index], normal_valid[0, view_index]), "RGB depth-derived normal", None, None, None),
            (normal_image(gt_normal[0, view_index], normal_valid[0, view_index]), "GT depth-derived normal", None, None, None),
        )
        fig, axes = plt.subplots(2, 3, figsize=(16, 10)); axes = axes.reshape(-1)
        for axis, (image, title, cmap, lo, hi) in zip(axes, panels):
            shown = axis.imshow(np.asarray(image), cmap=cmap, vmin=lo, vmax=hi)
            axis.set_title(title); axis.axis("off")
            if cmap is not None:
                fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
        raw = view.get("instance", [f"batch_{batch_index:06d}"])[0]
        safe = str(raw).replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.suptitle(f"DSEC RGB pretrained, no finetuning: {safe}, view={view_index}")
        fig.tight_layout()
        fig.savefig(root / f"batch_{batch_index:06d}_view_{view_index:02d}_{safe}.png", dpi=130)
        plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)
    loader = make_loader(args)
    depth_keys = ("MAE", "AbsRel", "RMSE", "RMSElog", "delta1", "delta2", "delta3")
    normal_keys = ("Nmean", "N11_25", "N22_5", "N30")
    sums = {key: 0.0 for key in depth_keys + normal_keys}
    depth_pixels = normal_pixels = 0.0
    medians = []; batches = visual_count = 0
    with torch.inference_mode():
        for index, cpu_views in enumerate(loader):
            if args.max_test_batches > 0 and index >= args.max_test_batches:
                break
            views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
            output = model(views)
            metrics, tensors = evaluate_batch(output, views, args.depth_scale)
            for key in depth_keys: sums[key] += metrics[key] * metrics["pixels"]
            for key in normal_keys: sums[key] += metrics[key] * metrics["normal_pixels"]
            depth_pixels += metrics["pixels"]; normal_pixels += metrics["normal_pixels"]
            medians.append(metrics["Nmedian"]); batches += 1
            if (args.visualize_every > 0 and index % args.visualize_every == 0
                    and (args.max_visualizations <= 0 or visual_count < args.max_visualizations)):
                save_visuals(output_dir / "visualizations", index, views, tensors)
                visual_count += len(views)
            if index % 20 == 0:
                print(f"[DSEC RGB test] {index:05d}/{len(loader):05d} MAE={metrics['MAE']:.5f} "
                      f"AbsRel={metrics['AbsRel']:.5f} RMSE={metrics['RMSE']:.5f} "
                      f"Nmean={metrics['Nmean']:.3f}", flush=True)
    result = {key: sums[key] / max(depth_pixels, 1.) for key in depth_keys}
    result.update({key: sums[key] / max(normal_pixels, 1.) for key in normal_keys})
    result["Nmedian_batch_mean"] = float(np.mean(medians)) if medians else 0.0
    result.update(batches=batches, pixels=int(depth_pixels), normal_pixels=int(normal_pixels),
                  depth_scale=args.depth_scale, scale_protocol="fixed_no_test_gt_alignment")
    result["delta_lt_1.25"] = result["delta1"]
    result["delta_lt_1.25^2"] = result["delta2"]
    result["delta_lt_1.25^3"] = result["delta3"]
    payload = {"experiment": "DSEC RGB pretrained without finetuning",
               "checkpoint": args.checkpoint, "test_root": args.root, "metrics": result}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
