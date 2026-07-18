"""Lightly adapt a trained refiner-first checkpoint to single-stream DSEC."""
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
from paired_token_reliability.common import move_views_to_device, strip_module_prefix, torch_load
from paired_token_reliability.train_linear_voxel_cur_event_hf_residual import build_model
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="config/finetune_event.yaml")
    p.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    p.add_argument("--output", default="exp_f/dsec_refiner_first_light_finetune")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--max-train-steps", type=int, default=2000)
    p.add_argument("--max-test-batches", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--visualize-every", type=int, default=10)
    p.add_argument("--max-visualizations", type=int, default=30,
                   help="maximum final-test figures; 0 means unlimited")
    return p.parse_args()


def load_model(args, device):
    raw = torch_load(args.checkpoint)
    state = raw.get("model", raw)
    cfg = pipeline.load_cfg(args.config, [])
    # build_model needs a base file, but its freshly initialized state is
    # immediately replaced by the complete refiner-first checkpoint.
    base = raw.get("config", {}).get("pretrained", "ckpt/model.pt") if isinstance(raw, dict) else "ckpt/model.pt"
    holder = argparse.Namespace(pretrained=base)
    if not Path(holder.pretrained).is_file(): holder.pretrained = "ckpt/model.pt"
    model = build_model(cfg, holder, device)
    loaded = model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(state)), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {loaded}")
    runtime = (raw.get("trainer_state") or {}).get("runtime_state", {})
    model._dual_alignment_step = max(int(runtime.get("dual_alignment_step", 0)), 2500)
    # DSEC supplies only the inference-time event stream.  Keep the pretrained
    # Full->Geo aligner and frozen learned C, but disable teacher-only checks.
    # Using stage="geo" here is incorrect: it means a clean E_geo training
    # epoch, bypasses the aligner, and requires geometry_event_voxel.
    model.require_geo_teacher = False
    model.require_hdr_teacher = False
    model.set_confidence_stage("full")
    model.requires_grad_(False)
    for module in (model.event_encoder, model.event_normal_decoder, model.pixel_depth_refiner):
        module.requires_grad_(True)
    model.event_token_projection.requires_grad_(True)
    model.ldr_event_hdr_aligner.requires_grad_(True)
    model.aggregator.eval(); model.camera_head.eval(); model.depth_head.eval(); model.point_head.eval()
    return model


def loader(args, split, shuffle):
    dataset = get_dsec_dataset(
        args.root, split=split, num_views=args.num_views, resolution=(518, 392),
        seed=0, event_window_ms=50., event_resize_bins=5, clip_stride=4,
        allow_unaligned_rgb=False, depth_scale=1., max_depth=80.,
    )
    print(f"[DSEC] {split}: {dataset.get_stats()}", flush=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True,
                      collate_fn=event_multiview_collate, drop_last=False)


def objective(output, views):
    pred = torch.stack([x["depth"][..., 0] for x in output.ress], 1).float()
    base = torch.stack([x["depth_hdr_base"][..., 0] for x in output.ress], 1).float()
    gt = fe.stack_view_field(views, "depthmap").float()
    k = fe.stack_view_field(views, "camera_intrinsics").float()
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > .1) & (gt < 80.) & (pred > 1e-6)
    pixels = valid.sum().clamp_min(1)
    signed_log_error = torch.log(pred.clamp_min(1e-6)) - torch.log(gt.clamp_min(1e-6))
    log_error = signed_log_error.abs()
    depth = (log_error * valid).sum() / pixels
    difference = (pred - gt).abs()
    mae = (difference * valid).sum() / pixels
    abs_rel = ((difference / gt.clamp_min(1e-6)) * valid).sum() / pixels
    rmse_log = torch.sqrt((signed_log_error.square() * valid).sum() / pixels)
    ratio = torch.maximum(pred / gt.clamp_min(1e-6), gt / pred.clamp_min(1e-6))
    delta1 = ((ratio < 1.25) & valid).sum().float() / pixels
    delta2 = ((ratio < 1.25 ** 2) & valid).sum().float() / pixels
    delta3 = ((ratio < 1.25 ** 3) & valid).sum().float() / pixels
    pn = F.normalize(fe.depth_to_normals(pred, k), dim=-1, eps=1e-6)
    gn = F.normalize(fe.depth_to_normals(gt, k), dim=-1, eps=1e-6)
    cosine = (pn * gn).sum(-1).clamp(-1, 1)
    normal_valid = fe.normal_stencil_valid_mask(valid, pred, eps=1e-6)
    normal_pixels = normal_valid.sum().clamp_min(1)
    normal = ((1. - cosine) * normal_valid).sum() / normal_pixels
    normal_mean = (torch.rad2deg(torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))) * normal_valid).sum() / normal_pixels
    residual = torch.log(gt.clamp_min(1e-6)) - torch.log(base.clamp_min(1e-6))
    bv = residual.flatten(0, 1).unsqueeze(1); vv = valid.flatten(0, 1).unsqueeze(1).float()
    weight = F.avg_pool2d(vv, 9, 1, 4)
    low = F.avg_pool2d(bv * vv, 9, 1, 4) / weight.clamp_min(1e-6)
    target = (bv - low).reshape_as(residual).detach()
    update = torch.stack([x["pixel_refiner_bounded_update"] for x in output.ress], 1).float()
    hf = (F.smooth_l1_loss(update, target, beta=.01, reduction="none") * valid).sum() / pixels
    return depth + .2 * normal + hf, dict(
        depth=float(depth.detach()), normal=float(normal.detach()), hf=float(hf.detach()),
        MAE=float(mae.detach()), AbsRel=float(abs_rel.detach()), RMSElog=float(rmse_log.detach()),
        delta1=float(delta1.detach()), delta2=float(delta2.detach()), delta3=float(delta3.detach()),
        Nmean=float(normal_mean.detach()), pixels=float(pixels.detach()),
    )


def _normal_image(normal, mask):
    image = ((normal.detach().float().cpu() + 1.) * .5).clamp(0, 1)
    return image * mask.detach().float().cpu().unsqueeze(-1)


def save_test_visuals(root, index, views, output):
    root.mkdir(parents=True, exist_ok=True)
    pred = torch.stack([x["depth"][..., 0] for x in output.ress], 1).float()
    gt = fe.stack_view_field(views, "depthmap").float()
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > .1) & (gt < 80.) & (pred > 1e-6)
    pred_normal = F.normalize(fe.depth_to_normals(pred, intrinsics), dim=-1, eps=1e-6)
    gt_normal = F.normalize(fe.depth_to_normals(gt, intrinsics), dim=-1, eps=1e-6)
    normal_valid = fe.normal_stencil_valid_mask(valid, gt, eps=1e-6)
    contribution = torch.stack([x["event_contribution"] for x in output.ress], 1).float()
    for view_index, view in enumerate(views):
        mask = valid[0, view_index]; values = torch.cat((pred[0, view_index][mask], gt[0, view_index][mask]))
        if values.numel() == 0: continue
        vmin, vmax = float(values.min()), float(values.max())
        rgb = view["img"][0].detach().float().permute(1, 2, 0).cpu().clamp(0, 1)
        event = view["event_voxel"][0].detach().float().abs().sum(0).cpu()
        error = ((pred[0, view_index] - gt[0, view_index]).abs() * mask).detach().cpu()
        panels = (
            (rgb, "input RGB", None, None, None),
            (event, "event voxel |sum|", "gray", 0, float(torch.quantile(event.flatten(), .995).clamp_min(1e-6))),
            (pred[0, view_index].detach().cpu() * mask.cpu(), "pred depth", "viridis", vmin, vmax),
            (gt[0, view_index].detach().cpu() * mask.cpu(), "GT depth", "viridis", vmin, vmax),
            (error, "|pred-GT| depth", "magma", 0, float(torch.quantile(error.flatten(), .995).clamp_min(1e-6))),
            (_normal_image(pred_normal[0, view_index], normal_valid[0, view_index]), "pred normal", None, None, None),
            (_normal_image(gt_normal[0, view_index], normal_valid[0, view_index]), "GT normal", None, None, None),
            (contribution[0, view_index].detach().cpu(), "predicted contribution C", "magma", 0, 1),
        )
        fig, axes = plt.subplots(2, 4, figsize=(20, 10)); axes = axes.reshape(-1)
        for axis, (image, title, cmap, lo, hi) in zip(axes, panels):
            shown = axis.imshow(np.asarray(image), cmap=cmap, vmin=lo, vmax=hi)
            axis.set_title(title); axis.axis("off")
            if cmap is not None: fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
        raw = view.get("instance", [f"batch_{index:06d}"])[0]
        safe = str(raw).replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.suptitle(f"DSEC final test: {safe}, view={view_index}")
        fig.tight_layout(); fig.savefig(root / f"batch_{index:06d}_view_{view_index:02d}_{safe}.png", dpi=130); plt.close(fig)


def run(model, data, device, optimizer=None, max_batches=0, visual_dir=None,
        visualize_every=10, max_visualizations=30):
    training = optimizer is not None; model.train(training)
    model.aggregator.eval(); model.camera_head.eval(); model.depth_head.eval(); model.point_head.eval()
    metric_keys = ("depth", "normal", "hf", "MAE", "AbsRel", "RMSElog", "delta1", "delta2", "delta3", "Nmean")
    totals = {key: 0. for key in metric_keys}; loss_total = pixel_total = 0.; count = 0
    visual_count = 0
    for index, cpu_views in enumerate(data):
        if max_batches > 0 and index >= max_batches: break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        with torch.set_grad_enabled(training):
            output = model(views); loss, details = objective(output, views)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
        if (visual_dir is not None and visualize_every > 0 and index % visualize_every == 0
                and (max_visualizations <= 0 or visual_count < max_visualizations)):
            save_test_visuals(Path(visual_dir), index, views, output)
            visual_count += len(views)
        weight = details["pixels"]; loss_total += float(loss.detach())
        for key in metric_keys: totals[key] += details[key] * weight
        pixel_total += weight
        count += 1
        if index % 20 == 0:
            phase = "train" if training else "test"
            print(f"[DSEC {phase}] step={index:05d}/{len(data):05d} loss={float(loss):.5f} "
                  f"MAE={details['MAE']:.5f} AbsRel={details['AbsRel']:.5f} "
                  f"RMSElog={details['RMSElog']:.5f} Nmean={details['Nmean']:.3f}", flush=True)
    result = {key: value / max(pixel_total, 1.) for key, value in totals.items()}
    result.update(loss=loss_total / max(count, 1), batches=count, pixels=int(pixel_total))
    return result


def main():
    args = parse_args(); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model = load_model(args, device)
    train = loader(args, "train", True); test = loader(args, "test", False)
    optimizer = torch.optim.AdamW([
        {"params": model.pixel_depth_refiner.parameters(), "lr": args.lr},
        {"params": model.event_encoder.parameters(), "lr": args.lr},
        {"params": model.event_normal_decoder.parameters(), "lr": args.lr},
        {"params": model.event_token_projection.parameters(), "lr": .2 * args.lr},
        {"params": model.ldr_event_hdr_aligner.parameters(), "lr": .2 * args.lr},
    ], weight_decay=1e-5, betas=(.9, .95))
    history=[]; global_steps=0
    for epoch in range(args.epochs):
        remaining = max(args.max_train_steps - global_steps, 0) if args.max_train_steps > 0 else 0
        train_limit = min(len(train), remaining) if args.max_train_steps > 0 else 0
        tr = run(model, train, device, optimizer, train_limit)
        global_steps += train_limit if train_limit > 0 else len(train)
        te = run(model, test, device, None, args.max_test_batches)
        history.append(dict(epoch=epoch, train=tr, test=te)); print(history[-1], flush=True)
        torch.save({"schema": "dsec_refiner_first_light_v1", "model": model.state_dict(),
                    "source_checkpoint": args.checkpoint, "history": history}, out / "checkpoint-last.pth")
        (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.max_train_steps > 0 and global_steps >= args.max_train_steps: break
    print("[DSEC final-test] running complete held-out test after fine-tuning", flush=True)
    final_test = run(model, test, device, None, args.max_test_batches,
                     visual_dir=out / "final_test_visualizations",
                     visualize_every=args.visualize_every,
                     max_visualizations=args.max_visualizations)
    payload = {"checkpoint": str(out / "checkpoint-last.pth"), "source_checkpoint": args.checkpoint,
               "test_split": "DSEC_EV_VGGT/test", "metrics": final_test}
    (out / "final_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DSEC final-test] {json.dumps(final_test, ensure_ascii=False)}", flush=True)


if __name__ == "__main__": main()
