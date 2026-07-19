"""Light-tune the best refiner-first checkpoint on MVSEC day and evaluate nights."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.mvsec_event_dataset import get_mvsec_dataset, event_multiview_collate
from paired_token_reliability.common import move_views_to_device, strip_module_prefix, torch_load
from paired_token_reliability.train_linear_voxel_cur_event_hf_residual import build_model
from paired_token_reliability import train_unified_geometry_contribution as pipeline


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="config/finetune_event.yaml")
    p.add_argument("--root", default="/data1/lzh/dataset/MVSEC_raw/converted_hdf5")
    p.add_argument("--output", default="exp_f/mvsec_outdoor_day2_to_night_gpu5")
    p.add_argument("--train-sequence", default="outdoor_day2")
    p.add_argument("--test-sequences", nargs="+", default=["outdoor_night1", "outdoor_night2", "outdoor_night3"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-train-steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--event-bins", type=int, default=5)
    p.add_argument("--max-depth", type=float, default=80.)
    p.add_argument("--intrinsics", nargs=4, type=float, default=None, metavar=("FX", "FY", "CX", "CY"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-test-batches", type=int, default=0)
    p.add_argument("--visualize-every", type=int, default=10)
    p.add_argument("--max-visualizations", type=int, default=30)
    return p.parse_args()


def load_model(a, device):
    raw = torch_load(a.checkpoint); state = raw.get("model", raw)
    cfg = pipeline.load_cfg(a.config, [])
    pretrained = raw.get("config", {}).get("pretrained", "ckpt/model.pt") if isinstance(raw, dict) else "ckpt/model.pt"
    if not Path(pretrained).is_file(): pretrained = "ckpt/model.pt"
    model = build_model(cfg, argparse.Namespace(pretrained=pretrained), device)
    mismatch = model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(state)), strict=True)
    if mismatch.missing_keys or mismatch.unexpected_keys: raise RuntimeError(f"checkpoint mismatch: {mismatch}")
    # MVSEC has no paired E_geo/HDR teachers.  Its single real event stream is
    # an inference-time Full stream, so retain the pretrained aligner and
    # frozen learned C while disabling only the teacher requirements.
    model.require_geo_teacher = False
    model.require_hdr_teacher = False
    model.set_confidence_stage("full")
    model.requires_grad_(False)
    for module in (model.event_encoder, model.event_normal_decoder, model.pixel_depth_refiner):
        module.requires_grad_(True)
    # Preserve the learned HDR-like space; permit only a small domain correction.
    model.event_token_projection.requires_grad_(True)
    model.ldr_event_hdr_aligner.requires_grad_(True)
    model.aggregator.eval(); model.camera_head.eval(); model.depth_head.eval(); model.point_head.eval()
    return model


def make_loader(a, sequences, shuffle):
    ds = get_mvsec_dataset(
        a.root, split="all", sequence_names=sequences, num_views=a.num_views,
        resolution=(518, 392), fps=20, seed=0, camera="left", intrinsics=a.intrinsics,
        event_resize_method="voxel_linear_time", event_resize_bins=a.event_bins,
    )
    if len(ds) == 0: raise RuntimeError(f"No MVSEC clips for {sequences} under {a.root}")
    print(f"[MVSEC] sequences={sequences}: {ds.get_stats()}", flush=True)
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=a.num_workers,
                      pin_memory=True, collate_fn=event_multiview_collate, drop_last=False)


def losses_and_metrics(output, views, max_depth):
    pred = torch.stack([x["depth"][..., 0] for x in output.ress], 1).float()
    base = torch.stack([x["depth_hdr_base"][..., 0] for x in output.ress], 1).float()
    gt = fe.stack_view_field(views, "depthmap").float()
    k = fe.stack_view_field(views, "camera_intrinsics").float()
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > .1) & (gt < max_depth) & (pred > 1e-6)
    n = valid.sum().clamp_min(1); logdiff = torch.log(pred.clamp_min(1e-6)) - torch.log(gt.clamp_min(1e-6))
    mae = ((pred - gt).abs() * valid).sum() / n
    absrel = (((pred - gt).abs() / gt.clamp_min(1e-6)) * valid).sum() / n
    rmselog = torch.sqrt((logdiff.square() * valid).sum() / n)
    ratio = torch.maximum(pred / gt.clamp_min(1e-6), gt / pred.clamp_min(1e-6))
    d1 = ((ratio < 1.25) & valid).sum() / n
    d2 = ((ratio < 1.25 ** 2) & valid).sum() / n
    d3 = ((ratio < 1.25 ** 3) & valid).sum() / n
    pn = F.normalize(fe.depth_to_normals(pred, k), dim=-1, eps=1e-6)
    gn = F.normalize(fe.depth_to_normals(gt, k), dim=-1, eps=1e-6)
    cosine = (pn * gn).sum(-1).clamp(-1, 1)
    normal_valid = fe.normal_stencil_valid_mask(valid, pred, eps=1e-6)
    nn = normal_valid.sum().clamp_min(1)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6)))
    nmean = (angle * normal_valid).sum() / nn
    normal_loss = ((1 - cosine) * normal_valid).sum() / nn
    residual = torch.log(gt.clamp_min(1e-6)) - torch.log(base.clamp_min(1e-6))
    flat, mask = residual.flatten(0, 1).unsqueeze(1), valid.flatten(0, 1).unsqueeze(1).float()
    mass = F.avg_pool2d(mask, 9, 1, 4); low = F.avg_pool2d(flat * mask, 9, 1, 4) / mass.clamp_min(1e-6)
    hf_target = (flat - low).reshape_as(residual).detach()
    update = torch.stack([x["pixel_refiner_bounded_update"] for x in output.ress], 1).float()
    hf = (F.smooth_l1_loss(update, hf_target, beta=.01, reduction="none") * valid).sum() / n
    objective = logdiff.abs().mul(valid).sum() / n + .2 * normal_loss + hf
    metrics = {"loss": objective, "MAE": mae, "AbsRel": absrel, "RMSElog": rmselog,
               "delta1": d1, "delta2": d2, "delta3": d3,
               "Nmean": nmean, "HF": hf, "pixels": valid.sum().float()}
    return objective, metrics


def _normal_image(normal, mask):
    image = ((normal.detach().float().cpu() + 1.) * .5).clamp(0, 1)
    return image * mask.detach().float().cpu().unsqueeze(-1)


def save_visuals(root, index, views, output, max_depth):
    root.mkdir(parents=True, exist_ok=True)
    pred = torch.stack([x["depth"][..., 0] for x in output.ress], 1).float()
    coarse = torch.stack([x["depth_hdr_base"][..., 0] for x in output.ress], 1).float()
    gt = fe.stack_view_field(views, "depthmap").float()
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > .1) & (gt < max_depth) & (pred > 1e-6)
    pred_normal = F.normalize(fe.depth_to_normals(pred, intrinsics), dim=-1, eps=1e-6)
    gt_normal = F.normalize(fe.depth_to_normals(gt, intrinsics), dim=-1, eps=1e-6)
    normal_valid = fe.normal_stencil_valid_mask(valid, gt, eps=1e-6)
    contribution = torch.stack([x["event_contribution"] for x in output.ress], 1).float()
    for view_index, view in enumerate(views):
        mask = valid[0, view_index]
        values = torch.cat((coarse[0, view_index][mask], pred[0, view_index][mask], gt[0, view_index][mask]))
        if values.numel() == 0: continue
        vmin, vmax = float(values.min()), float(values.max())
        rgb = view["img"][0].detach().float().permute(1, 2, 0).cpu().clamp(0, 1)
        event = view["event_voxel"][0].detach().float().abs().sum(0).cpu()
        error = ((pred[0, view_index] - gt[0, view_index]).abs() * mask).detach().cpu()
        panels = (
            (rgb, "input RGB", None, None, None),
            (event, "event voxel |sum|", "gray", 0, float(torch.quantile(event.flatten(), .995).clamp_min(1e-6))),
            (coarse[0, view_index].detach().cpu() * mask.cpu(), "coarse depth", "viridis", vmin, vmax),
            (pred[0, view_index].detach().cpu() * mask.cpu(), "final depth", "viridis", vmin, vmax),
            (gt[0, view_index].detach().cpu() * mask.cpu(), "GT depth", "viridis", vmin, vmax),
            (error, "|final-GT|", "magma", 0, float(torch.quantile(error.flatten(), .995).clamp_min(1e-6))),
            (_normal_image(pred_normal[0, view_index], normal_valid[0, view_index]), "final normal", None, None, None),
            (_normal_image(gt_normal[0, view_index], normal_valid[0, view_index]), "GT normal", None, None, None),
            (contribution[0, view_index].detach().cpu(), "contribution C [0,1]", "magma", 0, 1),
        )
        fig, axes = plt.subplots(3, 3, figsize=(18, 17)); axes = axes.reshape(-1)
        for axis, (image, title, cmap, lo, hi) in zip(axes, panels):
            shown = axis.imshow(np.asarray(image), cmap=cmap, vmin=lo, vmax=hi)
            axis.set_title(title); axis.axis("off")
            if cmap is not None: fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
        raw = view.get("instance", [f"batch_{index:06d}"])[0]
        safe = str(raw).replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.suptitle(f"MVSEC outdoor_day2 held-out: {safe}, view={view_index}")
        fig.tight_layout(); fig.savefig(root / f"batch_{index:06d}_view_{view_index:02d}_{safe}.png", dpi=130); plt.close(fig)


def run(model, loader, device, max_depth, optimizer=None, max_steps=0,
        visual_dir=None, visualize_every=10, max_visualizations=30):
    train = optimizer is not None; model.train(train)
    model.aggregator.eval(); model.camera_head.eval(); model.depth_head.eval(); model.point_head.eval()
    sums, batches, pixel_total = defaultdict(float), 0, 0.
    visual_count = 0
    for i, cpu_views in enumerate(loader):
        if max_steps and i >= max_steps: break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        with torch.set_grad_enabled(train):
            output = model(views); loss, metrics = losses_and_metrics(output, views, max_depth)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
        if (visual_dir is not None and visualize_every > 0 and i % visualize_every == 0
                and (max_visualizations <= 0 or visual_count < max_visualizations)):
            save_visuals(Path(visual_dir), i, views, output, max_depth)
            visual_count += len(views)
        pixels = float(metrics["pixels"].detach())
        for key, value in metrics.items():
            if key != "pixels": sums[key] += float(value.detach()) * pixels
        pixel_total += pixels
        batches += 1
        if i % 20 == 0:
            phase = "train" if train else "test"
            print(f"[MVSEC {phase}] step={i:05d}/{len(loader):05d} loss={float(loss):.5f} "
                  f"MAE={float(metrics['MAE']):.5f} AbsRel={float(metrics['AbsRel']):.5f} "
                  f"RMSElog={float(metrics['RMSElog']):.5f} Nmean={float(metrics['Nmean']):.3f}", flush=True)
    result = {k: v / max(pixel_total, 1.) for k, v in sums.items()}
    result.update(pixels=pixel_total, batches=batches)
    return result


def main():
    a = arguments(); device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    model = load_model(a, device)
    train_loader = make_loader(a, [a.train_sequence], True)
    test_loaders = {s: make_loader(a, [s], False) for s in a.test_sequences}
    optimizer = torch.optim.AdamW([
        {"params": model.pixel_depth_refiner.parameters(), "lr": a.lr},
        {"params": model.event_encoder.parameters(), "lr": a.lr},
        {"params": model.event_normal_decoder.parameters(), "lr": a.lr},
        {"params": model.event_token_projection.parameters(), "lr": .1 * a.lr},
        {"params": model.ldr_event_hdr_aligner.parameters(), "lr": .1 * a.lr},
    ], weight_decay=1e-5, betas=(.9, .95))
    history=[]; steps=0
    for epoch in range(a.epochs):
        limit = max(a.max_train_steps - steps, 0) if a.max_train_steps else 0
        tr = run(model, train_loader, device, a.max_depth, optimizer, limit)
        steps += min(len(train_loader), limit) if limit else len(train_loader)
        tests = {name: run(model, dl, device, a.max_depth, max_steps=a.max_test_batches) for name, dl in test_loaders.items()}
        row = {"epoch": epoch, "train_sequence": a.train_sequence, "train": tr, "test": tests}
        history.append(row); print(json.dumps(row, indent=2), flush=True)
        torch.save({"schema": "mvsec_refiner_first_day_to_night_v1", "model": model.state_dict(),
                    "source_checkpoint": a.checkpoint, "history": history}, out / "checkpoint-last.pth")
        (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if a.max_train_steps and steps >= a.max_train_steps: break
    final_tests = {}
    for name, dl in test_loaders.items():
        final_tests[name] = run(
            model, dl, device, a.max_depth, max_steps=a.max_test_batches,
            visual_dir=out / "final_test_visualizations" / name,
            visualize_every=a.visualize_every, max_visualizations=a.max_visualizations,
        )
    payload = {"source_checkpoint": a.checkpoint, "train_sequence": a.train_sequence,
               "test_sequences": a.test_sequences, "metrics": final_tests}
    (out / "final_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[MVSEC final-test] {json.dumps(final_tests, ensure_ascii=False)}", flush=True)


if __name__ == "__main__": main()
