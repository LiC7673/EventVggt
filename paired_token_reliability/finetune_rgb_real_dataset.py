"""Pure-RGB fine-tuning/evaluation on DSEC or MVSEC (no event model or event input)."""
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
import finetune_no_event as rgb_utils
from eventvggt.datasets.dsec_event_dataset import get_dsec_dataset
from eventvggt.datasets.mvsec_event_dataset import get_mvsec_dataset
from eventvggt.datasets.my_event_dataset import event_multiview_collate
from paired_token_reliability.common import move_views_to_device, torch_load
from streamvggt.models.streamvggt import StreamVGGT


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("dsec", "mvsec"))
    p.add_argument("--root", required=True)
    p.add_argument("--pretrained", default="ckpt/model.pt")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-train-steps", type=int, default=1500)
    p.add_argument("--max-test-batches", type=int, default=0)
    p.add_argument("--lr-head", type=float, default=2e-5)
    p.add_argument("--lr-backbone", type=float, default=2e-6)
    p.add_argument("--unfreeze-last-blocks", type=int, default=2)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-depth", type=float, default=80.)
    p.add_argument("--train-sequence", default="outdoor_day1")
    p.add_argument("--test-sequence", default="outdoor_day2")
    p.add_argument("--intrinsics", nargs=4, type=float, default=None)
    p.add_argument("--visualize-every", type=int, default=10)
    p.add_argument("--max-visualizations", type=int, default=30)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _state_dict(raw):
    state = raw
    while isinstance(state, dict):
        found = next((state[k] for k in ("model", "state_dict", "module")
                      if k in state and isinstance(state[k], dict)), None)
        if found is None: break
        state = found
    return {k.removeprefix("module."): v for k, v in state.items()}


def build_model(a, device):
    model = StreamVGGT(img_size=518, patch_size=14, embed_dim=1024)
    raw = torch_load(a.pretrained)
    mismatch = model.load_state_dict(_state_dict(raw), strict=False)
    missing = [k for k in mismatch.missing_keys if not k.startswith("track_head")]
    unexpected = [k for k in mismatch.unexpected_keys if "event" not in k]
    if missing or unexpected:
        print(f"[RGB checkpoint] missing={missing[:12]} unexpected={unexpected[:12]}", flush=True)
    model.to(device).requires_grad_(False)
    model.depth_head.requires_grad_(True)
    n = max(0, int(a.unfreeze_last_blocks))
    if n:
        for blocks in (model.aggregator.frame_blocks, model.aggregator.global_blocks):
            for block in list(blocks)[-n:]: block.requires_grad_(True)
    # These outputs are not supervised in this RGB depth/normal adaptation.
    model.camera_head.requires_grad_(False)
    model.point_head.requires_grad_(False)
    model.track_head.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    event_names = [name for name, _ in model.named_parameters() if "event" in name.lower()]
    if event_names: raise RuntimeError(f"pure RGB model unexpectedly contains event parameters: {event_names[:8]}")
    print(f"[PURE RGB] trainable={trainable:,}/{total:,}; event parameters=0", flush=True)
    return model


def _drop_events(views):
    views = rgb_utils.drop_event_fields(views)
    for view in views:
        for key in tuple(view):
            if key == "events" or key.startswith("event_") or key in {
                "geometry_event_voxel", "full_event_voxel", "cur_event_voxel"
            }:
                view.pop(key, None)
    return views


def _collate(batch):
    return _drop_events(event_multiview_collate(batch))


def loader(a, train):
    if a.dataset == "dsec":
        split = "train" if train else "test"
        dataset = get_dsec_dataset(
            a.root, split=split, num_views=a.num_views, resolution=(518, 392), seed=0,
            event_window_ms=50., event_resize_bins=5, clip_stride=4,
            allow_unaligned_rgb=False, depth_scale=1., max_depth=a.max_depth,
        )
        label = f"DSEC/{split}"
    else:
        sequence = a.train_sequence if train else a.test_sequence
        dataset = get_mvsec_dataset(
            a.root, split="all", sequence_names=[sequence], num_views=a.num_views,
            resolution=(518, 392), fps=20, seed=0, camera="left",
            intrinsics=a.intrinsics, event_resize_method="voxel_linear_time", event_resize_bins=5,
        )
        label = f"MVSEC/{sequence}"
    if len(dataset) == 0: raise RuntimeError(f"empty pure-RGB dataset: {label}, root={a.root}")
    print(f"[PURE RGB] {label}: {dataset.get_stats()}", flush=True)
    return DataLoader(dataset, batch_size=1, shuffle=train, num_workers=a.num_workers,
                      pin_memory=True, drop_last=False, collate_fn=_collate)


def objective(output, views, max_depth):
    pred = torch.stack([item["depth"][..., 0] for item in output.ress], 1).float()
    gt = fe.stack_view_field(views, "depthmap").float()
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
    valid = torch.isfinite(pred) & torch.isfinite(gt) & (pred > 1e-6) & (gt > .1) & (gt < max_depth)
    pixels = valid.sum().clamp_min(1)
    signed_log = torch.log(pred.clamp_min(1e-6)) - torch.log(gt.clamp_min(1e-6))
    difference = (pred - gt).abs()
    depth_loss = (signed_log.abs() * valid).sum() / pixels
    mae = (difference * valid).sum() / pixels
    absrel = ((difference / gt.clamp_min(1e-6)) * valid).sum() / pixels
    rmselog = torch.sqrt((signed_log.square() * valid).sum() / pixels)
    ratio = torch.maximum(pred / gt.clamp_min(1e-6), gt / pred.clamp_min(1e-6))
    delta = [((ratio < 1.25 ** power) & valid).sum() / pixels for power in (1, 2, 3)]
    pn = F.normalize(fe.depth_to_normals(pred, intrinsics), dim=-1, eps=1e-6)
    gn = F.normalize(fe.depth_to_normals(gt, intrinsics), dim=-1, eps=1e-6)
    cosine = (pn * gn).sum(-1).clamp(-1, 1)
    normal_valid = fe.normal_stencil_valid_mask(valid, pred, eps=1e-6)
    normal_pixels = normal_valid.sum().clamp_min(1)
    normal_loss = ((1 - cosine) * normal_valid).sum() / normal_pixels
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6)))
    nmean = (angle * normal_valid).sum() / normal_pixels
    n11 = ((angle < 11.25) & normal_valid).sum() / normal_pixels
    n22 = ((angle < 22.5) & normal_valid).sum() / normal_pixels
    n30 = ((angle < 30.) & normal_valid).sum() / normal_pixels
    loss = depth_loss + .2 * normal_loss
    details = dict(MAE=mae, AbsRel=absrel, RMSElog=rmselog, delta1=delta[0],
                   delta2=delta[1], delta3=delta[2], Nmean=nmean,
                   N11_25=n11, N22_5=n22, N30=n30,
                   depth_loss=depth_loss, normal_loss=normal_loss, pixels=valid.sum().float())
    return loss, details, (pred, gt, valid, pn, gn, normal_valid)


def save_visual(root, index, views, tensors):
    root.mkdir(parents=True, exist_ok=True)
    pred, gt, valid, pn, gn, normal_valid = tensors
    for view_index, view in enumerate(views):
        mask = valid[0, view_index]
        values = torch.cat((pred[0, view_index][mask], gt[0, view_index][mask]))
        if values.numel() == 0: continue
        lo, hi = float(values.min()), float(values.max())
        rgb = view["img"][0].detach().permute(1, 2, 0).cpu().clamp(0, 1)
        error = ((pred[0, view_index] - gt[0, view_index]).abs() * mask).detach().cpu()
        pred_n = ((pn[0, view_index].detach().cpu() + 1) * .5) * normal_valid[0, view_index].cpu().unsqueeze(-1)
        gt_n = ((gn[0, view_index].detach().cpu() + 1) * .5) * normal_valid[0, view_index].cpu().unsqueeze(-1)
        panels = ((rgb, "RGB input", None, None, None),
                  (pred[0, view_index].detach().cpu() * mask.cpu(), "RGB pred depth", "viridis", lo, hi),
                  (gt[0, view_index].detach().cpu() * mask.cpu(), "GT depth", "viridis", lo, hi),
                  (error, "|pred-GT|", "magma", 0, float(torch.quantile(error.flatten(), .995).clamp_min(1e-6))),
                  (pred_n, "RGB depth-derived normal", None, None, None),
                  (gt_n, "GT normal", None, None, None))
        fig, axes = plt.subplots(2, 3, figsize=(16, 10)); axes = axes.reshape(-1)
        for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
            shown = axis.imshow(np.asarray(image), cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title); axis.axis("off")
            if cmap: fig.colorbar(shown, ax=axis, fraction=.046, pad=.04)
        fig.tight_layout(); fig.savefig(root / f"batch_{index:06d}_view_{view_index:02d}.png", dpi=130); plt.close(fig)


def run(model, data, device, max_depth, optimizer=None, max_batches=0, visual_dir=None,
        visualize_every=10, max_visualizations=30):
    training = optimizer is not None
    model.train(training)
    model.camera_head.eval(); model.point_head.eval(); model.track_head.eval()
    keys = ("MAE", "AbsRel", "RMSElog", "delta1", "delta2", "delta3",
            "Nmean", "N11_25", "N22_5", "N30", "depth_loss", "normal_loss")
    totals = {key: 0. for key in keys}; pixel_total = loss_total = 0.; batches = visual_count = 0
    for index, cpu_views in enumerate(data):
        if max_batches > 0 and index >= max_batches: break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        if any("event_voxel" in view for view in views): raise RuntimeError("event input leaked into pure RGB forward")
        with torch.set_grad_enabled(training):
            output = model(views); loss, details, tensors = objective(output, views, max_depth)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
        weight = float(details["pixels"])
        for key in keys: totals[key] += float(details[key]) * weight
        pixel_total += weight; loss_total += float(loss); batches += 1
        if (visual_dir is not None and visualize_every > 0 and index % visualize_every == 0
                and (max_visualizations <= 0 or visual_count < max_visualizations)):
            save_visual(Path(visual_dir), index, views, tensors); visual_count += len(views)
        if index % 20 == 0:
            phase = "train" if training else "test"
            print(f"[RGB {phase}] {index:05d}/{len(data):05d} loss={float(loss):.5f} "
                  f"MAE={float(details['MAE']):.5f} AbsRel={float(details['AbsRel']):.5f} "
                  f"RMSElog={float(details['RMSElog']):.5f} Nmean={float(details['Nmean']):.3f}", flush=True)
    result = {key: value / max(pixel_total, 1.) for key, value in totals.items()}
    result.update(loss=loss_total / max(batches, 1), batches=batches, pixels=int(pixel_total))
    return result


def main():
    a = arguments(); device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    model = build_model(a, device); train = loader(a, True); test = loader(a, False)
    backbone_params = []
    n = max(0, int(a.unfreeze_last_blocks))
    if n:
        for blocks in (model.aggregator.frame_blocks, model.aggregator.global_blocks):
            for block in list(blocks)[-n:]: backbone_params.extend(block.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.depth_head.parameters(), "lr": a.lr_head},
        {"params": backbone_params, "lr": a.lr_backbone},
    ], weight_decay=1e-5, betas=(.9, .95))
    history=[]; steps=0
    for epoch in range(a.epochs):
        remaining = max(a.max_train_steps - steps, 0) if a.max_train_steps else 0
        limit = min(len(train), remaining) if a.max_train_steps else 0
        tr = run(model, train, device, a.max_depth, optimizer, limit)
        steps += limit if limit else len(train)
        te = run(model, test, device, a.max_depth, max_batches=a.max_test_batches)
        row = {"epoch": epoch, "train": tr, "test": te}; history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        torch.save({"schema": "pure_rgb_real_dataset_v1", "model": model.state_dict(),
                    "dataset": a.dataset, "source_pretrained": a.pretrained, "history": history}, out / "checkpoint-last.pth")
        (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if a.max_train_steps and steps >= a.max_train_steps: break
    final = run(model, test, device, a.max_depth, max_batches=a.max_test_batches,
                visual_dir=out / "final_test_visualizations", visualize_every=a.visualize_every,
                max_visualizations=a.max_visualizations)
    payload = {"dataset": a.dataset, "event_input": False, "source_pretrained": a.pretrained,
               "train_sequence": a.train_sequence if a.dataset == "mvsec" else "DSEC/train",
               "test_sequence": a.test_sequence if a.dataset == "mvsec" else "DSEC/test", "metrics": final}
    (out / "final_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[PURE RGB final] {json.dumps(payload, ensure_ascii=False)}", flush=True)


if __name__ == "__main__": main()
