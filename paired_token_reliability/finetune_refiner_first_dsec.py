"""Lightly adapt a trained refiner-first checkpoint to single-stream DSEC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    log_error = (torch.log(pred.clamp_min(1e-6)) - torch.log(gt.clamp_min(1e-6))).abs()
    depth = (log_error * valid).sum() / valid.sum().clamp_min(1)
    pn = F.normalize(fe.depth_to_normals(pred, k), dim=-1, eps=1e-6)
    gn = F.normalize(fe.depth_to_normals(gt, k), dim=-1, eps=1e-6)
    normal = ((1. - (pn * gn).sum(-1).clamp(-1, 1)) * valid).sum() / valid.sum().clamp_min(1)
    residual = torch.log(gt.clamp_min(1e-6)) - torch.log(base.clamp_min(1e-6))
    bv = residual.flatten(0, 1).unsqueeze(1); vv = valid.flatten(0, 1).unsqueeze(1).float()
    weight = F.avg_pool2d(vv, 9, 1, 4)
    low = F.avg_pool2d(bv * vv, 9, 1, 4) / weight.clamp_min(1e-6)
    target = (bv - low).reshape_as(residual).detach()
    update = torch.stack([x["pixel_refiner_bounded_update"] for x in output.ress], 1).float()
    hf = (F.smooth_l1_loss(update, target, beta=.01, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    return depth + .2 * normal + hf, dict(depth=float(depth.detach()), normal=float(normal.detach()), hf=float(hf.detach()))


def run(model, data, device, optimizer=None, max_batches=0):
    training = optimizer is not None; model.train(training)
    model.aggregator.eval(); model.camera_head.eval(); model.depth_head.eval(); model.point_head.eval()
    totals = dict(loss=0., depth=0., normal=0., hf=0.); count = 0
    for index, cpu_views in enumerate(data):
        if max_batches > 0 and index >= max_batches: break
        views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
        with torch.set_grad_enabled(training):
            output = model(views); loss, details = objective(output, views)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
        totals["loss"] += float(loss.detach())
        for key, value in details.items(): totals[key] += value
        count += 1
        if training and index % 20 == 0: print(f"[DSEC train] step={index:05d} loss={float(loss):.5f} {details}", flush=True)
    return {k: v / max(count, 1) for k, v in totals.items()}


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


if __name__ == "__main__": main()
