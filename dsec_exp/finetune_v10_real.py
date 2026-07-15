"""Fine-tune a pretrained V10 model on real single-stream DSEC data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.dsec_event_dataset import DSECEventDataset
from eventvggt.datasets.my_event_dataset import event_multiview_collate
from paired_token_reliability.common import torch_load
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10


def loader(args, split, shuffle):
    dataset = DSECEventDataset(
        ROOT=args.root, dsec_split=split, split=split,
        num_views=args.num_views, resolution=(args.width, args.height),
        seed=args.seed, event_window_ms=args.event_window_ms,
        event_resize_bins=5, clip_stride=args.train_stride if shuffle else args.test_stride,
        allow_unaligned_rgb=args.allow_unaligned_rgb,
        depth_scale=args.depth_scale, max_depth=args.depth_max,
    )
    print(f"[DSEC V10] split={split} scenes={dataset.get_active_scenes()} clips={len(dataset)}")
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True, drop_last=shuffle,
        collate_fn=event_multiview_collate,
    )


def build_model(checkpoint, device):
    compatible_schemas = {
        "linear_time_voxel_dual_alignment_hdr_decoupled_gates_v10",
        "linear_time_voxel_dual_alignment_hdr_no_point_refiner_v10",
        "linear_time_voxel_dual_alignment_hdr_event_conditioned_adapter_v10",
    }
    if checkpoint.get("schema") not in compatible_schemas:
        raise RuntimeError(f"expected trained V10 checkpoint, got schema={checkpoint.get('schema')!r}")
    cfg = OmegaConf.create(checkpoint["cfg"])
    m = cfg.model
    from paired_token_reliability.linear_voxel_dual_alignment_hdr_model import DualAlignmentHDRLinearVoxelModel
    model = DualAlignmentHDRLinearVoxelModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 1)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 5)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        alignment_confidence_tau=float(getattr(m, "alignment_confidence_tau", .10)),
        hdr_token_bottleneck=int(getattr(m, "hdr_token_bottleneck", 256)),
        hdr_warmup_steps=0, normal_refine_iterations=1,
        normal_refine_step_limit=float(getattr(m, "normal_refine_step_limit", .05)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
        require_geo_teacher=False, require_hdr_teacher=False,
    )
    state = dict(checkpoint["model"])
    old_prefix = "ldr_event_hdr_aligner.fusion."
    new_prefix = "ldr_event_hdr_aligner."
    if old_prefix + "0.weight" in state and new_prefix + "rgb_context.weight" not in state:
        old = state[old_prefix + "0.weight"]
        split = old.shape[1] // 2
        state[new_prefix + "rgb_context.weight"] = old[:, :split]
        state[new_prefix + "rgb_context.bias"] = state[old_prefix + "0.bias"]
        state[new_prefix + "event_modulation.weight"] = old[:, split:]
        state[new_prefix + "output.weight"] = state[old_prefix + "2.weight"]
        state[new_prefix + "output.bias"] = state[old_prefix + "2.bias"]
    loaded = model.load_state_dict(state, strict=False)
    ignored = ("point_refiner.", "point_fusion_gate.", "token_fusion_gate.",
               "ldr_event_hdr_aligner.fusion.", "ldr_event_hdr_aligner.event_norm.",
               "event_token_projection.bias")
    missing = [key for key in loaded.missing_keys if not key.startswith(ignored)]
    unexpected = [key for key in loaded.unexpected_keys if not key.startswith(ignored)]
    if missing or unexpected:
        raise RuntimeError(f"incompatible V10 state: missing={missing[:8]} unexpected={unexpected[:8]}")
    model._dual_alignment_step = max(model.hdr_warmup_steps + 1, 1)
    return model.to(device), cfg


def configure_real_trainable(model):
    model.requires_grad_(False)
    # Preserve synthetic source attribution and full->geo knowledge. Adapt the
    # representation/fusion strength to the real sensor using metric geometry.
    for module in (
        model.event_encoder, model.event_normal_decoder,
        model.event_token_projection, model.ldr_event_hdr_aligner,
        model.normal_fusion_gate,
        model.normal_depth_refiner,
    ):
        module.requires_grad_(True)
    model.depth_log_scale.requires_grad_(True)
    model.train()
    model.aggregator.eval(); model.depth_head.eval(); model.point_head.eval()
    model.full_geo_aligner.eval(); model.contribution_net.eval()


def move_views(views, device):
    return fe.maybe_denormalize_views([
        {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
         for key, value in view.items()} for view in views
    ])


def stack_output(output, key):
    values = [item.get(key) for item in output.ress]
    if not values or any(value is None for value in values):
        return None
    result = torch.stack(values, 1)
    return result.squeeze(-1) if result.shape[-1] == 1 else result


def loss_for(output, views, args):
    pred = stack_output(output, "depth")
    gt = fe.stack_view_field(views, "depthmap").to(pred)
    intrinsics = fe.stack_view_field(views, "camera_intrinsics").to(pred)
    valid = fe.build_valid_mask(views, gt, depth_min=args.depth_min, depth_max=args.depth_max)
    weight = valid.float()
    depth = ((pred.clamp_min(1e-4).log() - gt.clamp_min(1e-4).log()).abs() * weight).sum() / weight.sum().clamp_min(1)
    pred_n = fe.depth_to_normals(pred, intrinsics)
    gt_n = fe.depth_to_normals(gt, intrinsics)
    nvalid = fe.normal_stencil_valid_mask(valid, pred, eps=args.depth_min)
    nw = nvalid.float()
    cosine = 1.0 - (F.normalize(pred_n, dim=-1, eps=1e-6) * F.normalize(gt_n, dim=-1, eps=1e-6)).sum(-1).clamp(-1, 1)
    normal = (cosine * nw).sum() / nw.sum().clamp_min(1)
    event_n = stack_output(output, "event_normal")
    reliability = stack_output(output, "event_contribution").detach()
    event_support = torch.stack([item["event_normal_support"] for item in output.ress], 1).bool()
    ew = (nvalid & event_support).float() * (.25 + .75 * reliability)
    event_cos = 1.0 - (F.normalize(event_n, dim=-1, eps=1e-6) * F.normalize(gt_n, dim=-1, eps=1e-6)).sum(-1).clamp(-1, 1)
    event_normal = (event_cos * ew).sum() / ew.sum().clamp_min(1)
    total = depth + args.normal_weight * normal + args.event_normal_weight * event_normal
    return total, {"loss": float(total.detach()), "depth": float(depth.detach()),
                   "normal": float(normal.detach()), "event_normal": float(event_normal.detach())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default="/data1/lzh/dataset/DESC/DSEC_EV_VGGT")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--width", type=int, default=518); p.add_argument("--height", type=int, default=392)
    p.add_argument("--event-window-ms", type=float, default=50.0)
    p.add_argument("--train-stride", type=int, default=4); p.add_argument("--test-stride", type=int, default=4)
    p.add_argument("--depth-scale", type=float, default=1.0)
    p.add_argument("--depth-min", type=float, default=.1); p.add_argument("--depth-max", type=float, default=80.)
    p.add_argument("--normal-weight", type=float, default=.25); p.add_argument("--event-normal-weight", type=float, default=.10)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--allow-unaligned-rgb", action="store_true")
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(args.checkpoint)
    model, source_cfg = build_model(checkpoint, device)
    configure_real_trainable(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    train_loader = loader(args, "train", True)
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        totals = {"loss": 0., "depth": 0., "normal": 0., "event_normal": 0.}; count = 0
        for step, views in enumerate(train_loader):
            views = move_views(views, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(views); loss, metrics = loss_for(output, views, args)
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); count += 1
            for key in totals: totals[key] += metrics[key]
            if (step + 1) % 20 == 0: print(f"[DSEC V10] epoch={epoch+1} step={step+1} {metrics}", flush=True)
        record = {"epoch": epoch, **{key: value / max(count, 1) for key, value in totals.items()}}
        history.append(record); print(record, flush=True)
        payload = {"schema": model.checkpoint_schema, "model": model.state_dict(), "cfg": OmegaConf.to_container(source_cfg, resolve=True),
                   "dsec_args": vars(args), "epoch": epoch, "metrics": record, "source_checkpoint": args.checkpoint}
        torch.save(payload, out / "checkpoint-last.pth")
        if record["loss"] < best:
            best = record["loss"]; torch.save(payload, out / "checkpoint-best.pth")
        (out / "metrics_train.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
