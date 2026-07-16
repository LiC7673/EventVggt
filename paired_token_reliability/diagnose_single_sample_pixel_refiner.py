"""Overfit one fixed Geo-event sample to diagnose pixel-refiner capacity."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import finetune_event as fe
import real_reliability_stage.evaluate_stage2_heldout as protocol
from ablation.eag3r_metrics_eval import move_views_to_device, stack_output
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import build_model


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--scene", default="Bearded Man_Ceramic_Glazed_White")
    p.add_argument("--exposure", default="ev_2")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--depth-scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def masked_blur(x, valid, kernel=9):
    w = F.avg_pool2d(valid.float(), kernel, 1, kernel // 2)
    y = F.avg_pool2d(x * valid.float(), kernel, 1, kernel // 2)
    return y / w.clamp_min(1e-6)


def configure_input(actual, mode, event_channels):
    value = actual.clone()
    if mode == "derivative_geometry":
        value[:, :event_channels] = 0
    elif mode == "event_geometry":
        value[:, event_channels:event_channels + 6] = 0
    return value


def save_panel(path, base, gt, target, pred, final, valid, title):
    arrays = [base, gt, target, pred, final, (final - gt).abs()]
    names = ["HDR base", "GT", "target HF log residual", "pred HF log residual",
             "refined", "|refined-GT|"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, image, name in zip(axes.flat, arrays, names):
        image = image.detach().float().cpu()
        if name in ("HDR base", "GT", "refined"):
            image = image * valid.detach().cpu()
            cmap = "viridis"
        elif "residual" in name:
            lim = float(image.abs().max().clamp_min(1e-6))
            cmap = "coolwarm"
            shown = ax.imshow(image, cmap=cmap, vmin=-lim, vmax=lim)
            fig.colorbar(shown, ax=ax, fraction=.046); ax.set_title(name); ax.axis("off")
            continue
        else:
            cmap = "magma"
        shown = ax.imshow(image, cmap=cmap)
        fig.colorbar(shown, ax=ax, fraction=.046); ax.set_title(name); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    args = args_parser(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(args.checkpoint, device, args.depth_scale)
    OmegaConf.set_struct(cfg, False); OmegaConf.set_struct(cfg.data, False)
    cfg.data.event_source_mode = "cur_event"
    cfg.data.decomposition_supervision = True
    ns = SimpleNamespace(
        root=args.root, num_views=args.num_views, resolution=[518, 392],
        scene_names=[args.scene], initial_scene_idx=0, active_scene_count=1,
        test_frame_count=120, ldr_event_id=args.exposure,
        event_resize_method="voxel_linear_time", event_resize_bins=5,
        window_stride=1, batch_size=1, num_workers=0, pin_memory=False,
        max_batches=1,
    )
    _, loader = protocol.build_loader(cfg, ns)
    cpu_views = next(iter(loader))
    views = move_views_to_device(fe.maybe_denormalize_views(cpu_views), device)
    for view in views:
        if "geometry_event_voxel" not in view:
            raise RuntimeError("sample has no geometry_event_voxel; decomposition data is required")
        view["event_voxel"] = view["geometry_event_voxel"]

    model.capture_pixel_refiner_inputs = True
    with torch.no_grad():
        output = model(views)
    actual = model._captured_pixel_refiner_actual
    baseline_input = model._captured_pixel_refiner_baseline
    if actual is None:
        raise RuntimeError("pixel-refiner input was not captured")
    base = stack_output(output, "depth_hdr_base").float()
    gt = fe.stack_view_field(views, "depthmap").float()
    valid = (base > 1e-6) & (gt > 1e-6) & torch.isfinite(base) & torch.isfinite(gt)
    bv = base.flatten(0, 1).unsqueeze(1)
    gv = gt.flatten(0, 1).unsqueeze(1)
    vv = valid.flatten(0, 1).unsqueeze(1)
    residual = torch.log(gv.clamp_min(1e-6)) - torch.log(bv.clamp_min(1e-6))
    target = (residual - masked_blur(residual, vv)).detach()
    values = target.abs()[vv]
    threshold = torch.quantile(values, .70) if values.numel() else target.new_tensor(0.)
    strong = vv & (target.abs() >= threshold) & (target.abs() > 1e-4)
    weight = vv.float() * .1 + strong.float() * .9

    event_channels = model.pixel_depth_refiner.stem[0].in_channels - 11
    rows = []
    for mode in ("all", "derivative_geometry", "event_geometry"):
        refiner = copy.deepcopy(model.pixel_depth_refiner).to(device).train()
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=0.)
        x = configure_input(actual, mode, event_channels)
        limit = float(model.pixel_refine_log_limit)
        for step in range(args.steps + 1):
            pred = limit * torch.tanh((refiner(x) - refiner(baseline_input)) / limit)
            error = F.smooth_l1_loss(pred, target, beta=.01, reduction="none")
            loss = (error * weight).sum() / weight.sum().clamp_min(1.)
            if step < args.steps:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step % 10 == 0 or step == args.steps:
                rows.append(dict(mode=mode, step=step, loss=float(loss.detach()),
                                 pred_abs=float(pred.detach().abs()[vv].mean()),
                                 target_abs=float(target.abs()[vv].mean())))
        refined = torch.exp(torch.log(bv.clamp_min(1e-6)) + pred.detach())
        save_panel(out / f"{mode}.png", bv[0, 0], gv[0, 0], target[0, 0],
                   pred[0, 0], refined[0, 0], vv[0, 0], mode)

    with (out / "loss_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = {m: [r for r in rows if r["mode"] == m][-1] for m in
               ("all", "derivative_geometry", "event_geometry")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved single-sample refiner diagnostic to {out}", flush=True)


if __name__ == "__main__":
    main()
