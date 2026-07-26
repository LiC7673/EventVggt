"""Geo-to-full raw-event injection test for direct-add EventVGGT.

The model and its weights stay fixed.  Every level keeps all geometry events
and injects a deterministic nested subset of material/reflection and noise
events before five-bin linear-time voxelization.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import finetune_event as fe
from ablation import eag3r_metrics_eval as metric
from event_branch_ablation.evaluate_event_contribution import _update_condition
from paired_token_reliability import (
    evaluate_geo_to_full_10level_four_scenes as protocol,
)


CONDITION = "rgb_plus_direct_event"


def build_direct_model(checkpoint, device, _depth_scale):
    raw = metric.torch_load(checkpoint)
    cfg = metric.cfg_from_checkpoint(
        raw, str(protocol.ROOT / "config" / "finetune_event.yaml")
    )
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.model.variant = "base"
    model = fe.build_event_model(cfg)
    result = model.load_state_dict(
        metric.strip_module_prefix(fe.unwrap_state_dict(raw)), strict=True
    )
    print(
        f"[direct geo->full] loaded strict direct-add model: {result}; "
        "no C, alignment, teacher, or refiner",
        flush=True,
    )
    return model.to(device).eval(), cfg


def _normal_rgb(normal, valid):
    image = ((normal.detach().float().cpu() + 1.0) * 0.5).clamp(0, 1)
    return image * valid.detach().float().cpu().unsqueeze(-1)


def save_visual(root, scene, exposure, level, alpha, scale, batch_index,
                views, output, depth_gt, valid, intrinsics, geo, full, mixed,
                every_view):
    import matplotlib.pyplot as plt

    predicted = metric.stack_output(output, "depth").float() * float(scale)
    indices = range(len(views)) if every_view else (0,)
    for view_index in indices:
        pred = predicted[0, view_index]
        gt = depth_gt[0, view_index]
        mask = valid[0, view_index]
        if not mask.any():
            continue
        intr = intrinsics[:, view_index:view_index + 1]
        pred_n = fe.depth_to_normals(pred[None, None], intr)[0, 0]
        gt_n = fe.depth_to_normals(gt[None, None], intr)[0, 0]
        normal_valid = fe.normal_stencil_valid_mask(
            mask[None, None], pred[None, None], eps=1e-6
        )[0, 0]
        rgb = (
            views[view_index]["img"][0].detach().float()
            .permute(1, 2, 0).cpu().clamp(0, 1)
        )
        event_images = [
            value[view_index][0].detach().float().abs().sum(0).cpu()
            for value in (geo, mixed, full)
        ]
        values = torch.cat((pred[mask], gt[mask]))
        vmin, vmax = float(values.min()), float(values.max())
        error = (pred - gt).abs() * mask
        emax = float(torch.quantile(error[mask], 0.995).clamp_min(1e-6))
        panels = (
            (rgb, "LDR RGB", None, None, None),
            (event_images[0], "|E_geo|", "gray", None, None),
            (event_images[1], f"|E_mix| alpha={alpha:.4f}", "gray", None, None),
            (event_images[2], "|E_full|", "gray", None, None),
            (pred.cpu() * mask.cpu(), "direct-event depth", "viridis", vmin, vmax),
            (gt.cpu() * mask.cpu(), "GT depth", "viridis", vmin, vmax),
            (error.cpu(), "|prediction-GT|", "magma", 0, emax),
            (_normal_rgb(pred_n, normal_valid), "pred normal", None, None, None),
            (_normal_rgb(gt_n, normal_valid), "GT normal", None, None, None),
        )
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for axis, panel in zip(axes.reshape(-1), panels):
            image, title, cmap, lo, hi = panel
            shown = axis.imshow(
                image.numpy() if torch.is_tensor(image) else image,
                cmap=cmap, vmin=lo, vmax=hi,
            )
            axis.set_title(title)
            axis.axis("off")
            if cmap is not None:
                fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{scene} | {exposure} | level={level:02d} | "
            f"alpha={alpha:.4f} | fixed scale={scale:.4f}"
        )
        instance = views[view_index].get("instance", f"batch_{batch_index:06d}")
        if isinstance(instance, (list, tuple)):
            instance = instance[0]
        path = (
            root / "visualizations" / protocol._safe_name(scene) / exposure
            / f"{protocol._safe_name(instance)}_b{batch_index:05d}_"
              f"v{view_index:02d}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


@torch.inference_mode()
def evaluate_direct_loader(model, loader, args, device, accumulators, scene,
                           exposure, level_dir, level, alpha, scale):
    batches = selected = total = event_views = 0
    reconstruction_errors = []
    for batch_index, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        views = metric.move_views_to_device(
            fe.maybe_denormalize_views(cpu_views), device
        )
        for view in views:
            selected += int(view["injected_non_geometry_count"].sum().cpu())
            total += int(view["total_non_geometry_count"].sum().cpu())
            error = view["additive_reconstruction_relative_l1"].float()
            finite = error[torch.isfinite(error)]
            reconstruction_errors.extend(finite.cpu().tolist())
            event_views += int(view["injection_alpha"].numel())
        gt = fe.stack_view_field(views, "depthmap").float()
        intrinsics = fe.stack_view_field(views, "camera_intrinsics").float()
        poses = fe.stack_view_field(views, "camera_pose").float()
        valid = fe.build_valid_mask(views, gt, depth_min=1e-6, depth_max=None)
        geo, full, mixed = protocol._prepare_injected_events(views, alpha)
        enabled = args.amp != "none" and device.type == "cuda"
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=enabled
        ):
            output = model(views)
        depth = metric.stack_output(output, "depth").float() * float(scale)
        for accumulator in accumulators[CONDITION]:
            _update_condition(
                accumulator, CONDITION, output, depth, gt,
                intrinsics, poses, valid,
            )
        if args.visualize_every > 0 and batch_index % args.visualize_every == 0:
            save_visual(
                level_dir, scene, exposure, level, alpha, scale, batch_index,
                views, output, gt, valid, intrinsics, geo, full, mixed,
                args.save_every_view,
            )
        batches += 1
    return batches, {
        "selected_non_geometry_events": selected,
        "total_non_geometry_events": total,
        "realized_non_geometry_fraction": selected / max(total, 1),
        "event_views": event_views,
        "additive_reconstruction_relative_l1": (
            float(np.mean(reconstruction_errors))
            if reconstruction_errors else float("nan")
        ),
    }


def _force_fixed_scale(argv):
    result = list(argv)
    if "--geo-depth-scale" not in result:
        result.extend(("--geo-depth-scale", "1.0"))
    if "--full-depth-scale" not in result:
        result.extend(("--full-depth-scale", "1.0"))
    return result


def main():
    protocol.CONDITIONS = (CONDITION,)
    protocol.build_model = build_direct_model
    protocol.evaluate_loader = evaluate_direct_loader
    sys.argv = _force_fixed_scale(sys.argv)
    protocol.main()


if __name__ == "__main__":
    main()
