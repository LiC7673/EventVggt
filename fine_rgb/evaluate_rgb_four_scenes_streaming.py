"""Pure-RGB four-scene evaluation with one scene/exposure loader at a time."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import finetune_no_event as rgb
from fine_rgb.evaluate_rgb_pretrained_vs_finetuned import (
    DEFAULT_SCENES,
    build_model,
    make_loader,
    move_views,
)
from fine_rgb.launcher import normalize_ldr_id
from event_branch_ablation.evaluate_event_contribution import (
    ConditionAccumulator,
    _update_condition,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config/finetune_no_event.yaml")
    parser.add_argument("--pretrained", default="ckpt/model.pt")
    parser.add_argument(
        "--finetuned-template",
        default="checkpoints/fine_rgb_{ldr_event_id}/checkpoint-last.pth",
    )
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--skip-missing-finetuned", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/data1/lzh/dataset/reflective_raw")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--ldr-event-ids", default="0,1,2,5,10")
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--test-frame-count", type=int, default=120)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--visualize-every", type=int, default=1)
    parser.add_argument("--max-visuals-per-condition", type=int, default=0,
                        help="0 saves every selected batch")
    return parser.parse_args()


def _normal_rgb(normal, valid):
    image = ((normal.detach().float().cpu() + 1.0) * .5).clamp(0, 1)
    return image * valid.detach().float().cpu().unsqueeze(-1)


def save_visual(output_root, experiment, scene, exposure, batch_index,
                views, pred, target, intrinsics, valid):
    p, g, mask = pred[0, 0].float(), target[0, 0].float(), valid[0, 0].bool()
    pred_normal = rgb.depth_to_normals(pred.float(), intrinsics.float())[0, 0]
    gt_normal = rgb.depth_to_normals(target.float(), intrinsics.float())[0, 0]
    image = views[0]["img"][0].detach().float().permute(1, 2, 0).cpu().clamp(0, 1)
    values = torch.cat((p[mask], g[mask]))
    vmin, vmax = float(values.min()), float(values.max())
    error = (p - g).abs() * mask
    error_max = float(error.max().clamp_min(1e-6))
    panels = (
        (image, "RGB input", None, None, None),
        (p.cpu() * mask.cpu(), "RGB predicted depth", "viridis", vmin, vmax),
        (g.cpu() * mask.cpu(), "GT depth", "viridis", vmin, vmax),
        (error.cpu(), "|RGB depth - GT depth|", "magma", 0, error_max),
        (_normal_rgb(pred_normal, mask), "RGB depth-derived normal", None, None, None),
        (_normal_rgb(gt_normal, mask), "GT depth-derived normal", None, None, None),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 10)); axes = axes.reshape(-1)
    for axis, (shown_value, title, cmap, lo, hi) in zip(axes, panels):
        shown = axis.imshow(
            shown_value.numpy() if torch.is_tensor(shown_value) else shown_value,
            cmap=cmap, vmin=lo, vmax=hi,
        )
        axis.set_title(title); axis.axis("off")
        if cmap is not None:
            figure.colorbar(shown, ax=axis, fraction=.046, pad=.04)
    figure.suptitle(f"{experiment} | scene={scene} | exposure={exposure}")
    path = (Path(output_root) / "visualizations" / experiment / scene / exposure
            / f"sample_{batch_index:05d}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


@torch.inference_mode()
def evaluate_loader(model, loader, device, args, bundles, *, experiment,
                    scene, exposure):
    enabled = args.amp != "none" and device.type == "cuda"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    evaluated = 0
    visuals = 0
    for batch_index, cpu_views in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        views = move_views(cpu_views, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            output = model(views)
        pred = torch.stack(
            [item["depth"][..., 0] for item in output.ress], dim=1
        ).float()
        target = rgb.stack_view_field(views, "depthmap").float()
        intrinsics = rgb.stack_view_field(views, "camera_intrinsics").float()
        gt_pose = rgb.stack_view_field(views, "camera_pose").float()
        valid = rgb.build_valid_mask(views, target)
        for bundle in bundles:
            _update_condition(
                bundle, "coarse_rgb", output, pred,
                target, intrinsics, gt_pose, valid,
            )
        visual_budget = (args.max_visuals_per_condition <= 0
                         or visuals < args.max_visuals_per_condition)
        if (args.visualize_every > 0 and batch_index % args.visualize_every == 0
                and visual_budget):
            save_visual(args.output_dir, experiment, scene, exposure, batch_index,
                        views, pred, target, intrinsics, valid)
            visuals += 1
        evaluated += 1
        if args.print_freq > 0 and evaluated % args.print_freq == 0:
            print(f"  [stream] batches={evaluated}/{len(loader)}", flush=True)
    return evaluated


def checkpoint_path(template, ldr_event_id):
    return Path(template.format(ldr_event_id=ldr_event_id, ldr=ldr_event_id))


def append_row(rows, experiment, scope, scene, exposure, checkpoint, batches, bundle):
    current = {
        "experiment": experiment,
        "scope": scope,
        "scene": scene,
        "ldr_event_id": exposure,
        "condition": "rgb_only",
        "checkpoint": str(checkpoint),
        "evaluated_batches": batches,
        **bundle.compute(),
    }
    rows.append(current)
    return current


def evaluate_experiment(
    *, name, checkpoint_for_exposure, exposures, args, config_path, device, rows,
):
    per_exposure = {exposure: ConditionAccumulator() for exposure in exposures}
    per_exposure_batches = {exposure: 0 for exposure in exposures}
    overall = ConditionAccumulator()
    overall_batches = 0
    nested = {}

    # The pretrained RGB model is shared by all exposure levels. Fine-tuned
    # RGB has one checkpoint per level, so it is reloaded only when the level
    # changes, never when the scene changes.
    shared_model = None
    if name == "rgb_pretrained_no_finetune":
        shared_checkpoint = checkpoint_for_exposure(exposures[0])
        shared_model, _ = build_model(config_path, shared_checkpoint, device)

    for exposure in exposures:
        checkpoint = checkpoint_for_exposure(exposure)
        if not checkpoint.is_file():
            if name == "rgb_finetuned" and args.skip_missing_finetuned:
                print(f"[skip] missing {name} {exposure}: {checkpoint}", flush=True)
                continue
            raise FileNotFoundError(checkpoint)
        if shared_model is None:
            model, _ = build_model(config_path, checkpoint, device)
        else:
            model = shared_model
        nested[exposure] = {}
        for scene in args.scenes:
            print(f"\n[{name}] [scene={scene}] [exposure={exposure}]", flush=True)
            loader = make_loader(args, scene, exposure)
            local = ConditionAccumulator()
            evaluated = evaluate_loader(
                model, loader, device, args,
                (local, per_exposure[exposure], overall),
                experiment=name, scene=scene, exposure=exposure,
            )
            row = append_row(
                rows, name, "scene", scene, exposure,
                checkpoint, evaluated, local,
            )
            nested[exposure][scene] = row
            per_exposure_batches[exposure] += evaluated
            overall_batches += evaluated
            print(
                f"  MAE={row['mae']:.6f} AbsRel={row['abs_rel']:.6f} "
                f"RMSElog={row['rmse_log']:.6f} "
                f"Nmean={row['normal_mean_deg']:.4f} "
                f"N<11.25={row['normal_11_25']:.4f}",
                flush=True,
            )
            del local, loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        append_row(
            rows, name, "all_scenes_pixel_weighted", "ALL", exposure,
            checkpoint, per_exposure_batches[exposure], per_exposure[exposure],
        )
        if shared_model is None:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    overall_row = append_row(
        rows, name, "all_scenes_exposures_pixel_weighted", "ALL", "ALL",
        checkpoint_for_exposure(exposures[0]), overall_batches, overall,
    )
    if shared_model is not None:
        del shared_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"results": nested, "overall_pixel_weighted": overall_row}


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    config_path = Path(args.base_config)
    exposures = [
        normalize_ldr_id(item.strip())
        for item in args.ldr_event_ids.split(",") if item.strip()
    ]
    exposures = list(dict.fromkeys(exposures))
    if not exposures:
        raise ValueError("--ldr-event-ids is empty")
    pretrained = Path(args.pretrained)
    if not args.skip_pretrained and not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, summary_results = [], {}

    if not args.skip_pretrained:
        summary_results["rgb_pretrained_no_finetune"] = evaluate_experiment(
            name="rgb_pretrained_no_finetune",
            checkpoint_for_exposure=lambda _exposure: pretrained,
            exposures=exposures, args=args, config_path=config_path,
            device=device, rows=rows,
        )
    if not args.skip_finetuned:
        summary_results["rgb_finetuned"] = evaluate_experiment(
            name="rgb_finetuned",
            checkpoint_for_exposure=lambda exposure: checkpoint_path(
                args.finetuned_template, exposure
            ),
            exposures=exposures, args=args, config_path=config_path,
            device=device, rows=rows,
        )

    payload = {
        "streaming_unit": "one scene and one RGB exposure",
        "scenes": list(args.scenes),
        "ldr_event_ids": exposures,
        "results": summary_results,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved pure-RGB streaming metrics to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
