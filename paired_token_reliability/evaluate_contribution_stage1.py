"""Held-out causal evaluation for a Stage-1 event-contribution checkpoint.

Besides the learned map, this evaluates ``C=0``, ``C=1``, a random spatial
permutation with the same active-event values, and equal-count removal of the
highest/lowest scored events.  The latter two are the important causal sanity
check: removing high-score events should hurt more than removing low-score
events.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from paired_token_reliability.common import torch_load
from paired_token_reliability.contribution_dataset import parse_ordered_pairs
from paired_token_reliability.contribution_stage1 import (
    build_model_from_checkpoint,
    contribution_condition,
)
from paired_token_reliability.train_contribution_stage1 import (
    build_frozen_rgb_model,
    frozen_rgb_geometry,
    make_dataset,
    make_loader,
    prepare_pair,
)


BASE_CONDITIONS = ("coarse_rgb", "full_event", "learned", "random_same_mean", "drop_high", "drop_low")


class GeometryAccumulator:
    def __init__(self) -> None:
        self.abs_rel = 0.0
        self.delta_1 = 0.0
        self.log_squared = 0.0
        self.depth_count = 0
        self.normal_degrees = 0.0
        self.normal_11_25 = 0.0
        self.normal_count = 0

    def update(self, depth, normals, depth_gt, normal_gt, valid) -> None:
        valid = valid.bool() & torch.isfinite(depth) & torch.isfinite(depth_gt) & (depth_gt > 1.0e-6)
        predicted = depth.float().clamp_min(1.0e-6)
        target = depth_gt.float().clamp_min(1.0e-6)
        self.abs_rel += float((((predicted - target).abs() / target) * valid).sum())
        ratio = torch.maximum(predicted / target, target / predicted)
        self.delta_1 += float(((ratio < 1.25) & valid).sum())
        self.log_squared += float((((predicted.log() - target.log()).square()) * valid).sum())
        self.depth_count += int(valid.sum())

        normal_valid = valid & (normals.float().norm(dim=-1) > 0.5) & (normal_gt.float().norm(dim=-1) > 0.5)
        cosine = (
            F.normalize(normals.float(), dim=-1, eps=1.0e-6)
            * F.normalize(normal_gt.float(), dim=-1, eps=1.0e-6)
        ).sum(dim=-1).clamp(-1.0, 1.0)
        degrees = torch.rad2deg(torch.acos(cosine))
        self.normal_degrees += float((degrees * normal_valid).sum())
        self.normal_11_25 += float(((degrees < 11.25) & normal_valid).sum())
        self.normal_count += int(normal_valid.sum())

    def result(self):
        return {
            "AbsRel": self.abs_rel / max(self.depth_count, 1),
            "delta1": self.delta_1 / max(self.depth_count, 1),
            "RMSElog": math.sqrt(self.log_squared / max(self.depth_count, 1)),
            "normal_deg": self.normal_degrees / max(self.normal_count, 1),
            "normal_11_25": self.normal_11_25 / max(self.normal_count, 1),
            "depth_pixels": self.depth_count,
            "normal_pixels": self.normal_count,
        }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--rgb-checkpoint", default=None)
    result.add_argument(
        "--no-bridge-checkpoint",
        default=None,
        help="Optional Stage-1 checkpoint trained with --supervision-region event_support",
    )
    result.add_argument("--output", default=None)
    result.add_argument("--device", default="cuda")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--max-batches", type=int, default=100)
    result.add_argument("--drop-fraction", type=float, default=0.20)
    result.add_argument("overrides", nargs="*", help="Optional data.root=... config overrides")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    checkpoint = torch_load(args.checkpoint)
    model = build_model_from_checkpoint(checkpoint)
    no_bridge_model = None
    if args.no_bridge_checkpoint:
        no_bridge_checkpoint = torch_load(args.no_bridge_checkpoint)
        if no_bridge_checkpoint.get("supervision_region") != "event_support":
            raise ValueError(
                "--no-bridge-checkpoint must have supervision_region='event_support'."
            )
        no_bridge_model = build_model_from_checkpoint(no_bridge_checkpoint)
    cfg = OmegaConf.create(checkpoint["cfg"])
    config_overrides = [value for value in args.overrides if "=" in value]
    if config_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(config_overrides))
    training_args = dict(checkpoint.get("training_args", {}))
    bridge_args = argparse.Namespace(
        saturation_threshold=float(training_args.get("saturation_threshold", 0.98)),
        reference_gradient_threshold=float(training_args.get("reference_gradient_threshold", 0.02)),
    )
    pair_strings = checkpoint.get(
        "ordered_pairs", ("ev_1->ev_5", "ev_1->ev_10", "ev_2->ev_10")
    )
    pairs = parse_ordered_pairs(pair_strings)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    rgb_checkpoint = args.rgb_checkpoint or checkpoint.get("frozen_rgb_checkpoint")
    if not rgb_checkpoint or not Path(rgb_checkpoint).is_file():
        raise FileNotFoundError(f"Frozen RGB checkpoint not found: {rgb_checkpoint}")
    rgb_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rgb_model = build_frozen_rgb_model(cfg, rgb_checkpoint, device, rgb_dtype)
    model.to(device).eval()
    if no_bridge_model is not None:
        no_bridge_model.to(device).eval()

    dataset = make_dataset(cfg, "test", pairs)
    if len(dataset) == 0:
        raise RuntimeError("The held-out Stage-1 dataset has no valid samples.")
    loader = make_loader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, train=False
    )
    conditions = list(BASE_CONDITIONS)
    if no_bridge_model is not None:
        conditions.insert(2, "learned_no_bridge")
    accumulators = {condition: GeometryAccumulator() for condition in conditions}
    bridge_accumulators = {condition: GeometryAccumulator() for condition in conditions}
    contribution_sum = 0.0
    contribution_count = 0
    generator = torch.Generator(device=device).manual_seed(12345)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            prepared = prepare_pair(batch, device, bridge_args)
            coarse = frozen_rgb_geometry(rgb_model, prepared["rgb_bad"], prepared["intrinsics"])
            learned = model.contribution_net(
                prepared["event"],
                prepared["rgb_bad"],
                coarse["depth"],
                coarse["normals"],
                coarse["features"],
            )
            learned_no_bridge = None
            if no_bridge_model is not None:
                learned_no_bridge = no_bridge_model.contribution_net(
                    prepared["event"],
                    prepared["rgb_bad"],
                    coarse["depth"],
                    coarse["normals"],
                    coarse["features"],
                )
            active = prepared["event"].abs().sum(dim=2) > 0
            contribution_sum += float(learned[active].sum())
            contribution_count += int(active.sum())
            for condition in conditions:
                current_model = model
                if condition == "learned_no_bridge":
                    selected = learned_no_bridge
                    current_model = no_bridge_model
                else:
                    selected = contribution_condition(
                        condition,
                        learned,
                        prepared["event"],
                        drop_fraction=args.drop_fraction,
                        generator=generator,
                    )
                refined = current_model.event_refiner(
                    selected.unsqueeze(2) * prepared["event"],
                    coarse["depth"],
                    coarse["normals"],
                )
                accumulators[condition].update(
                    refined["depth"],
                    refined["normals"],
                    prepared["depth_gt"],
                    prepared["normal_gt"],
                    prepared["valid_mask"],
                )
                bridge_accumulators[condition].update(
                    refined["depth"],
                    refined["normals"],
                    prepared["depth_gt"],
                    prepared["normal_gt"],
                    prepared["valid_mask"] & prepared["bridge"].bridge,
                )

    results = {
        condition: {
            "all_valid": accumulators[condition].result(),
            "bridge": bridge_accumulators[condition].result(),
        }
        for condition in conditions
    }
    results["contribution_active_mean"] = contribution_sum / max(contribution_count, 1)
    high = results["drop_high"]["bridge"]
    low = results["drop_low"]["bridge"]
    results["causal_check"] = {
        "drop_high_minus_low_RMSElog": high["RMSElog"] - low["RMSElog"],
        "drop_high_minus_low_normal_deg": high["normal_deg"] - low["normal_deg"],
        "passes": bool(
            high["RMSElog"] >= low["RMSElog"]
            or high["normal_deg"] >= low["normal_deg"]
        ),
    }

    print("Held-out Stage-1 contribution evaluation (all valid pixels)")
    print("condition         AbsRel   delta1  RMSElog  normal(deg)  <11.25")
    for condition in conditions:
        metric = results[condition]["all_valid"]
        print(
            f"{condition:16s} {metric['AbsRel']:7.5f}  {metric['delta1']:6.4f}  "
            f"{metric['RMSElog']:7.5f}  {metric['normal_deg']:11.4f}  {metric['normal_11_25']:7.4f}"
        )
    print(f"causal_check={results['causal_check']}")
    output = Path(args.output) if args.output else Path(args.checkpoint).with_name("stage1_causal_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
