"""Controlled source ablation: direct noisy E_full versus clean E_geo normals."""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr_staged_geo as staged
from paired_token_reliability.linear_voxel_dual_alignment_hdr_staged_model import (
    StagedGeoTeacherDualAlignmentModel,
)


SOURCE = os.environ.get("EVENT_NORMAL_SOURCE", "geo").strip().lower()
if SOURCE not in {"full", "geo"}:
    raise RuntimeError("EVENT_NORMAL_SOURCE must be full or geo")


def one_source_schedule(epochs, phase_b, phase_c=0):
    if int(epochs) <= 0 or int(phase_b) or int(phase_c):
        raise ValueError("source ablation requires epochs-a>0 and epochs-b=c=0")
    return ["adapter"] * int(epochs)


def configure_phase(model, phase, _train_heads_a=False):
    if phase != "adapter":
        raise ValueError(f"source ablation only supports adapter, got {phase}")
    model.requires_grad_(False)
    if SOURCE == "full":
        # Deliberately no aligner: noise in E_full is treated as valid evidence.
        model.event_encoder.requires_grad_(True)
        model.event_normal_decoder.requires_grad_(True)
        model.decode_raw_full_normal = True
    else:
        model.geo_event_encoder.requires_grad_(True)
        model.geo_normal_decoder.requires_grad_(True)
    model.train()
    model.aggregator.eval(); model.depth_head.eval(); model.point_head.eval()
    print(
        f"[event-normal ablation] source=E_{SOURCE} direct normal supervision; "
        "full-to-geo alignment disabled", flush=True,
    )


def optimizer_for(model, _phase, args):
    encoder = model.event_encoder if SOURCE == "full" else model.geo_event_encoder
    decoder = model.event_normal_decoder if SOURCE == "full" else model.geo_normal_decoder
    return torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": 2.0 * args.lr},
        {"params": decoder.parameters(), "lr": 5.0 * args.lr},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


class DirectSourceNormalObjective:
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic

    def __call__(self, output, views, *args, **kwargs):
        result = self.diagnostic(output, views, *args, **kwargs)
        normal_key = "event_normal_full_raw" if SOURCE == "full" else "event_normal_geo"
        support_key = "event_normal_support" if SOURCE == "full" else "geo_event_support"
        pred = F.normalize(torch.stack(
            [item[normal_key] for item in output.ress], 1
        ).float(), dim=-1, eps=1e-6)
        gt = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6)
        support = torch.stack([item[support_key] for item in output.ress], 1).bool()
        valid = result.aux["normal_valid_live"].bool() & support
        weight = valid.float()
        cosine = ((1.0 - (pred * gt).sum(-1).clamp(-1, 1)) * weight).sum() / weight.sum().clamp_min(1)
        hf = pred.new_zeros(())
        for kernel in (3, 7):
            error = (v10._hf(pred, kernel) - v10._hf(gt, kernel)).abs().mean(-1)
            hf = hf + (error * weight).sum() / weight.sum().clamp_min(1)
        hf = .5 * hf
        result.loss = 2.0 * cosine + hf
        result.details["event_normal"] = cosine
        result.details["depth_event_normal"] = hf
        result.details["source_normal_cosine"] = cosine
        result.details["source_normal_hf"] = hf
        result.details["loss"] = result.loss
        # Make the common visualizer show the actually supervised branch
        # (raw E_full or clean E_geo), not the unused aligned-full branch.
        result.aux["event_normal_live"] = pred
        result.aux["event_normal_valid_live"] = valid
        return result


def criterion_for(args, _phase):
    return DirectSourceNormalObjective(v10.criterion_for(args, "adapter"))


def main(argv=None):
    pipeline.prepare_pair = v10.prepare_dual_alignment_pair
    pipeline.build_alternating_phase_schedule = one_source_schedule
    pipeline.build_model = staged.build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = v10.save_visual
    pipeline.capture_runtime_state = staged.capture_runtime_state
    pipeline.restore_runtime_state = staged.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = StagedGeoTeacherDualAlignmentModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
