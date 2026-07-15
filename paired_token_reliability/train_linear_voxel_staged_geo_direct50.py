"""A: direct E_geo geometry, B: E_full alignment+C, C: optional joint tune."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_signed_multiscale as pixel_base
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_staged_geo_direct50_model import (
    StagedGeoDirect50Model,
)


def sequential_schedule(geo_epochs, full_epochs, joint_epochs=0):
    if int(geo_epochs) <= 0 or int(full_epochs) <= 0:
        raise ValueError("staged direct route requires epochs-a>0 and epochs-b>0")
    return (["adapter"] * int(geo_epochs)
            + ["contribution"] * int(full_epochs)
            + ["joint"] * int(joint_epochs))


def build_model(cfg, args, device):
    m = cfg.model
    model = StagedGeoDirect50Model(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 5)),
        depth_update_scale=.50,
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in own and own[key].shape == value.shape
    }
    loaded = model.load_state_dict(compatible, strict=False)
    required = [key for key in loaded.missing_keys if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print("[staged direct50] E_geo direct pixel geometry -> frozen decoder -> E_full alignment", flush=True)
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.set_stage("geo")
        model.event_encoder.backbone.requires_grad_(True)
        model.event_normal_decoder.requires_grad_(True)
        model.depth_local_head.requires_grad_(True)
        label = "A E_geo: encoder+normal+direct depth head"
    elif phase == "contribution":
        model.set_stage("full")
        model.event_encoder.aligner.requires_grad_(True)
        model.event_encoder.aligner.reliability.requires_grad_(False)
        model.contribution_net.requires_grad_(True)
        label = "B E_full: full-to-geo aligner+Contribution; geometry decoder frozen"
    elif phase == "joint":
        model.set_stage("joint")
        model.event_encoder.backbone.requires_grad_(True)
        model.event_encoder.aligner.requires_grad_(True)
        model.event_encoder.aligner.reliability.requires_grad_(False)
        model.event_normal_decoder.requires_grad_(True)
        model.depth_local_head.requires_grad_(True)
        model.contribution_net.requires_grad_(True)
        label = "C joint: low-rate geometry+alignment+Contribution"
    else:
        raise ValueError(phase)
    model.train(); model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(f"[staged direct50/{phase}] {label}", flush=True)


def optimizer_for(model, phase, args):
    groups = []
    if phase == "adapter":
        groups = [
            {"params": model.event_encoder.backbone.parameters(), "lr": 2.0 * args.lr},
            {"params": model.event_normal_decoder.parameters(), "lr": 5.0 * args.lr},
            {"params": model.depth_local_head.parameters(), "lr": 5.0 * args.lr},
        ]
    elif phase == "contribution":
        groups = [
            {"params": model.event_encoder.aligner.parameters(), "lr": 3.0 * args.lr},
            {"params": model.contribution_net.parameters(), "lr": 3.0 * args.lr},
        ]
    else:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        groups = [{"params": parameters, "lr": .5 * args.lr}]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def normal_derivative(normal):
    dx = torch.zeros_like(normal); dy = torch.zeros_like(normal)
    dx[:, :, :, :-1] = normal[:, :, :, 1:] - normal[:, :, :, :-1]
    dy[:, :, :-1] = normal[:, :, 1:] - normal[:, :, :-1]
    return torch.stack((dx, dy), -2)


class StagedObjective:
    def __init__(self, base, phase):
        self.base, self.phase = base, phase

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        pred_normal = torch.stack([item["event_normal"] for item in output.ress], 1).float()
        gt_normal = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6)
        pred_d = normal_derivative(F.normalize(pred_normal, dim=-1, eps=1e-6))
        gt_d = normal_derivative(gt_normal)
        valid = result.aux["normal_valid_live"].bool()
        support = torch.stack([item["full_geo_alignment_support"] for item in output.ress], 1).bool()
        mask = (valid & support).unsqueeze(-1).unsqueeze(-1).float()
        hf = (F.smooth_l1_loss(pred_d, gt_d, beta=.02, reduction="none") * mask).sum()
        hf = hf / (mask.sum().clamp_min(1) * 6.0)
        align = torch.stack([item["full_geo_feature_error"] for item in output.ress], 1).float()
        align = (align * (valid & support).float()).sum() / (valid & support).float().sum().clamp_min(1)
        if self.phase == "adapter":
            result.loss = result.loss + .50 * hf
        elif self.phase == "contribution":
            result.loss = result.loss + 1.0 * align + .10 * hf
        else:
            result.loss = result.loss + .50 * align + .25 * hf
        result.details["event_normal_hf_aux"] = hf
        result.details["full_geo_alignment"] = align
        result.details["loss"] = result.loss
        return result


def criterion_for(args, phase):
    return StagedObjective(pixel_base.criterion_for(args, phase), phase)


def main(argv=None):
    pipeline.build_alternating_phase_schedule = sequential_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = pixel_base.save_visual
    pipeline.UnifiedGeometryContributionModel = StagedGeoDirect50Model
    pipeline.main(argv)


if __name__ == "__main__":
    main()
