"""Two stages only: E_geo geometry/normal, then delayed E_full confidence C."""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_signed_multiscale as pixel_base
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_geo_normal_then_full_c_model import (
    GeoNormalThenFullCModel,
)


def two_stage_schedule(geo_epochs, full_epochs, joint_epochs=0):
    if int(geo_epochs) <= 0 or int(full_epochs) <= 0 or int(joint_epochs) != 0:
        raise ValueError("this route requires A>0, B>0 and C=0; there is no third stage")
    return ["adapter"] * int(geo_epochs) + ["contribution"] * int(full_epochs)


def build_model(cfg, args, device):
    m = cfg.model
    model = GeoNormalThenFullCModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 5)),
        depth_update_scale=.50,
        event_decay_tau=float(getattr(m, "event_decay_tau", .003)),
        c_delay_steps=int(getattr(m, "c_delay_steps", 1000)),
        c_transition_steps=int(getattr(m, "c_transition_steps", 1000)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    compatible = {key: value for key, value in state.items()
                  if key in own and own[key].shape == value.shape}
    loaded = model.load_state_dict(compatible, strict=False)
    required = [key for key in loaded.missing_keys if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print("[two-stage v2] A=E_geo normal+direct50 depth; B=E_full delayed C only", flush=True)
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.set_stage("geo")
        model.event_encoder.requires_grad_(True)
        model.event_normal_decoder.requires_grad_(True)
        model.depth_local_head.requires_grad_(True)
        label = "E_geo, C=1, train normal/HF/direct pixel depth"
    elif phase == "contribution":
        model.set_stage("full")
        model.contribution_net.learned.requires_grad_(True)
        label = "E_full, geometry frozen, C delayed 1k then ramps for 1k"
    else:
        raise ValueError(f"no third stage in this route: {phase}")
    model.train(); model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(f"[two-stage/{phase}] {label}", flush=True)


def optimizer_for(model, phase, args):
    if phase == "adapter":
        groups = [
            {"params": model.event_encoder.parameters(), "lr": 2.0 * args.lr},
            {"params": model.event_normal_decoder.parameters(), "lr": 5.0 * args.lr},
            {"params": model.depth_local_head.parameters(), "lr": 5.0 * args.lr},
        ]
    else:
        groups = [{"params": model.contribution_net.learned.parameters(), "lr": 3.0 * args.lr}]
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


def normal_derivative(normal):
    dx = torch.zeros_like(normal); dy = torch.zeros_like(normal)
    dx[:, :, :, :-1] = normal[:, :, :, 1:] - normal[:, :, :, :-1]
    dy[:, :, :-1] = normal[:, :, 1:] - normal[:, :, :-1]
    return torch.stack((dx, dy), -2)


class NormalFirstObjective:
    def __init__(self, base, phase):
        self.base, self.phase = base, phase

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        pred = F.normalize(torch.stack(
            [item["event_normal"] for item in output.ress], 1
        ).float(), dim=-1, eps=1e-6)
        gt = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6)
        pred_d, gt_d = normal_derivative(pred), normal_derivative(gt)
        valid = result.aux["normal_valid_live"].bool()
        mask = valid.unsqueeze(-1).unsqueeze(-1).float()
        hf = (F.smooth_l1_loss(pred_d, gt_d, beta=.02, reduction="none") * mask).sum()
        hf = hf / (mask.sum().clamp_min(1) * 6.0)
        ramp = output.ress[0]["contribution_learning_ramp"].float()
        # Phase A strengthens normal high frequencies. Phase B has no normal
        # gradient: its only trainable object is C, whose influence itself is
        # already multiplied by the same deployment ramp in the model.
        if self.phase == "adapter":
            result.loss = result.loss + .50 * hf
        result.details["event_normal_hf_aux"] = hf
        result.details["contribution_learning_ramp"] = ramp
        result.details["loss"] = result.loss
        step = int(output.ress[0]["contribution_full_step"].item())
        if (self.phase == "contribution" and torch.is_grad_enabled()
                and step % 100 == 0 and int(os.environ.get("RANK", "0")) == 0):
            predicted = torch.stack(
                [item["predicted_full_contribution"] for item in output.ress], 1
            ).float()
            deployed = torch.stack(
                [item["event_contribution"] for item in output.ress], 1
            ).float()
            print(
                f"[full-C@{step:05d}] ramp={float(ramp):.3f} "
                f"Cpred={float(predicted.mean().detach()):.4f} "
                f"Cdeploy={float(deployed.mean().detach()):.4f}", flush=True,
            )
        return result


def criterion_for(args, phase):
    return NormalFirstObjective(pixel_base.criterion_for(args, phase), phase)


def capture_runtime_state(model):
    return {"full_c_step": int(model.contribution_net.full_step.item())}


def restore_runtime_state(model, state):
    model.contribution_net.full_step.fill_(int(state.get("full_c_step", 0)))


def main(argv=None):
    pipeline.build_alternating_phase_schedule = two_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = pixel_base.save_visual
    pipeline.capture_runtime_state = capture_runtime_state
    pipeline.restore_runtime_state = restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = GeoNormalThenFullCModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
