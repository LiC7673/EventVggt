"""One Geo warm-up epoch, then Full-C/Geo alternation with recall-first HF loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import finetune_event as fe

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr_final_pixel_refiner as final_base
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr_derivative as derivative
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_alternating_detail_first_model import (
    AlternatingDetailFirstModel,
)


def schedule(geo_warmup, cycles, joint_epochs=0):
    if int(geo_warmup) != 1 or int(cycles) <= 0 or int(joint_epochs) != 0:
        raise ValueError("use --epochs-a 1, --epochs-b <cycles>, --epochs-c 0")
    result = ["adapter"]
    for _ in range(int(cycles)):
        result.extend(("contribution", "adapter"))
    return result


def prepare_pair(batch, device, args, phase):
    # Adapter really consumes E_geo; contribution really consumes E_full.
    target, reference, event, bridge = pipeline._ORIGINAL_PREPARE_PAIR(
        batch, device, args, phase
    ) if hasattr(pipeline, "_ORIGINAL_PREPARE_PAIR") else pipeline.prepare_pair(
        batch, device, args, phase
    )
    for student, teacher in zip(target, reference):
        student["hdr_img"] = teacher["img"]
        student["event_source_label"] = "E_geo" if phase == "adapter" else "E_full"
    return target, reference, event, bridge


def build_model(cfg, args, device):
    m = cfg.model
    model = AlternatingDetailFirstModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        event_count_cmax=float(getattr(m, "event_count_cmax", 3.0)),
        pixel_refiner_hidden=int(getattr(m, "pixel_refiner_hidden", 64)),
        pixel_refine_log_limit=float(getattr(m, "pixel_refine_log_limit", .30)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .0015)),
        alignment_confidence_tau=.10, hdr_token_bottleneck=256,
        hdr_warmup_steps=0, normal_refine_iterations=1, normal_refine_step_limit=.05,
        c_delay_steps=int(getattr(m, "c_delay_steps", 1000)),
        c_transition_steps=int(getattr(m, "c_transition_steps", 1000)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    loaded = model.load_state_dict(compatible, strict=False)
    required = [k for k in loaded.missing_keys if k.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"missing frozen VGGT weights: {required[:10]}")
    print("[detail-first] Geo x1, then (Full-C,Geo) alternating; false-negative HF >> noise", flush=True)
    return model.to(device)


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.set_confidence_stage("geo")
        for module in (model.event_encoder, model.event_normal_decoder,
                       model.event_token_projection, model.ldr_event_hdr_aligner,
                       model.pixel_depth_refiner):
            module.requires_grad_(True)
        label = "E_geo; C_fusion=C_refine=1; geometry/HF train"
    elif phase == "contribution":
        model.set_confidence_stage("full")
        model.contribution_net.learned.requires_grad_(True)
        model.normal_fusion_gate.learned.requires_grad_(True)
        label = "E_full; geometry frozen; train both delayed C gates"
    else:
        raise ValueError(phase)
    model.train(); model.aggregator.eval(); model.camera_head.eval()
    model.depth_head.eval(); model.point_head.eval()
    print(f"[detail-first/{phase}] {label}", flush=True)


def optimizer_for(model, phase, args):
    if phase == "adapter":
        modules = (model.event_encoder, model.event_normal_decoder,
                   model.event_token_projection, model.ldr_event_hdr_aligner,
                   model.pixel_depth_refiner)
    else:
        modules = (model.contribution_net.learned, model.normal_fusion_gate.learned)
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=(3 if phase == "adapter" else 2) * args.lr,
                             weight_decay=args.weight_decay, betas=(.9, .95))


def normal_derivative(normal):
    dx, dy = torch.zeros_like(normal), torch.zeros_like(normal)
    dx[:, :, :, :-1] = normal[:, :, :, 1:] - normal[:, :, :, :-1]
    dy[:, :, :-1] = normal[:, :, 1:] - normal[:, :, :-1]
    return torch.stack((dx, dy), -2)


class RecallFirstObjective:
    def __init__(self, base, phase): self.base, self.phase = base, phase

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        pred = torch.stack([x["event_normal_derivative_full"] for x in output.ress], 1).float()
        gt_n = F.normalize(result.aux["normal_gt_live"].float().detach(), dim=-1, eps=1e-6)
        target = normal_derivative(gt_n)
        valid = result.aux["normal_valid_live"].bool()
        pm, tm = pred.norm(dim=-1), target.norm(dim=-1)
        valid2 = valid.unsqueeze(-1).expand_as(tm)
        values = tm[valid2]
        threshold = torch.quantile(values.detach(), .70) if values.numel() else tm.new_tensor(0.)
        strong = valid2 & (tm >= threshold) & (tm > 1.e-4)
        weak = valid2 & ~strong
        vector = F.smooth_l1_loss(pred, target, beta=.01, reduction="none").mean(-1)
        strong_loss = (vector * strong).sum() / strong.sum().clamp_min(1)
        weak_loss = (vector * weak).sum() / weak.sum().clamp_min(1)
        # Missing GT detail is four times more expensive than extra response.
        miss = (F.relu(tm - pm) * strong).sum() / strong.sum().clamp_min(1)
        extra = (F.relu(pm - tm) * weak).sum() / weak.sum().clamp_min(1)
        hf = strong_loss + .10 * weak_loss + 4.0 * miss + .05 * extra
        if self.phase == "adapter":
            result.loss = result.loss + 2.0 * hf
        result.details["recall_first_hf"] = hf
        result.details["hf_pred_gt_ratio"] = (pm[strong].mean() / tm[strong].mean().clamp_min(1e-6)) if strong.any() else pm.new_zeros(())
        result.details["c_fusion_ramp"] = output.ress[0]["c_fusion_ramp"]
        result.details["c_refine_ramp"] = output.ress[0]["c_refine_ramp"]
        result.details["loss"] = result.loss
        return result


def criterion_for(args, phase):
    # update_weight=0 is supplied by the launcher: do not reward smooth/zero detail.
    return RecallFirstObjective(final_base.criterion_for(args, phase), phase)


def capture_runtime_state(model):
    state = v10.capture_runtime_state(model)
    state.update(c_fusion_step=int(model.contribution_net.full_step.item()),
                 c_refine_step=int(model.normal_fusion_gate.full_step.item()))
    return state


def restore_runtime_state(model, state):
    v10.restore_runtime_state(model, state)
    model.contribution_net.full_step.fill_(int(state.get("c_fusion_step", 0)))
    model.normal_fusion_gate.full_step.fill_(int(state.get("c_refine_step", 0)))


def main(argv=None):
    # Preserve a stable handle before replacing the pipeline hook.
    pipeline._ORIGINAL_PREPARE_PAIR = pipeline.prepare_pair
    pipeline.prepare_pair = prepare_pair
    pipeline.build_alternating_phase_schedule = schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = final_base.save_visual
    pipeline.capture_runtime_state = capture_runtime_state
    pipeline.restore_runtime_state = restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = AlternatingDetailFirstModel
    pipeline.main(argv)


if __name__ == "__main__": main()
