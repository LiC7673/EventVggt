"""Train the final HDR-base + event/coarse-geometry pixel-refiner model."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import finetune_event as fe

from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr as v10
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr_derivative as derivative
from paired_token_reliability import train_linear_voxel_dual_alignment_hdr_pixel_hf as pixel_hf
from paired_token_reliability.common import strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_dual_alignment_hdr_final_pixel_refiner_model import (
    FinalEventGeometryPixelRefinerModel,
)


def build_model(cfg, args, device):
    m = cfg.model
    model = FinalEventGeometryPixelRefinerModel(
        img_size=int(m.img_size), patch_size=int(m.patch_size), embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m, "head_frames_chunk_size", 2)),
        voxel_bins=5, pixel_hidden=int(getattr(m, "signed_pixel_hidden", 32)),
        event_count_cmax=float(getattr(m, "event_count_cmax", 3.0)),
        pixel_refiner_hidden=int(getattr(m, "pixel_refiner_hidden", 64)),
        pixel_refine_log_limit=float(getattr(m, "pixel_refine_log_limit", .20)),
        support_dilation_kernel=int(getattr(m, "support_dilation_kernel", 3)),
        depth_update_scale=float(getattr(m, "depth_update_scale", .50)),
        event_decay_tau=float(getattr(m, "event_decay_tau", .0015)),
        depth_log_scale_limit=float(getattr(m, "depth_log_scale_limit", 2.0)),
        alignment_confidence_tau=float(getattr(m, "alignment_confidence_tau", .10)),
        hdr_token_bottleneck=int(getattr(m, "hdr_token_bottleneck", 256)),
        hdr_warmup_steps=int(getattr(m, "hdr_warmup_steps", 1000)),
        normal_refine_iterations=1,
        normal_refine_step_limit=float(getattr(m, "normal_refine_step_limit", .05)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70,
    )
    state = strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained)))
    own = model.state_dict()
    new_prefixes = ("event_normal_decoder.", "pixel_depth_refiner.")
    compatible = {
        key: value for key, value in state.items()
        if key in own and own[key].shape == value.shape
        and not key.startswith(new_prefixes)
    }
    loaded = model.load_state_dict(compatible, strict=False)
    required = [key for key in loaded.missing_keys if key.startswith(("aggregator.", "camera_head."))]
    if required:
        raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    print(
        "[FINAL pixel refiner] HDR base + (coarse geometry,event feature,dN,C)->pixel log-depth; RGB excluded",
        flush=True,
    )
    return model.to(device)


def configure_phase(model, phase, train_heads_a=False):
    derivative.configure_phase(model, phase, train_heads_a)
    model.pixel_depth_refiner.requires_grad_(True)
    if not any(parameter.requires_grad for parameter in model.pixel_depth_refiner.parameters()):
        raise RuntimeError("pixel_depth_refiner is frozen")
    print("[FINAL trainable] pixel_depth_refiner=ON; direct RGB input=NONE", flush=True)


def optimizer_for(model, _phase, args):
    scale_id = id(model.depth_log_scale)
    encoder_ids = {id(parameter) for parameter in model.event_encoder.parameters()}
    fast_modules = (
        model.event_normal_decoder, model.full_geo_aligner, model.contribution_net,
        model.event_token_projection, model.ldr_event_hdr_aligner,
        model.normal_fusion_gate, model.pixel_depth_refiner,
    )
    fast_ids = {id(parameter) for module in fast_modules for parameter in module.parameters()}
    regular, encoder, fast, scale = [], [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) == scale_id:
            scale.append(parameter)
        elif id(parameter) in fast_ids:
            fast.append(parameter)
        elif id(parameter) in encoder_ids:
            encoder.append(parameter)
        else:
            regular.append(parameter)
    groups = []
    if regular: groups.append({"params": regular, "lr": args.lr})
    if encoder: groups.append({"params": encoder, "lr": 2.0 * args.lr})
    if fast: groups.append({"params": fast, "lr": 5.0 * args.lr})
    if scale: groups.append({"params": scale, "lr": 10.0 * args.lr, "weight_decay": 0.0})
    if not fast:
        raise RuntimeError("final pixel-refiner optimizer has no fast parameters")
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))


class FinalRefinerObjective:
    def __init__(self, base):
        self.base = base

    def __call__(self, output, views, *args, **kwargs):
        result = self.base(output, views, *args, **kwargs)
        update = torch.stack([item["depth_pixel_update"] for item in output.ress], 1).float()
        ratio = torch.stack([item["depth_delta_ratio"] for item in output.ress], 1).float()
        recency = torch.stack([item["event_detail_recency"] for item in output.ress], 1).float()
        coupling = output.ress[0]["pixel_refiner_coupling"].float()
        result.details["pixel_refiner_update_abs"] = update.abs().mean()
        result.details["pixel_refiner_ratio_abs"] = ratio.abs().mean()
        result.details["pixel_refiner_ratio_p95"] = torch.quantile(ratio.detach().abs().flatten(), .95)
        result.details["event_recency_mean"] = recency.mean()
        result.details["pixel_refiner_coupling"] = coupling
        result.details["loss"] = result.loss
        return result


def criterion_for(args, phase):
    return FinalRefinerObjective(pixel_hf.criterion_for(args, phase))


def save_visual(output_root, phase, epoch, batch_index, views, reference_views,
                event, bridge, output, aux):
    derivative.save_visual(
        output_root, phase, epoch, batch_index, views, reference_views,
        event, bridge, output, aux,
    )
    item = output.ress[0]
    panels = (
        (item["event_detail_recency"][0].detach().float().cpu(), "recent-event gate"),
        (item["pixel_refiner_raw_update"][0].detach().float().cpu(), "raw pixel log-depth"),
        (item["pixel_refiner_bounded_update"][0].detach().float().cpu(), "bounded pixel log-depth"),
        (item["depth_pixel_update"][0].detach().float().cpu(), "applied pixel depth update"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for axis, (image, title) in zip(axes, panels):
        shown = axis.imshow(image.numpy(), cmap="coolwarm")
        axis.set_title(title); axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=.046, pad=.04)
    path = Path(output_root) / "visualizations" / phase / f"epoch_{epoch+1:03d}" / f"batch_{batch_index+1:06d}_final_pixel_refiner.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(path, dpi=130); plt.close(figure)


def main(argv=None):
    pipeline.prepare_pair = v10.prepare_dual_alignment_pair
    pipeline.build_alternating_phase_schedule = v10.one_stage_schedule
    pipeline.build_model = build_model
    pipeline.configure_phase = configure_phase
    pipeline.optimizer_for = optimizer_for
    pipeline.criterion_for = criterion_for
    pipeline.save_visual = save_visual
    pipeline.capture_runtime_state = v10.capture_runtime_state
    pipeline.restore_runtime_state = v10.restore_runtime_state
    pipeline.UnifiedGeometryContributionModel = FinalEventGeometryPixelRefinerModel
    pipeline.main(argv)


if __name__ == "__main__":
    main()
