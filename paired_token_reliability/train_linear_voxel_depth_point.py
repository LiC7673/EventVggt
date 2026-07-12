"""Depth/point-only trainer for the linear-time voxel model."""
import torch
from paired_token_reliability import train_linear_voxel_multiscale as base
from paired_token_reliability import train_signed_multiscale as shared
from paired_token_reliability import train_unified_geometry_contribution as pipeline
from paired_token_reliability.linear_voxel_multiscale_model import LinearVoxelMultiscalePixelModel


def configure_phase(model, phase, _train_heads_a=False):
    model.requires_grad_(False)
    if phase == "adapter":
        model.event_encoder.requires_grad_(True)
        model.depth_local_head.requires_grad_(True)
        model.depth_head.requires_grad_(True)
        model.depth_head.geometry_adapters.requires_grad_(False)
        model.point_head.requires_grad_(True)
        # These historical modules are bypassed because point_head is called
        # without an event pyramid; leaving them trainable breaks DDP.
        model.point_head.geometry_adapters.requires_grad_(False)
    elif phase == "contribution":
        model.contribution_net.requires_grad_(True)
    elif phase == "joint":
        model.contribution_net.requires_grad_(True)
        model.event_encoder.requires_grad_(True)
        model.depth_local_head.requires_grad_(True)
        model.depth_head.requires_grad_(True)
        model.depth_head.geometry_adapters.requires_grad_(False)
        model.point_head.requires_grad_(True)
        model.point_head.geometry_adapters.requires_grad_(False)
    else:
        raise ValueError(phase)
    model.train()
    model.aggregator.eval(); model.camera_head.eval()
    if not any(p.requires_grad for p in model.point_head.parameters()): model.point_head.eval()


def optimizer_for(model, phase, args):
    pretrained_ids={id(p) for module in (model.depth_head,model.point_head) for p in module.parameters()}
    pretrained, primary=[],[]
    for parameter in model.parameters():
        if not parameter.requires_grad: continue
        (pretrained if id(parameter) in pretrained_ids else primary).append(parameter)
    groups=[]
    if primary: groups.append({"params":primary,"lr":args.lr})
    # Keep pretrained point decoding stable; it receives the same GT point loss
    # but a much smaller LR than newly initialized pixel modules.
    if pretrained: groups.append({"params":pretrained,"lr":args.lr*0.02})
    return torch.optim.AdamW(groups,weight_decay=args.weight_decay,betas=(0.9,0.95))


def main(argv=None):
    pipeline.build_model=base.build_model
    pipeline.configure_phase=configure_phase
    pipeline.optimizer_for=optimizer_for
    pipeline.criterion_for=shared.criterion_for
    pipeline.save_visual=shared.save_visual
    pipeline.UnifiedGeometryContributionModel=LinearVoxelMultiscalePixelModel
    pipeline.main(argv)


if __name__ == "__main__": main()
