"""Independent trainer entry for linear-time polarity voxel grids."""
from paired_token_reliability.linear_voxel_multiscale_model import LinearVoxelMultiscalePixelModel
from paired_token_reliability import train_signed_multiscale as base
from paired_token_reliability import train_unified_geometry_contribution as pipeline
import finetune_event as fe
from paired_token_reliability.common import strip_module_prefix, torch_load


def build_model(cfg, args, device):
    model = LinearVoxelMultiscalePixelModel(img_size=int(cfg.model.img_size), patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim), head_frames_chunk_size=int(getattr(cfg.model,"head_frames_chunk_size",2)),
        voxel_bins=5, pixel_hidden=int(getattr(cfg.model,"signed_pixel_hidden",32)),
        support_dilation_kernel=int(getattr(cfg.model,"support_dilation_kernel",5)),
        depth_update_scale=float(getattr(cfg.model,"depth_update_scale",.03)),
        event_decay_tau=float(getattr(cfg.model,"event_decay_tau",.003)),
        event_hidden_dim=32, event_pyramid_channels=32, adapter_hidden_channels=64,
        contribution_channels=32, contribution_initial_value=.70)
    message=model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(torch_load(args.pretrained))),strict=False)
    required=[k for k in message.missing_keys if k.startswith(("aggregator.","camera_head."))]
    if required: raise RuntimeError(f"base checkpoint missing VGGT weights: {required[:10]}")
    return model.to(device)


class _ContributionObjective:
    """Keep attribution semantics instead of letting task loss open every gate."""
    def __init__(self, criterion, task_weight=0.10):
        self.criterion=criterion; self.task_weight=float(task_weight)
    def __call__(self,*args,**kwargs):
        result=self.criterion(*args,**kwargs); d=result.details
        result.loss=(self.task_weight*d["geometry"]
                     + self.criterion.decomposition_weight*d["decomposition"]
                     + self.criterion.pair_weight*d["pair"]
                     + self.criterion.budget_weight*d["budget"]
                     + self.criterion.geometry_rank_weight*d["geometry_rank"])
        return result


def criterion_for(args,phase):
    criterion=base.criterion_for(args,phase)
    return _ContributionObjective(criterion,task_weight=0.10) if phase=="contribution" else criterion


def main(argv=None):
    pipeline.build_model=build_model; pipeline.configure_phase=base.configure_phase
    pipeline.criterion_for=criterion_for; pipeline.save_visual=base.save_visual
    pipeline.UnifiedGeometryContributionModel=LinearVoxelMultiscalePixelModel
    pipeline.main(argv)


if __name__ == "__main__": main()
