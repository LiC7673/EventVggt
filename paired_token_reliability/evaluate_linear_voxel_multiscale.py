from pathlib import Path
import torch
import finetune_event as fe
from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_multiscale_model import LinearVoxelMultiscalePixelModel


def build_model(checkpoint: Path, _override, device: torch.device):
    raw=torch_load(checkpoint)
    if raw.get("schema") != LinearVoxelMultiscalePixelModel.checkpoint_schema: raise ValueError(raw.get("schema"))
    cfg=cfg_from_checkpoint(raw,None); m=cfg.model
    model=LinearVoxelMultiscalePixelModel(img_size=int(m.img_size),patch_size=int(m.patch_size),embed_dim=int(m.embed_dim),
        head_frames_chunk_size=int(getattr(m,"head_frames_chunk_size",2)),voxel_bins=5,
        pixel_hidden=int(getattr(m,"signed_pixel_hidden",32)),support_dilation_kernel=int(getattr(m,"support_dilation_kernel",5)),
        depth_update_scale=float(getattr(m,"depth_update_scale",.03)),event_decay_tau=float(getattr(m,"event_decay_tau",.003)),
        event_hidden_dim=32,event_pyramid_channels=32,adapter_hidden_channels=64,contribution_channels=32)
    model.load_state_dict(strip_module_prefix(fe.unwrap_state_dict(raw)),strict=True)
    return model.to(device).eval(),cfg


def main():
    import paired_token_reliability.evaluate_unified_all_exposures as driver
    driver.build_model=build_model; driver.main()


if __name__ == "__main__": main()
