from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import finetune_event as fe
from ablation.eag3r_metrics_eval import cfg_from_checkpoint, strip_module_prefix, torch_load
from paired_token_reliability.linear_voxel_multiscale_model import LinearVoxelMultiscalePixelModel
import real_reliability_stage.evaluate_stage2_heldout as protocol


_base_visual = protocol.save_full_event_visuals


def save_weight_visuals(args, views, output, depth, depth_gt, valid, batch_idx):
    _base_visual(args, views, output, depth, depth_gt, valid, batch_idx)
    if not getattr(args, "visualize_all", False): return
    every=int(getattr(args,"visualize_every",1))
    if every>1 and batch_idx%every: return
    root=Path(args.output_dir)/"visualizations"
    bins=views[0]["event_voxel"].shape[1]//2
    for sample_idx in range(depth.shape[0]):
        voxel=views[0]["event_voxel"][sample_idx].detach().float().cpu()
        pos,neg=voxel[:bins].sum(0),voxel[bins:].sum(0)
        item=output.ress[0]
        contribution=item["event_contribution_spatial"][sample_idx].detach().float().cpu()
        weights=item["temporal_decay_weights"][sample_idx].detach().float().cpu()
        update=item["depth_pixel_update"][sample_idx].detach().float().cpu()
        panels=((pos,"positive voxel","magma"),(neg,"negative voxel","magma"),
                (pos-neg,"signed projection","coolwarm"),(pos+neg,"event mass","gray"),
                (contribution,"contribution","magma"),(update,"pixel depth update","coolwarm"))
        fig,axes=plt.subplots(2,4,figsize=(20,10)); axes=np.asarray(axes).reshape(-1)
        for axis in axes: axis.axis("off")
        for axis,(image,title,cmap) in zip(axes,panels):
            array=image.numpy(); limit=max(float(np.abs(array).max()),1e-8)
            if cmap=="coolwarm": axis.imshow(array,cmap=cmap,vmin=-limit,vmax=limit)
            else: axis.imshow(array,cmap=cmap)
            axis.set_title(title)
        axes[6].axis("on"); axes[6].bar(np.arange(len(weights)),weights.numpy())
        axes[6].set_ylim(0,1); axes[6].set_title("temporal decay weights"); axes[6].set_xlabel("bin: old → new")
        rgb=views[0]["img"][sample_idx].detach().float().permute(1,2,0).cpu().numpy().clip(0,1)
        axes[7].imshow(rgb); axes[7].set_title("single RGB input")
        instance=views[0].get("instance",f"batch_{batch_idx:06d}")
        if isinstance(instance,(list,tuple)): instance=instance[sample_idx]
        safe=str(instance).replace("/","_").replace("\\","_").replace(" ","_")
        fig.suptitle("weights="+", ".join(f"{float(x):.3f}" for x in weights))
        fig.tight_layout(); fig.savefig(root/f"{safe}_b{batch_idx:06d}_weights.png",dpi=130); plt.close(fig)


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
    protocol.save_full_event_visuals=save_weight_visuals
    driver.build_model=build_model; driver.main()


if __name__ == "__main__": main()
