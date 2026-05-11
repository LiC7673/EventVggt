from pathlib import Path

import hydra
from omegaconf import OmegaConf

import finetune_event as fe
from eventvggt.models.streamvggt_global_token import StreamVGGT as GlobalTokenEventStreamVGGT


def configure_global_token_trainable_params(model, cfg) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_tokens = (
        "event_encoder",
        "event_patch_embed",
        "event_token_proj",
        "global_fusion",
    )
    for name, param in model.named_parameters():
        if any(token in name for token in trainable_tokens):
            param.requires_grad = True

    if cfg.train.unfreeze_heads:
        for module in (model.camera_head, model.depth_head, model.point_head, model.track_head):
            for param in module.parameters():
                param.requires_grad = True

    if cfg.train.unfreeze_aggregator_blocks:
        for param in model.aggregator.frame_blocks.parameters():
            param.requires_grad = True
        for param in model.aggregator.global_blocks.parameters():
            param.requires_grad = True


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "config"),
    config_name="finetune_event_global_token.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    fe.EventStreamVGGT = GlobalTokenEventStreamVGGT
    fe.configure_trainable_params = configure_global_token_trainable_params
    fe.train(cfg)


if __name__ == "__main__":
    run()
