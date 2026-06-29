# Staged Reliability Finetuning

This folder connects the independently pretrained `ReliabilityUNet` to the
best temporal-detail StreamVGGT path.

## Stages

1. **Stage 1** learns `R_geo(full event, RGB)` from additive synthetic labels:
   `R_geo_gt = |V_geometry| / (|V_full| + eps)`.
2. **Stage 2** freezes ReliabilityNet and trains the event detail/depth/point
   modules. The model consumes `V_gated = R_geo.detach() * V_full`.
3. **Stage 3** starts from Stage 2, unfreezes ReliabilityNet, uses a lower LR,
   and retains the additive reliability loss.

The temporal and polarity channels are never collapsed by the gate. A single
spatial reliability value weights all channels at the same pixel.

## Run on GPUs 2-7

```bash
bash reliability_staged_finetune/run_stages_123_gpus_234567.sh
```

The launcher reuses Stage 1/2 checkpoints when present. Stage 3 only starts
after Stage 2 successfully writes `checkpoint-last.pth`.

Default outputs:

```text
checkpoints/reliability_net_stage1_scene12/
checkpoints/staged_reliability_stage2_frozen_scene12/
checkpoints/staged_reliability_stage3_joint_scene12/
```

Useful ablation comparison:

| Experiment | ReliabilityNet | Geometry model |
| --- | --- | --- |
| full event | none | full event input |
| Stage 2 | frozen learned gate | trained with gated events |
| Stage 3 | jointly finetuned | trained with gated events |
| geometry oracle | GT geometry event branch | upper bound |

