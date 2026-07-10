# Stage 2 — Contribution-Guided Geometry Adapters

This directory is independent of the earlier residual and direct-token-fusion
experiments. It consumes the new Stage-1
`multi_ldr_event_contribution_v1` checkpoint.

Data flow:

```text
RGB -> frozen VGGT aggregator -------------------------> RGB camera head
                  |                                       (pose unchanged)
                  +-> coarse D/N/F -> frozen ContributionNet -> C

C * event voxel -> polarity/temporal encoder -> four-scale event pyramid
                                                       |
VGGT DPT projected features at layers 4/11/17/23 ------+-> adapters
                                                       |
                          F_rgb_l + tanh(alpha_l) C_l A_l
                                                       |
                                      original DPT depth/point decoder
```

There is no raw depth residual, log-depth correction, residual clipping, or
event pose branch. At initialization every `alpha_l` is exactly zero, so depth,
point, and pose equal the pretrained RGB model. Contribution is applied both
to the input voxel and to every DPT scale.

Run on physical GPUs 2–7:

```bash
STAGE1_CKPT=abl_event_exp/event_contribution_stage1/checkpoint-best.pth \
bash stage2_geometry_adapter/run_gpu2_7.sh \
  data.root=/data/reflective_raw
```

The script runs:

1. CPU adapter invariance tests;
2. Phase A: only event encoder and geometry adapters;
3. Phase B: low-LR DPT heads and the last two aggregator blocks. Set
   `RUN_PHASE_B=0` to stop after Phase A or `TRAIN_CONTRIBUTION_B=true` to
   update ContributionNet with its smaller LR multiplier.

Outputs default to:

```text
abl_event_exp/stage2_geometry_adapter/geometry_adapter_stage2_a/
abl_event_exp/stage2_geometry_adapter/geometry_adapter_stage2_b/
```

Held-out event counterfactual evaluation:

```bash
CUDA_VISIBLE_DEVICES=2 python -m stage2_geometry_adapter.evaluate \
  --checkpoint abl_event_exp/stage2_geometry_adapter/geometry_adapter_stage2_b/checkpoint-last.pth \
  --reliability-checkpoint "$STAGE1_CKPT" \
  --output-dir abl_event_exp/stage2_geometry_adapter/eval_b
```

The inherited evaluator option is still named `--reliability-checkpoint` for
CLI compatibility; here it must point to the new Stage-1 contribution
checkpoint, not a legacy ReliabilityUNet.
