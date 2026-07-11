# Unified geometry-aware event contribution pipeline

The deployable path is now one model:

`RGB backbone -> temporal ContributionNet -> C*event -> event encoder -> DPT feature adapters -> original depth/point heads`.

The camera head remains RGB-only. There is no raw-depth residual branch. The
contribution tensor has shape `[batch, views, 2B, height, width]`, preserving
temporal bins and polarity. A mass-weighted spatial projection is used only to
gate multi-scale DPT feature updates and for visualization.

## Training phases

- **A / adapter**: force full events, randomly retain 50--100% of active event
  measurements, freeze ContributionNet/RGB/camera, and train the final event
  encoder and geometry adapters.
- **B / contribution**: freeze the complete geometry path and train only
  ContributionNet. Geometry is supervised on all valid pixels; Bridge only
  adds weight. The objective also contains the event-mass budget, paired-LDR
  consistency, and low-contribution update constraint.
- **C / joint (optional)**: one or two short epochs with learning-rate ratios
  `1.0 / 0.1 / 0.03` for ContributionNet, adapters, and geometry heads.

Bridge, the paired reference exposure, GT depth, and GT normals are
training-only. Inference requires only RGB and the event voxel.

## Command

```bash
GPUS=2,3,4,5 EPOCHS_A=5 EPOCHS_B=10 EPOCHS_C=1 \
  bash paired_token_reliability/run_unified_geometry_contribution.sh
```

Useful ablations are `--no-pair-consistency`, `--no-geometry-prior`, and
`--no-budget`. Programmatic inference supports `learned`, `full`, `random`,
`no_contribution` (bypass with all events), and `zero` via
`contribution_override()`.

## Decomposition supervision experiments

Both experiments use scenes 0--11 for training and scenes 12--15 for
validation. They load only the geometry-motion decomposition branch in
addition to the selected main event stream.

```bash
# denominator/model input = cur_best_event; does not load additive full
GPUS=2,3,4,5 bash paired_token_reliability/run_decomp_cur_best_as_full_12train_4test.sh

# denominator/model input = events_additive/full; does not load cur_* events
GPUS=2,3,4,5 bash paired_token_reliability/run_decomp_full_as_event_12train_4test.sh
```

The target is the clipped soft mass ratio `sum(abs(E_geo)) /
(sum(abs(E_input)) + eps)`. It supervises the mass-weighted spatial projection
of temporal contribution using Smooth-L1 during phases B/C only.

## Experiment outputs and automatic evaluation

Launchers save under `exp/<experiment-name>/`. Training and validation save
the first batch of every epoch and then periodic visualizations. After training,
the launcher automatically evaluates `ev_0,ev_1,ev_2,ev_5,ev_10` on the four
held-out scenes. Each exposure directory contains condition metrics, per-batch
metrics, causal diagnostics, and RGB/event/contribution/depth visualizations;
`all_exposures_summary.json` collects the exposure-level results.

The optional geometry-ranking term compares neighboring event-supported valid
pixels. When one pixel has a sufficiently higher GT geometry-detail score, its
predicted spatial contribution must exceed the other by a margin. This is an
ordering constraint only; contribution is never regressed to the geometry map.
Defaults are `weight=0.10`, `margin=0.05`, and geometry difference threshold
`0.10`, active only in phases B/C.

Standalone evaluation:

```bash
CHECKPOINT=exp/my_run/checkpoint-best.pth \
OUTPUT_DIR=exp/my_run/test_all_exposures \
GPU=2 bash paired_token_reliability/run_unified_all_exposures_eval.sh
```
