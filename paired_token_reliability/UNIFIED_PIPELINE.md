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
GPU=2 EPOCHS_A=5 EPOCHS_B=10 EPOCHS_C=1 \
  bash paired_token_reliability/run_unified_geometry_contribution.sh
```

Useful ablations are `--no-pair-consistency`, `--no-geometry-prior`, and
`--no-budget`. Programmatic inference supports `learned`, `full`, `random`,
`no_contribution` (bypass with all events), and `zero` via
`contribution_override()`.
