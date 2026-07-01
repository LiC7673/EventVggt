# Additive Event Branch Ablations

These experiments are isolated additions. Existing model and training files
are not modified. All checkpoints, logs, visualizations, and code snapshots
are written under `abl_event_exp/`, not `checkpoints/`.

## 1. Geometry-motion events only

```bash
accelerate launch --multi_gpu --num_processes 2 \
  event_branch_ablation/finetune_geometry_motion_full_reliability.py
```

This reproduces the `full_img_reliability` configuration while replacing the
event source with:

```text
events_additive/geometry_motion/events.h5
```

It tests whether clean diffuse geometry-motion events alone improve depth and
normal detail.

## 2. Full event to additive branch tokens

```bash
accelerate launch --multi_gpu --num_processes 2 \
  event_branch_ablation/finetune_full_to_additive_tokens.py
```

Training labels are synchronized voxels from:

```text
events_additive/geometry_motion/events.h5
events_additive/material_reflection/events.h5
events_additive/noise/events.h5
```

The model input is only `events_additive/full/events.h5 + RGB`. A three-way
softmax partitions every polarity/time channel into predicted geometry,
material, and noise tokens. Only the predicted geometry token enters the
temporal-detail refiner. Test loading intentionally omits all three target
branches, proving inference requires only full events and RGB.

## Parallel run on GPUs 2-7

```bash
bash event_branch_ablation/run_two_ablation_gpus_234567.sh
```

Default outputs:

```text
abl_event_exp/geometry_motion_full_img_reliability_v5_stable_scene12/
abl_event_exp/full_to_additive_tokens_img_reliability_v4_stable_scene12/
```

Both controlled variants initialize from
`checkpoints/ablation_full_img_reliability_scene12/checkpoint-last.pth` and
freeze coarse VGGT/depth/point/camera parameters. Only the event detail branch
and, when present, the additive decomposer are trained. This keeps coarse
geometry fixed so differences measure event-stream quality.

Before retraining, verify that all four branches remain additive after fixed
time-window voxelization:

```bash
python event_branch_ablation/diagnose_additive_alignment.py
```

`mean_additive_relative_l1` should be close to zero. The script also saves
full/geometry/material/noise event panels under
`abl_event_exp/additive_alignment_debug/`.

## Paired held-out event contribution test

Run both trained checkpoints on the same unseen scene:

```bash
bash event_branch_ablation/run_event_contribution_test.sh
```

The default uses scene index 12, after the 12 training scenes, and evaluates
six controlled inputs: `coarse_rgb`, `zero_event`, `full_event`,
`geometry_event`, `material_event`, and `noise_event`. Override the held-out
range when needed:

```bash
INITIAL_SCENE_IDX=12 ACTIVE_SCENE_COUNT=4 \
  bash event_branch_ablation/run_event_contribution_test.sh
```

Each result directory contains `condition_metrics.csv`,
`per_batch_metrics.csv`, and `summary.json`. In the summary,
`full_event_net_gain` is the event-only gain over the same refinement head
with zero events. `refiner_structure_zero_vs_coarse` measures the gain caused
by the extra RGB/depth refinement capacity and must not be attributed to
events.
