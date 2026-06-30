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
abl_event_exp/geometry_motion_full_img_reliability_scene12/
abl_event_exp/full_to_additive_tokens_img_reliability_scene12/
```

