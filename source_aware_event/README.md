# Source-Aware Event Guidance

This experiment gives `Source-Aware` a literal, testable meaning. It is not a
decorative module.

## Mechanism

Stage 1 receives additive event supervision:

```text
full = geometry_motion + material_reflection + noise
```

Given only `full event + RGB`, the decomposer predicts a three-way source
partition for every polarity, temporal bin, and pixel. It is supervised by the
three aligned branch voxels. The 12 held-out test scenes are never loaded.

Stage 2 freezes this source network. At inference it still receives only the
ordinary full event voxel and RGB. The predicted geometry token drives the
causal depth-detail refiner; predicted material and noise events are rejected,
apart from a small configurable floor. Paired Multi-LDR depth/normal
consistency and GT geometry-detail supervision remain enabled.

## Train on GPUs 4,5,6,7

```bash
bash source_aware_event/run_source_aware_gpus_4567.sh
```

Stage 1 runs on physical GPU 4. Stage 2 runs on GPUs 4,5,6,7. Outputs are under:

```text
abl_event_exp/source_aware_60_12_12/
```

## Final held-out test

```bash
GPU=7 bash source_aware_event/run_source_aware_test.sh
```

## Evidence required before using "Source-Aware" in the title

1. `additive_error` must be close to zero, otherwise branch supervision is not
   temporally aligned.
2. Validation `source_accuracy` and geometry IoU must substantially exceed a
   trivial/uniform partition.
3. Source-aware geometry must outperform direct full-event guidance on the
   same 12 held-out scenes.
4. An oracle `geometry_motion` input should remain an upper-bound row.
5. Report `full event`, scalar reliability, source-aware, and oracle geometry
   under the same split and optimization budget.

If item 3 fails, this remains a negative ablation and should not be promoted as
the paper's core mechanism.

## Compute-matched placebo

The source network can be executed but its prediction replaced with a uniform
one-third partition. This preserves source-module parameters and compute while
removing learned source information:

```bash
OUTPUT_ROOT=abl_event_exp/source_uniform_60_12_12 \
SOURCE_CKPT=abl_event_exp/source_aware_60_12_12/source_net/checkpoint-best.pth \
SOURCE_MODE=uniform bash source_aware_event/run_source_aware_gpus_4567.sh
```

`material_as_geometry` is a stronger negative control. These rows are controls,
not methods to be presented as useful components.
