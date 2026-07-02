# Frozen Reliability Stage 2

This experiment keeps the standalone ReliabilityNet frozen and uses it inside
VGGT temporal-detail finetuning.

## Data split

- Train: scene indices `0..11`, all frames.
- Held-out prediction/test: scene indices `12..15`, all frames.
- The launcher aborts if scene names overlap.

## Forward path

```text
full event voxel + RGB
        |
frozen ReliabilityNet
        |
R_geo, G = 0.1 + 0.9 R_geo
        |
G * full event voxel
        |
temporal-detail refiner + VGGT depth/point heads
```

Event-aware losses receive `R_geo.detach() * event_voxel`. ReliabilityNet is
therefore fixed in Stage 2, while the detail refiner and depth/point heads are
optimized.

## One command

Use an existing Stage-1 checkpoint when available:

```bash
bash real_reliability_stage/run_stage2_vggt_12train_4test.sh
```

Run Stage 1 automatically if its checkpoint is missing, then run Stage 2:

```bash
bash real_reliability_stage/run_full_two_stage_12train_4test.sh
```

Override GPUs or split indices through environment variables:

```bash
GPUS=2,3 TEST_INITIAL_SCENE_IDX=12 \
bash real_reliability_stage/run_stage2_vggt_12train_4test.sh
```

Outputs are written under
`abl_event_exp/stage2_reliability_residual_train12_test4/`.

## Causal event-only residual experiment

This variant enforces `zero event -> zero detail residual`, preventing the
refiner from improving geometry using RGB/coarse depth alone:

```bash
bash real_reliability_stage/run_stage2_causal_event_train_and_eval.sh
```

Training output:
`abl_event_exp/stage2_causal_event_reliability_train12_test4/`

The automatic held-out evaluation compares coarse RGB, zero event, full event,
reverse-time event, and swapped-polarity event on scene indices `12..15`.
