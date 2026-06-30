# Two-stage Full-event Filtering

This is the staged comparison for the two experiments in
`event_branch_ablation/`. It adds new files only and writes all artifacts to
`abl_event_exp/`.

## Stage 1

Train an additive event decomposer:

```text
full event + RGB
    -> predicted geometry event
    -> predicted material-reflection event
    -> predicted noise event
```

The three synchronized additive branches are supervision only.

## Stage 2

Freeze Stage 1 and train the original `full_img_reliability` detail path using
the predicted geometry event stream:

```text
full + RGB -> frozen Stage 1 -> predicted geometry event
                                  |
                                  v
                   temporal detail + image reliability
```

By default Stage 2 strictly uses:

```text
predicted_geometry
```

The successful high-pass, zero-mean, bounded residual, and internal reliability
floor from `full_img_reliability` remain active. If Stage 1 filters weak detail
too aggressively, retain 20% of full events with `EVENT_FLOOR=0.2`.

## Run

```bash
bash event_filter_two_stage/run_two_stage_gpus_234567.sh
```

Outputs:

```text
abl_event_exp/additive_decomposer_stage1_v2_scene12/
abl_event_exp/two_stage_frozen_geometry_full_img_reliability_v4_stable_scene12/
```

Recommended comparison:

| Experiment | Event stream entering detail refiner |
| --- | --- |
| geometry-motion oracle | GT `geometry_motion` |
| joint decomposition | jointly predicted geometry token |
| two-stage frozen | frozen Stage-1 predicted geometry event |
| full_img_reliability | original full event |
