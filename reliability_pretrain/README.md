# Geometry-event Reliability Pretraining

This folder implements the first, non-end-to-end version of the new
reliability idea.

## Data Layout

Each scene should contain additive event branches:

```text
scene_xxx/
  LDR/ev_5/*.png
  events_additive/
    geometry_motion/events.h5
    material_reflection/events.h5
    noise/events.h5
    full/events.h5
```

The intended construction is:

```text
full = geometry_motion + material_reflection + noise
```

The dataset voxelizes every branch with the same time window, bins, polarity
split, spatial transform, and resolution, then supervises:

```text
R_geo_gt = abs(V_geometry) / (abs(V_full) + eps)
```

## Stage 1

Train the small U-Net reliability estimator:

```bash
bash reliability_pretrain/run_stage1_reliability_net.sh
```

Input:

```text
event voxel: [B, 2T, H, W]
RGB/LDR:     [B, 3, H, W]
```

Output:

```text
R_geo: [B, 1, H, W]
```

Default uses 5 temporal bins, positive/negative polarity split, so the U-Net
input has 13 channels.

## Planned Ablations

| Setting | Purpose |
| --- | --- |
| `no event` | RGB-only control. |
| `full event` | Direct event usage, vulnerable to reflection/noise. |
| `oracle reliability` | Weight events by `R_geo_gt`, upper bound. |
| `learned reliability` | Weight events by frozen ReliabilityNet. |
| `joint finetune` | Unfreeze ReliabilityNet with a small LR and retain reliability loss. |
