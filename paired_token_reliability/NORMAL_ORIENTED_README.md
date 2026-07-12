# Normal-oriented event refinement

This additive variant implements the architecture requested by the root
`toDo.md`; none of the historical scripts are overwritten.

Key differences from `unified_model.py` / `stage2_geometry_adapter/model.py`:

- the event encoder consumes the full event voxel (never `C * V`);
- contribution is applied once, as a continuous final update gate;
- every event adapter is event-only and cannot inspect RGB features;
- event features are explicitly masked by a locally dilated pixel support;
- only DPT levels 0 and 1 are enabled by default (the two local/high-resolution
  projected maps in the current DPT ordering);
- the primary event output is a bounded delta-normal added to the RGB coarse
  normal; there is no independent depth-residual head;
- normal-gradient, detached depth-normal consistency, update-magnitude, and
  outside-support invariance losses are added.

Run the CPU architecture tests and training with:

```bash
GPU=0 PRETRAINED=ckpt/model.pt \
  bash paired_token_reliability/run_normal_oriented.sh \
  data.root=/data/reflective_raw data.train_scene_count=12 data.test_scene_count=4
```

Useful ablations can be supplied as config overrides:

```bash
model.event_adapter_levels='[0,1,2,3]' model.normal_update_scale=0.10
```

`enable_event_depth_residual=true` is deliberately rejected because the first
implementation requested by `toDo.md` keeps that optional path disabled.
