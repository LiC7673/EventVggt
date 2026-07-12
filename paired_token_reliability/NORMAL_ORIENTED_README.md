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
- the event branch predicts an absolute unit normal using event features only;
  RGB/coarse normals never enter this decoder, and there is no depth-residual head;
- the event-normal decoder operates at native pixel resolution and never uses
  DPT/ViT patch-grid features; DPT levels 0/1 are used only for geometry refinement;
- normal-gradient, detached depth-normal consistency, update-magnitude, and
  outside-support invariance losses are added.

Run the CPU architecture tests and training with:

```bash
GPU=0 PRETRAINED=ckpt/model.pt \
  bash paired_token_reliability/run_normal_oriented.sh \
  data.root=/data/reflective_raw data.train_scene_count=12 data.test_scene_count=4
```

The launcher mirrors `run_decomp_full_as_event_12train_4test.sh`: it runs DDP
training on scenes 0--11, validation/held-out evaluation on scenes 12--15,
saves `checkpoint-best.pth`, and then evaluates all five exposures. Outputs are
written under `exp/normal_oriented_12train_4test/` by default; metrics are in
`metrics.json` and `test_all_exposures/all_exposures_summary.json`, while logs
are kept in `logs/train.log` and `logs/evaluate_all_exposures.log`.

For sparse events this variant defaults to four temporal bins per polarity
(`EVENT_BINS=4`): 4 positive + 4 negative channels. Training and held-out
evaluation use the same value. Override it only for an explicit ablation.

Useful ablations can be supplied as config overrides:

```bash
model.event_adapter_levels='[0,1,2,3]'
```

`enable_event_depth_residual=true` is deliberately rejected because the first
implementation requested by `toDo.md` keeps that optional path disabled.
