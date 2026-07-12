# Depth metric optimized preset

Run:

```bash
GPUS=0,1 PRETRAINED=ckpt/model.pt \
  bash paired_token_reliability/run_linear_voxel_depth_metric_optimized.sh \
  data.root=/real/data/root
```

This preset is deliberately selected for held-out depth and point-map metrics,
not prettier visualization. It disables all normal/event-normal consistency
losses, increases the bounded pixel-depth correction from 3% to 100%, doubles
pixel-branch capacity, expands local event support, and fine-tunes the pretrained
point decoder at 2% of the new pixel branch learning rate.

The separate 15% preset is:

```bash
bash paired_token_reliability/run_linear_voxel_depth_point_15pct.sh data.root=/real/data/root
```

Default output: `exp/linear_voxel_depth_metric_optimized/`.

For a fair paper table, do not tune on the three reported test scenes. Select
the checkpoint/hyperparameters using validation metrics, then run the existing
fixed all-exposure test once. Report RGB-only and zero-event results from the
same evaluator alongside the full-event result.
