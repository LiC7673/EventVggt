# Linear-voxel paper ablation

Run one corrected full-fusion model:

```bash
GPUS=0 PRETRAINED=ckpt/model.pt \
  bash paired_token_reliability/run_linear_voxel_paper_ablation.sh \
  data.root=/real/data/root
```

The experiment is not named or organized by a correction percentage. The
internal bounded update remains only a numerical safety mechanism. The paper
comparison comes from the same checkpoint and test samples:

- `coarse_rgb`: RGB-only coarse geometry;
- `zero_event`: exact empty-event counterfactual;
- `full_event`: complete method;
- `reverse_time`: reversed temporal bins;
- `swap_polarity`: positive/negative polarity swap.

The evaluator reports AbsRel, delta1, RMSE-log, normal angle, ATE, paired
confidence intervals, causal prediction differences, and depth change from the
coarse result. This isolates whether gains require real event timing/polarity
instead of comparing arbitrary output caps.

Default output: `exp/linear_voxel_paper_ablation/`.
