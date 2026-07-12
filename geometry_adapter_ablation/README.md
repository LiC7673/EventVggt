# GeometryAdapter failure ablations

This directory contains four isolated variants. It does not modify the model,
dataset, ContributionNet supervision, or Multi-LDR consistency in the main
pipeline.

## Tensor shapes

- RGB DPT feature at adapter level `l`: `[N*V, C_l, H_l, W_l]`.
- Event feature before DPT resize: `[N*V, 64, 28, 37]` for a `392x518`
  input with patch size 14.
- Contribution/support gate: `[N*V, 1, 28, 37]`.
- Adapter update after resize: identical to the corresponding RGB DPT feature.
- Predicted depth and zero-event difference: `[N, V, H, W]`.

Experiment A changes only the first adapter convolution input from
`C_l + 64` channels to 64 event channels. B and C retain all tensor shapes.
Experiment D retains all shapes but returns exact zero updates at levels 2/3.

## Variants

- `experiment_a`: event-only update prediction (no RGB shortcut).
- `experiment_b`: continuous contribution gating.
- `experiment_c`: continuous contribution gating multiplied by an explicit
  raw-event support pyramid.
- `experiment_d`: event injection only at high-resolution DPT levels 0/1.

Every entry point uses the existing unified A/B/C trainer, including its
checkpoint and qualitative visualization behavior. Additional metrics are
recorded in `metrics.json` and `tensorboard/`:

- `contribution_mean`, `contribution_std`;
- `update_norm`;
- `zero_event_difference`, computed as the final-depth difference to the
  exact RGB-only/zero-event path.

Run one experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m geometry_adapter_ablation.experiment_a.train \
  --pretrained ckpt/model.pt --output exp/adapter_ablation/experiment_a
```

Run all four on separate GPUs:

```bash
GPU_A=0 GPU_B=1 GPU_C=2 GPU_D=3 \
  bash geometry_adapter_ablation/run_all_ablation.sh
```

Outputs are placed under `exp/geometry_adapter_failure_ablation` by default.

