# Clean Paper Main-Table Ablation

This folder replaces the old mixed-configuration ablation with five
independently trained checkpoints. Every row starts from `ckpt/model.pt`, uses
training scenes `0..11`, and shares the same optimizer, epochs, frozen
aggregator, trainable camera/depth/point heads, normal loss, and regularizers.

| Row | Event residual | Detail GT | Multi-LDR | Frozen ReliabilityNet |
| --- | ---: | ---: | ---: | ---: |
| M0 matched RGB |  |  |  |  |
| M1 | yes |  |  |  |
| M2 | yes | yes |  |  |
| M3 | yes | yes | yes |  |
| M4 full | yes | yes | yes | yes |

The event residual is causally gated in M1-M4: zero events force zero event
correction. M4 reuses the existing Stage-1 ReliabilityNet in frozen mode; it
does not retrain Stage 1. The historical full-model residual post-filter is
disabled here, so M3 to M4 changes only the reliability module.

## Train on GPUs 2-7

```bash
bash paper_main_ablation/run_train_gpus_234567.sh
```

Outputs are isolated under `abl_event_exp/paper_main_table/`. Each experiment
also saves `ablation_contract.json`, so the active modules can be audited.

## Evaluate unseen scenes 12-15

```bash
GPU=7 bash paper_main_ablation/run_eval_heldout.sh
```

The paper-facing results are written to:

```text
abl_event_exp/paper_main_table/results_heldout_scene12_15/
  summary.csv
  summary.json
  paper_main_table.csv
```

For a quick pipeline smoke test before the full run:

```bash
MAX_BATCHES=2 GPU=7 bash paper_main_ablation/run_eval_heldout.sh
```
