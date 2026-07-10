# Fast Stage-1/Stage-2 Ablation Validation

This directory implements the five conditions in `toDo_abl.md` without
replacing the canonical Stage-1 or Stage-2 experiments.

## Conditions

| Method | Stage-1 contribution | Stage-2 event input |
|---|---|---|
| `rgb_only` | unused | `C=0` |
| `raw_event` | unused | `C=1` |
| `ours` | Multi-LDR bridge A/B training | `C_geo * V` |
| `no_multildr` | one target exposure, event-support geometry loss | `C_single * V` |
| `saturation_mask` | unused | `M_sat * V` |

All Stage-2 event conditions use the same event encoder and geometry adapters.
Only the source of `C` changes. The RGB-only result uses the frozen base VGGT
checkpoint and an exact zero contribution map.

The split is scene-disjoint by construction:

- train scene indices: `0..19`;
- test scene indices: `20..24`;
- views within a scene are never randomly split;
- evaluation loads only the target exposure and its event voxel.

For `no_multildr`, target-exposure frequencies and optimizer-step counts match
`ours` exactly; the paired reference image is simply never loaded. This avoids
confounding Multi-LDR supervision with additional exposure augmentation or
training compute.

## Run

```bash
GPU=2 GPUS=2,3 \
DATA_ROOT=/data1/lzh/dataset/reflective_raw \
bash ab_st1_st2/run_ablation.sh
```

Fast defaults are Stage-1 `3 + 5` epochs and Stage-2 `5` epochs. They can be
changed uniformly:

```bash
EPOCHS_PROXY=3 EPOCHS_CONTRIBUTION=5 EPOCHS_STAGE2=5 \
bash ab_st1_st2/run_ablation.sh
```

Resume only evaluation with existing checkpoints:

```bash
RUN_STAGE1=0 RUN_STAGE2=0 RUN_EVAL=1 \
bash ab_st1_st2/run_ablation.sh
```

## Outputs

Everything is saved under `experiments/ablation/`:

```text
rgb_only/
raw_event/
ours/
no_multildr/
saturation_mask/
ablation_results.csv
analysis.json
```

Each method contains its checkpoint or checkpoint link, logs,
`evaluation.json`, and fixed-scene visualizations for `ev_0`, `ev_1`, `ev_2`,
`ev_5`, and `ev_10`. Evaluation always reports `C mean/std/min/max`; maps with
`C_std <= 1e-3` are explicitly marked as collapsed. Constant maps are expected
for the controlled `rgb_only` and `raw_event` baselines but not for learned
contribution maps.
