# Paper Module Ablation

This is the corrected paper-facing leave-one-out experiment. All rows use the same fixed
12 training scenes, original VGGT initialization, optimization budget, frozen
backbone, trainable geometry heads, event voxelization, and four held-out test
scenes.

The reliability rows use the intended frozen external ReliabilityUNet. Its
soft output filters the full temporal voxel before temporal-detail refinement.
To prevent the filtered event texture from becoming surface-slope noise, the
Geometry-detail module adds global normal and normal-gradient supervision
derived strictly from GT depth. The rendered-normal field is not used by this
term, and the term is disabled in every row marked `w/o Detail`.

| Row | Event | Detail GT | Paired Multi-LDR | Geometry reliability |
| --- | ---: | ---: | ---: | ---: |
| A0 RGB only |  |  |  |  |
| A1 Direct event | yes |  |  |  |
| A2 w/o Reliability | yes | yes | yes |  |
| A3 w/o Multi-LDR | yes | yes |  | yes |
| A4 w/o Detail | yes |  | yes | yes |
| A5 Full | yes | yes | yes | yes |

`A0` is pure VGGT fine-tuning: no event branch is instantiated. `A1` is an
additional direct-event control. Rows A2-A4 each remove exactly one of the
three paper modules from the full model.

Multi-LDR training uses paired observations of the same scene/window at
`ev_2/ev_5/ev_10`, rather than independent random exposure augmentation.
`ev_1` is held out from Multi-LDR training and included in evaluation as a
harder exposure-generalization condition.

## One-click background run on GPUs 2-7

```bash
bash paper_main_ablation/run_module_ablation_background.sh
```

Before rerunning the complete table, the corrected external-reliability Full
row can be checked independently:

```bash
bash paper_main_ablation/run_full_external_reliability_normal.sh
```

This trains A5 on GPUs 2,3 and then evaluates the four LDR levels in parallel
on GPUs 2,3,4,5.

The command returns immediately and prints a PID and master log path. Monitor
with the printed `tail -f ...` command. Internally, training uses three
two-GPU groups:

```text
2,3    4,5    6,7
```

After all six checkpoints finish, evaluation starts automatically on GPUs
2,3,4,5,6,7. Every model is tested at:

```text
ev_1, ev_2, ev_5, ev_10
```

The same four unseen scenes are used in every job. The scene manifest requires
all four exposures and is shared by training and testing to prevent silent
scene changes across LDR levels.

## Outputs

```text
abl_event_exp/paper_module_ablation_extrel_normal/
  scene_manifest.json
  a0_rgb_only/checkpoint-last.pth
  ...
  a5_full/checkpoint-last.pth
  test_4scenes_ldr_1_2_5_10/
    all_scene_metrics.csv
    mean_metrics_by_model_ldr.csv
    mean_metrics_all_ldrs.csv
    <variant>/ev_<LDR>/per_scene_metrics.csv
    <variant>/ev_<LDR>/visuals/<scene>/*.png
```

Each visualization contains RGB, event bins, GT/predicted depth, log-depth
error, event reliability, and GT/predicted normals. Each scene is recorded as
an independent metric row before four-scene averaging.

Previous `paper_module_ablation*` directories are intentionally left untouched
and must not be mixed with this retraining run.

## Useful overrides

```bash
EPOCHS=30 SKIP_EXISTING=false \
VISUAL_BATCHES=2 NUM_WORKERS=2 \
bash paper_main_ablation/run_module_ablation_background.sh
```

For a quick evaluation smoke test after checkpoints exist:

```bash
MAX_BATCHES=2 bash paper_main_ablation/run_ldr_scene_eval_gpus_234567.sh
```
