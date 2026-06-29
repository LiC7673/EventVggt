# Paper Ablation Scripts

This folder collects the paper-facing ablations for the current best route:
multi-LDR exposure invariance, GT geometry-detail supervision, and
image-guided event reliability.

## Train

Run the default ablations in parallel.  By default the runner uses GPU groups
`1,4`, `5,6`, and `7`, skips checkpoints that already have
`checkpoint-last.pth`, and excludes the completed `rgb_baseline` variant:

```bash
bash ablation/run_paper_ablation_2gpu.sh
```

Useful environment variables:

```bash
GPU_GROUPS="1,4 5,6 7" EPOCHS=20 NUM_VIEWS=4 LDR_ID=ev_5 bash ablation/run_paper_ablation_2gpu.sh
VARIANTS=rgb_baseline,raw_event,full_img_reliability bash ablation/run_paper_ablation_2gpu.sh
```

`GPU_GROUPS` is space-separated.  A group like `1,4` launches one two-GPU DDP
job, while a single-card group like `7` launches a one-GPU job with
`SINGLE_GPU_ACCUM_ITER=2` by default to keep the effective batch closer to the
two-GPU jobs.  Logs are written to `ablation_logs/paper_ablation_parallel_*`.

The ablation runner disables in-training test evaluation by default
(`EVAL_EVERY_STEPS=0`, `SKIP_FINAL_EVAL=true`) to avoid long DDP waits.  Run
the EAG3R-style metric script after training for paper numbers.

When launching `finetune_paper_ablation.py` manually, use Hydra's append
syntax for the variant field:

```bash
accelerate launch --multi_gpu --num_processes 2 ablation/finetune_paper_ablation.py +ablation_variant=full_img_reliability exp_name=ablation_full_img_reliability
```

Available variants:

| Variant | Purpose |
| --- | --- |
| `rgb_baseline` | RGB-only VGGT finetune. |
| `rgb_detail_gt` | RGB-only with GT high-frequency geometry-detail supervision. |
| `raw_event` | Event StreamVGGT without extra detail/reliability losses. |
| `raw_event_detail_gt` | Raw event branch plus GT geometry-detail supervision. |
| `multildr` | Multi-LDR exposure sampling without extra detail/reliability losses. |
| `multildr_detail_gt` | Multi-LDR sampling plus GT geometry-detail supervision. |
| `full_img_reliability` | Current best: temporal-detail residual, high-pass/zero-mean constraints, and image-guided event reliability. |

## Evaluate

Evaluate the default manifest with EAG3R-style metrics:

```bash
python ablation/eag3r_metrics_eval.py --manifest ablation/eag3r_eval_manifest.json --device cuda:0
```

Evaluate on held-out scenes that were not used by the default ablation training
(`initial_scene_idx=0`, `active_scene_count=3`).  This starts from scene index
3 by default and evaluates all frames from those scenes:

```bash
bash ablation/run_eag3r_metrics_heldout_scenes.sh
```

Useful held-out overrides:

```bash
GPU=7 HELDOUT_INITIAL_SCENE_IDX=3 HELDOUT_ACTIVE_SCENE_COUNT=3 SPLIT=all bash ablation/run_eag3r_metrics_heldout_scenes.sh
SPLIT=test bash ablation/run_eag3r_metrics_heldout_scenes.sh
```

Main reported metrics follow the EAG3R table style:

- Depth: `AbsRel`, `delta1`, `RMSElog`
- Pose: `ATE`, `RPE_t`, `RPE_r`

The evaluator also reports normal/detail diagnostics for our geometry-detail
story: normal angular error, high-event/low-event normal error, and event-detail
correlation.

## Combined Best Route

To combine the global-depth advantage of `multildr_detail_gt` with the normal
detail advantage of `full_img_reliability`, run a frozen-coarse second stage:

```bash
bash mul_loss_fine/run_multildr_detail_then_reliability_2gpu.sh
```

This initializes from
`checkpoints/ablation_multildr_detail_gt/checkpoint-last.pth`, freezes the
VGGT backbone and all coarse heads, and trains only the temporal event detail
and reliability branch. Scale, low-frequency, and non-detail no-regression
constraints prevent the local residual from damaging the coarse prediction.

After training, rerun:

```bash
bash ablation/run_eag3r_metrics_heldout_scenes.sh
```

The intended success condition is that held-out `AbsRel` and `RMSElog` remain
close to `multildr_detail_gt`, while normal angular error approaches or
improves upon `full_img_reliability`.

## Twelve-scene Paper Run

The paper-scale split uses 12 training scenes and 6 entirely held-out scenes.
GPUs 2-7 are divided into three two-GPU groups:

```bash
bash ablation/run_scene12_paper_ablation_gpus_234567.sh
```

Phase 1 trains `rgb_baseline`, `rgb_detail_gt`, `raw_event_detail_gt`,
`multildr`, `multildr_detail_gt`, and `full_img_reliability`, all with
identical scene selection. Phase 2 automatically trains the frozen-coarse
combined model from the resulting `multildr_detail_gt_scene12` checkpoint.

Evaluate scene indices 12-17, which are disjoint from training indices 0-11:

```bash
GPU=7 bash ablation/run_scene12_heldout_eval.sh
```

Results are saved under `ablation/results/scene12_heldout6`.

## Staged Reliability Ablations

The independently supervised reliability route adds two paper ablations:

```bash
bash reliability_staged_finetune/run_stages_123_gpus_234567.sh
```

- `staged_reliability_stage2_frozen_scene12`: freezes the Stage-1
  ReliabilityNet and trains geometry/detail heads with
  `R_geo.detach() * V_full`.
- `staged_reliability_stage3_joint_scene12`: resumes Stage 2, unfreezes the
  ReliabilityNet with a lower LR, and retains the additive reliability loss.

Both checkpoints are included in `eag3r_eval_manifest_scene12.json` for the
same held-out-scene depth, pose, and normal evaluation.
