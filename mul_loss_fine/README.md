# Mul-Loss Ablations

These scripts test the idea in `mul_loss.md`: events are not used as dense
geometry labels. Instead, cross-view event co-support is used as a confidence
weight for multi-view geometry/detail consistency.

| Script | Extra loss |
| --- | --- |
| `finetune_mul_loss_baseline.py` | Original `finetune_event.py` loss only |
| `finetune_mul_loss_mv_normal.py` | Event-supported cross-view normal consistency |
| `finetune_mul_loss_mv_presence.py` | Event-supported detail presence margin |
| `finetune_mul_loss_mv_hf.py` | Event-supported high-frequency normal consistency |
| `finetune_mul_loss_mv_presence_hf.py` | Recommended detail setup: presence + high-frequency, no full-normal consistency |
| `finetune_mul_loss_mv_normal_hf.py` | Normal consistency + high-frequency consistency |
| `finetune_mul_loss_mv_all.py` | Normal + presence + high-frequency |
| `finetune_mul_loss_mv_all_orient.py` | All losses + small detail-orientation term |
| `finetune_mul_loss_detail_gt.py` | GT-normal/detail weighted high-frequency supervision |
| `finetune_mul_loss_detail_gt_uniform.py` | Depth-derived GT-detail control without event reweighting |
| `finetune_mul_loss_detail_gt_selective_event.py` | Same depth-derived GT detail, boosted only at top-20% temporal/polarity event support |
| `finetune_mul_loss_detail_gt_temporal_bins.py` | Same uniform GT detail, with correctly fused polarity-preserving temporal-bin event tokens |
| `finetune_mul_loss_detail_gt_temporal_detail.py` | Recommended event-detail variant: temporal/polarity voxel CNN predicts dense bounded log-depth residual without patch-token grid injection |
| `finetune_mul_loss_detail_gt_temporal_gated.py` | Highlight-ripple resistant variant: events provide a low-pass gate, while RGB/coarse depth proposes residual geometry |
| `finetune_mul_loss_detail_gt_temporal_gated_multildr.py` | Exposure-invariant gate: paired LDR training matches stable RGB/event cues; evaluation remains single LDR |
| `finetune_mul_loss_detail_gt_temporal_reliability_v2.py` | Reliability V2: temporal event statistics gate GT-taught depth corrections and multi-LDR consistency |
| `finetune_mul_loss_detail_gt_geo_event_teacher.py` | High-exposure teacher learns geometry-contributing events; LDR students inherit a reliability-filtered event residual proposal |
| `finetune_mul_loss_detail_gt_temporal_adapter.py` | Initialize from uniform, freeze RGB/heads, and train only temporal event tokens for incremental detail gain |
| `finetune_mul_loss_detail_gt_salient.py` | Strong GT detail supervision focused on salient high-frequency geometry |
| `finetune_mul_loss_mv_all_detail_gt.py` | Cross-view event losses + GT detail supervision |

Run one script on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  mul_loss_fine/finetune_mul_loss_mv_all.py num_workers=0 pin_mem=false
```

Run all ablations on eight GPUs, two GPUs per script:

```bash
bash mul_loss_fine/run_mul_loss_2gpu_8gpu.sh
```

Common useful overrides:

```bash
bash mul_loss_fine/run_mul_loss_2gpu_8gpu.sh \
  data.root=/data1/lzh/dataset/reflective_raw \
  data.num_views=6 \
  +loss.mv_max_pairs=3
```

The default projection pose is GT pose for stability. To use predicted pose for
the multi-view projection:

```bash
+loss.mv_projection_pose=pred
```

If the model improves scalar metrics but still produces over-smoothed normals,
start from:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  mul_loss_fine/finetune_mul_loss_detail_gt.py num_workers=0 pin_mem=false
```

This uses GT depth/normal-derived detail as the target; events only boost the
weight in co-supported areas.

To verify whether events improve the difficult detail regions, first run the
strict pair below. Both scripts derive normals from GT depth, use the same
GT-detail weights, and disable
multi-view terms; only the second one boosts the strongest temporal/polarity
event support pixels:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  mul_loss_fine/finetune_mul_loss_detail_gt_uniform.py num_workers=0 pin_mem=false

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 \
  mul_loss_fine/finetune_mul_loss_detail_gt_selective_event.py num_workers=0 pin_mem=false
```

Run the same pair concurrently on physical GPUs 5-8 (CUDA device IDs
`4,5` and `6,7`):

```bash
bash mul_loss_fine/run_detail_gt_event_pair_2gpu_5678.sh
```

Important diagnostic: the historical `base` event model builds event features,
but its token fusion checks `[B,S,P,C]` tensors using the wrong axis and an
incompatible token channel width. As a result, `detail_gt_uniform` is a useful
RGB + GT-detail control, not proof that event input helped.

To test event input itself, compare that control against the fixed temporal-bin
event model. Both use identical GT-detail losses:

```bash
bash mul_loss_fine/run_temporal_bins_compare_2gpu_5678.sh \
  data.root=/data1/lzh/dataset/reflective_raw data.num_views=4

bash finetune_vaild/run_temporal_bins_compare_gpus_5678.sh
```

The `temporal_bins` token-fusion experiment can expose patch-grid patterns in
depth-derived normals because the same patch-resolution residual enters DPT
features directly. Dense event-written residuals can instead reproduce moving
highlight trails as contour-like ripples. For final fine-detail training,
prefer the gated variant:

```bash
bash mul_loss_fine/run_temporal_gated_detail_2gpu.sh
bash finetune_vaild/run_event_counterfactual_gpus_5678.sh
```

To make the gated branch robust to reflective appearance changes across
exposure, train its exposure-invariant extension. It loads the same frame and
event voxel at two LDR levels, learns coarse RGB/event agreement, and teaches
the agreement gate with GT-supported geometry detail. Test inference still
uses one LDR sequence:

```bash
bash mul_loss_fine/run_temporal_gated_multildr_2gpu.sh \
  data.root=/data1/lzh/dataset/reflective_raw

bash finetune_vaild/run_temporal_gated_multildr_counterfactual.sh
```

The default initialization is
`checkpoints/mul_loss_detail_gt_temporal_gated/checkpoint-last.pth`. Common
overrides are `LDR_TRAIN_IDS=ev_2,ev_5,ev_10`, `EVAL_LDR_ID=ev_5`, and
`NUM_VIEWS=4`.

For the event-dependence experiment, the V2 gate observes temporal
persistence, polarity mixture and temporal entropy. The final depth residual
is multiplied by this event gate, so zeroed events remove the V2 correction.
The command below trains on two GPUs and automatically runs
`real/zero/reverse_time/swap_polarity` validation afterward:

```bash
bash mul_loss_fine/run_temporal_reliability_v2_2gpu.sh \
  data.root=/data1/lzh/dataset/reflective_raw
```

During training, `ldr_pair_count` must be greater than zero in train logs,
while `residual_target_loss` and `gate_reliability_loss` confirm that the new
adapter receives supervision. After training, check
`event_output_sensitivity_detected=true` in the generated counterfactual
`summary.json`; a useful event path should also perform better with `real`
events than with `zero` or temporally reversed events. The one-click runner
exits with an error when the counterfactual test cannot detect event influence.

To train the reliability gate to keep only events that explain GT geometry,
use the high-exposure teacher setup. The batch always contains a teacher LDR
level, by default `ev_10`, plus LDR students such as `ev_2/ev_5`; test still
uses one LDR sequence:

```bash
bash mul_loss_fine/run_geo_event_teacher_2gpu.sh \
  data.root=/data1/lzh/dataset/reflective_raw
```

Useful overrides are `GEO_TEACHER_LDR_ID=ev_10`,
`GEO_STUDENT_LDR_IDS=ev_2,ev_5`, `EVAL_LDR_ID=ev_5`, and `NUM_VIEWS=4`.
Watch `geo_event_reliability_pos_mean` become larger than
`geo_event_reliability_neg_mean`; that gap means non-geometric event support is
being rejected before it reaches the LDR residual. Also check
`geo_event_delta_abs` and `geo_event_delta_loss`; if `geo_event_delta_abs` stays
near zero, the event branch is still not writing geometry.

For the strictest event-contribution check, train only the temporal event
adapter on top of the converged uniform checkpoint:

```bash
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes 2 \
  mul_loss_fine/finetune_mul_loss_detail_gt_temporal_adapter.py \
  data.root=/data1/lzh/dataset/reflective_raw data.num_views=4
```

Multi-LDR training from `mul_ldr.md`:

```bash
bash mul_loss_fine/run_mul_ldr_2gpu.sh
```

Useful overrides:

```bash
LDR_TRAIN_IDS=ev_2,ev_5,ev_10 \
EVAL_LDR_ID=ev_10 \
EXPOSURES_PER_SAMPLE=2 \
GPU_LIST=0,1 \
bash mul_loss_fine/run_mul_ldr_2gpu.sh \
  data.root=/data1/lzh/dataset/reflective_raw
```

The test-set visualizations are saved under:

```text
checkpoints/<exp_name>/test_vis/step_xxxxxxx/
```
