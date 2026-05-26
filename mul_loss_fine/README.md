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
