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
| `finetune_mul_loss_mv_normal_hf.py` | Normal consistency + high-frequency consistency |
| `finetune_mul_loss_mv_all.py` | Normal + presence + high-frequency |
| `finetune_mul_loss_mv_all_orient.py` | All losses + small detail-orientation term |
| `finetune_mul_loss_detail_gt.py` | GT-normal/detail weighted high-frequency supervision |
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
