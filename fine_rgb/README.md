# Pure RGB LDR Ablations

This folder contains pure RGB/no-event finetuning scripts for checking how each
LDR exposure level affects training. The dataset variant here does not load
event slices or event voxels.

Detect available LDR levels:

```bash
python fine_rgb/detect_ldr_levels.py \
  --root /data1/lzh/dataset/reflective_raw \
  --active-scene-count 3 \
  --format lines
```

Run one fixed LDR level:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  fine_rgb/finetune_rgb_ldr.py \
  data.root=/data1/lzh/dataset/reflective_raw \
  data.ldr_event_id=ev_5 \
  exp_name=fine_rgb_ev_5 \
  num_workers=0 pin_mem=false
```

Detect all common LDR levels and train each one, two GPUs per job:

```bash
DATA_ROOT=/data1/lzh/dataset/reflective_raw \
GPU_LIST=0,1,2,3,4,5,6,7 \
bash fine_rgb/run_all_rgb_ldr_2gpu_8gpu.sh
```

To bypass auto-detection:

```bash
LDR_LIST=ev_2,ev_5,ev_10 bash fine_rgb/run_all_rgb_ldr_2gpu_8gpu.sh
```
