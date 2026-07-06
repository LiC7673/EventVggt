# DSEC val-to-test experiment

This integration follows the official DSEC event slicing, timestamp offset,
rectification, and disparity encoding. It uses every scene under `val/` for
training and every scene under `test/` only for final evaluation.

## Extract downloaded archives

First audit the existing downloads and redownload missing/corrupt files:

```bash
bash dsec_exp/repair_dsec_downloads.sh \
  /data1/lzh/dataset/DESC/DSEC_EV_VGGT
```

Add `EXTRACT=1` to repair and then extract in one run. The repair script keeps
the existing val/test directory assignment and also downloads event-camera
disparity (`disparity_event`) required for event-frame geometry supervision.

Recursively extract every archive into a sibling subdirectory:

```bash
bash dsec_exp/extract_archives_recursive.sh \
  /data1/lzh/dataset/DESC/DSEC_EV_VGGT
```

The archives are retained. Completed outputs have an extraction marker, so the
command can safely be rerun and will also process nested archives.

The export must contain event-camera-aligned RGB. Raw official 1440x1080 frame
camera images cannot be paired directly with 640x480 event-camera disparity.
The loader accepts a clearly named event-aligned directory or a custom 640x480
RGB export; untouched 1440x1080 frame-camera images are rejected.

## 1. Fine-tune then test

Event method on GPUs 4-7:

```bash
GPUS=4,5,6,7 NUM_PROCESSES=4 \
  bash dsec_exp/run_dsec_finetune_and_test.sh
```

Pure RGB baseline:

```bash
APPROACH=rgb EXP_NAME=dsec_rgb_baseline \
  bash dsec_exp/run_dsec_finetune_and_test.sh
```

## 2. Zero-shot held-out test

```bash
CHECKPOINT=/path/to/checkpoint-last.pth APPROACH=full_img_reliability \
  bash dsec_exp/run_dsec_zero_shot_test.sh
```

Results are written below `dsec_exp/results/<experiment>/`, including one CSV
row per test scene, an aggregate JSON report, and visual panels. DSEC does not
provide trajectory ground truth in this layout, so ATE/RPE are intentionally
not reported. Pose and world-point losses are disabled during fine-tuning.

Before launching distributed training, the one-click script also writes
`loader_check/first_clip.png` and `first_clip.json`. Inspect this preview once:
RGB edges, event edges, and depth boundaries must occupy the same pixels.
