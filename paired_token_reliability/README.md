# Paired-token ReliabilityUNet

This experiment keeps the successful paired-token Multi-LDR teacher but moves
event reliability into a separately supervised U-Net.

## Pipeline

1. A frozen `paired_token_full` model processes two LDR exposures of the same
   window. The event voxel, depth, pose, and frame indices are identical.
2. The cached soft target is
   `event support * GT normal-gradient detail * paired-token agreement`.
3. `ReliabilityUNet(full event voxel, LDR)` predicts that target for both
   exposures. BCE, cross-exposure consistency, and positive/negative ranking
   train it without a global smoothness loss.
4. Stage 2 freezes the U-Net. The complete event voxel enters the temporal
   detail refiner; a 3x3-dilated reliability map gates only the final log-depth
   correction with a nonzero floor.

The design intentionally avoids `filtered_event = event * reliability`, which
previously deleted sparse detail evidence before event encoding.

## Run

```bash
bash paired_token_reliability/run_full_pipeline.sh
```

Useful overrides:

```bash
STAGE1_GPU=6 STAGE2_GPUS=6,7 EPOCHS_STAGE1=20 EPOCHS_STAGE2=20 \
  bash paired_token_reliability/run_full_pipeline.sh
```

Results are written under `abl_event_exp/paired_token_reliability`, never the
repository-level `checkpoints` directory. Existing model files are not edited;
the new VGGT variant lives in
`eventvggt/models/streamvggt_paired_token_reliability_detail.py`.
