# DSEC 4/2 Split Preflight

The intended custom split is:

- `val/`: four training sequences
- `test/`: two held-out test sequences

For hyperparameter selection, first use three `val` sequences for training and
one for validation. After fixing all settings, retrain on all four and evaluate
the two test sequences once.

Before adding a loader, run:

```bash
bash paper_main_ablation/run_dsec_preflight.sh
```

Official DSEC stores unrectified 640x480 events, rectified 1440x1080 global
shutter images, and disparity in a declared reference camera. A usable
EventVGGT export must therefore provide or document:

1. event rectification using `rectify_map`;
2. timestamp alignment using `t_offset` and image timestamps;
3. RGB warped/aligned to the chosen event-camera coordinate frame;
4. metric depth converted from disparity using calibration;
5. a valid mask for zero disparity;
6. camera poses if pose loss and ATE/RPE are required.

The inspector deliberately blocks direct training when it cannot prove RGB,
events, and supervision are pixel-aligned. This avoids a numerically runnable
but scientifically invalid DSEC experiment.
