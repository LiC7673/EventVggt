# Paper-Scale 60/12/12 Training

This pipeline freezes one scene-disjoint split and never uses test scenes for
either ReliabilityNet training or VGGT checkpoint selection:

- 60 train scenes: all adjacent frame pairs, paired Multi-LDR observations;
- 12 validation scenes: fixed LDR, evaluated after every epoch;
- 12 test scenes: loaded only by the final evaluation command.

Stage 1 renders reliability labels from the 60 train scenes and 12 validation
scenes only. Train scenes are distributed across `ev_2/ev_5/ev_10`, keeping the
label set near 7200 frames instead of tripling every frame. Stage 2 freezes the
selected ReliabilityNet and trains the full model with event-supported paired
cross-exposure depth/normal consistency, GT geometry-detail supervision, and
causal event residuals. Paired exposure samples are forwarded sequentially to
control GPU memory, then concatenated before consistency loss computation.

## Train on GPUs 4,5,6,7

```bash
bash paper_scale_training/run_train_gpus_4567.sh
```

If the split-specific Stage-1 checkpoint is absent, the script first trains it
for 20 epochs on physical GPU 4. It then launches VGGT on GPUs 4,5,6,7.
Stage-2 defaults are `num_views=2`, 30 epochs, validation every epoch, and early
stopping patience 5. Thus a 120-frame scene contributes 119 adjacent pairs.
The best checkpoint is selected only by validation loss. Set `NUM_VIEWS=4`
only when a four-frame window experiment is intentionally required.

Optional controls:

```bash
SCENE_METADATA=/path/to/scenes.csv EVAL_LDR_ID=ev_5 \
  bash paper_scale_training/run_train_gpus_4567.sh
```

The optional CSV should contain `scene,material,lighting,motion`; without it,
the split generator uses material-like keywords in scene names as an
approximate stratum. Add a `split` column containing exactly 60 `train`, 12
`val`, and 12 `test` rows to enforce a manually curated extreme-lighting
holdout. Once generated, `scene_split.json` is reused unchanged.
Inspect progress with:

```bash
tail -f abl_event_exp/paper_scale_60_12_12/stage1_train.log
tail -f abl_event_exp/paper_scale_60_12_12/train_gpus_4567.log
```

## Final test

Run this after all architecture and hyperparameter decisions are frozen:

```bash
GPU=7 bash paper_scale_training/run_test_once.sh
```

The immutable split is stored in `scene_split.json`, and the final metrics are
stored under `final_test_12_scenes/`. The evaluator reports depth metrics,
normal angular metrics, ATE, translational RPE, and rotational RPE.

Runtime checks for Stage 2:

- `ldr_exp_depth_loss` and `ldr_exp_normal_loss` must be non-zero during train;
- validation logs must list exactly 12 scenes and about `12 x 119` pairs;
- `checkpoint-best.pth` must be under `full_model_train60_val12/`;
- no test scene name should appear in either Stage-1 or Stage-2 training logs.
