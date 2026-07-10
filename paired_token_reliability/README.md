# Multi-LDR Event Contribution — Stage 1

The canonical Stage-1 experiment now learns **which events improve geometry
under the bad exposure**. It no longer distills a token-cosine reliability
label.

## Data flow

1. Exposure levels default to `ev_0 < ev_1 < ev_2 < ev_5 < ev_10` and all
   ten lower-to-higher pairs are sampled. The measured saturation ratio still
   verifies/reorients each pair before supervision.
2. The bridge mask is the intersection of bad-exposure saturation,
   visible reference structure, and event support. Pairs with too little
   bridge area are skipped.
3. A frozen RGB-VGGT predicts coarse depth, normals, and patch features from
   the bad exposure. These tensors are detached.
4. Stage 1-A fixes `C=1` and trains only `SelectedEventRefiner`, establishing
   a proxy event-utility evaluator.
5. Stage 1-B freezes that proxy and trains only
   `ContributionNet(V, I_bad, stopgrad(G_coarse))`. This removes the
   ContributionNet/Refiner scale ambiguity.
6. Optional Stage 1-C jointly tunes both modules for 1--2 epochs with the
   refiner learning rate scaled to `0.1` of the contribution learning rate.
   It is disabled by default.
7. `SelectedEventRefiner` receives only `C * V` and detached coarse geometry.
   There is no full-event bypass. Zero selected events return the RGB coarse
   result exactly.
8. Final depth and normal are compared with fixed GT inside the bridge. GT
   geometry detail is only a soft emphasis weight. An event-mass budget keeps
   the mean contribution near `rho=0.5`.

`checkpoint-best.pth` is always the Stage 1-B checkpoint consumed by Stage 2;
optional joint checkpoints are saved separately as `checkpoint-joint-*.pth`.
The checkpoint contains both `contribution_net` and `event_refiner`. Legacy
`ReliabilityUNet` checkpoints are intentionally schema-incompatible.

## Run

```bash
GPU=0 PRETRAINED=ckpt/model.pt \
  bash paired_token_reliability/run_contribution_stage1.sh \
  data.root=/data/reflective_raw data.active_scene_count=12
```

Schedule controls:

```bash
EPOCHS_PROXY=5 EPOCHS_CONTRIBUTION=15 EPOCHS_JOINT=0 \
  bash paired_token_reliability/run_contribution_stage1.sh ...
```

Pair strategies:

```bash
# All C(5,2)=10 pairs (default)
PAIR_MODE=all EXPOSURES=0,1,2,5,10 bash paired_token_reliability/run_contribution_stage1.sh ...

# Only 0->1, 1->2, 2->5, 5->10
PAIR_MODE=adjacent bash paired_token_reliability/run_contribution_stage1.sh ...

# Only 0->1, 0->2, 0->5, 0->10
PAIR_MODE=anchor bash paired_token_reliability/run_contribution_stage1.sh ...

# Explicit pairs: invoke the Python entry directly
python -m paired_token_reliability.train_contribution_stage1 \
  --pair-mode explicit --exposures 0,1,2,5,10 \
  --pair 'ev_0->ev_5' --pair 'ev_1->ev_10' \
  data.root=/data/reflective_raw
```

The runner first executes CPU architectural tests, trains Stage 1, then runs
the held-out counterfactual conditions:

- RGB coarse / `C=0`
- full events / `C=1`
- learned contribution
- random spatial contribution with the same active-event values
- remove the highest-scored events
- remove the lowest-scored events

Set `RUN_NO_BRIDGE=1` to additionally train the required
`ContributionNet without bridge` ablation and include it in the same held-out
table. Together with RGB coarse, full events, and the full bridge method this
covers the four minimum comparisons in `toDo.md`.

The learned score passes its causal sanity check only if removing high-score
events hurts more than removing the same number of low-score events.

Tests alone:

```bash
bash paired_token_reliability/run_contribution_tests.sh

# Also evaluate a real checkpoint:
bash paired_token_reliability/run_contribution_tests.sh \
  abl_event_exp/event_contribution_stage1/checkpoint-best.pth \
  --rgb-checkpoint ckpt/model.pt
```

The old `export_targets.py`, `train_reliability.py`, and `run_full_pipeline.sh`
remain only to reproduce historical experiments; they are not part of this
Stage-1 design.
