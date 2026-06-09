# Paper Ablation Scripts

This folder collects the paper-facing ablations for the current best route:
multi-LDR exposure invariance, GT geometry-detail supervision, and
image-guided event reliability.

## Train

Run all default ablations, one two-GPU job at a time:

```bash
bash ablation/run_paper_ablation_2gpu.sh
```

Useful environment variables:

```bash
GPUS=0,1 EPOCHS=20 NUM_VIEWS=4 LDR_ID=ev_5 bash ablation/run_paper_ablation_2gpu.sh
VARIANTS=rgb_baseline,raw_event,full_img_reliability bash ablation/run_paper_ablation_2gpu.sh
```

The ablation runner disables in-training test evaluation by default
(`EVAL_EVERY_STEPS=0`, `SKIP_FINAL_EVAL=true`) to avoid long DDP waits.  Run
the EAG3R-style metric script after training for paper numbers.

When launching `finetune_paper_ablation.py` manually, use Hydra's append
syntax for the variant field:

```bash
accelerate launch --multi_gpu --num_processes 2 ablation/finetune_paper_ablation.py +ablation_variant=full_img_reliability exp_name=ablation_full_img_reliability
```

Default variants:

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

Main reported metrics follow the EAG3R table style:

- Depth: `AbsRel`, `delta1`, `RMSElog`
- Pose: `ATE`, `RPE_t`, `RPE_r`

The evaluator also reports normal/detail diagnostics for our geometry-detail
story: normal angular error, high-event/low-event normal error, and event-detail
correlation.
