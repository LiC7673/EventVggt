# Full Image-Reliability Core Ablation

| Variant | Gate | Image-guided reliability loss | Direct GT detail loss | Exposure strategy |
| --- | ---: | ---: | ---: | --- |
| `temporal_detail_no_gate` |  |  | yes | random LDR |
| `gate_no_img_supervision` | yes |  | yes | random LDR |
| `img_reliability_no_detail_gt` | yes | yes |  | random LDR |
| `full_img_reliability` | yes | yes | yes | random LDR |
| `full_img_reliability_token_multildr` | yes | yes | yes | paired token |

All rows use the same 12 scenes, original VGGT initialization, event residual
architecture, LDR set `ev_1/ev_2/ev_5/ev_10`, and 20-epoch budget.

`img_reliability_no_detail_gt` removes direct normal/HF/gradient supervision
from the geometry output. GT geometry is still used to construct the
reliability target; otherwise it would no longer be an image-guided reliability
experiment.

Run all five rows on GPUs 2-7:

```bash
bash full_img_core_ablation/run_core_ablation_gpus_234567.sh
```

The runner executes three jobs in the first wave and two in the second. Outputs
are stored under `abl_event_exp/full_img_core_ablation/`.
