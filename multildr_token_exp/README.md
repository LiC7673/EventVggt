# Multi-LDR Strategy Comparison

All three rows use the historical `full_img_reliability` model, the same 12
training scenes, LDR levels `ev_1/ev_2/ev_5/ev_10`, GT detail loss, and
trainable internal image-guided reliability. Only the Multi-LDR strategy
changes.

| Script | Strategy |
| --- | --- |
| `finetune_random_ldr_full.py` | One randomly selected exposure per batch; no pair loss. |
| `finetune_paired_output_full.py` | Two exposures of the same window; final depth/normal consistency. |
| `finetune_paired_token_full.py` | Two exposures; asymmetric patch-token consistency only. |

The token strategy inserts a zero-initialized bounded residual adapter between
the frozen VGGT aggregator and the unchanged DPT decoders. Camera/register
tokens are untouched. The better-exposed observation is selected as a detached
teacher, preventing mutual averaging.

Run all three jobs concurrently on GPUs 2-7:

```bash
bash multildr_token_exp/run_three_strategies_gpus_234567.sh
```

Outputs are written to:

```text
abl_event_exp/multildr_token_strategy/
  scene_manifest.json
  random_ldr_full/
  paired_output_full/
  paired_token_full/
```

For the token row, verify that `ldr_token_pair_count > 0`, token cosine rises,
and depth-derived normal quality does not regress. A zero pair count means the
paired sampler or instance grouping is broken.
