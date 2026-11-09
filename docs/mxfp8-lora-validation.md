# MXFP8 frozen base with BF16 LoRA

`qwen35_35b_a3b_mxfp8_lora_8k` uses MXFP8 frozen base projections in the
Miles LoRA trainer and SGLang sampler. Trainable adapter parameters and adapter
publications stay BF16. `qwen35_35b_a3b_lora_8k` is the matched BF16 control.
Both use Qwen3.5-35B-A3B, rank 16, four B300 trainer GPUs, one B300 sampler GPU,
and an 8,192-token context. The model revision is pinned in both recipes.

Set `miles_cfg.base_quantization="mxfp8"` for the frozen trainer base and
`sglang_cfg.quantization="mxfp8"` for the sampler. The trainer verifies native
MXFP8 storage and BF16 trainable parameters before accepting work. Embeddings,
the output head, the vision tower, and normalization retain their normal precision. The sampler also
keeps the 64-channel `in_proj_ba` projection in BF16.

Use the stock Tinker SDK's `create_lora_training_client` with rank 16 and
`train_unembed=False`. The recipes target attention and routed/shared MLP
projections, including expert adapters. They do not train the output head.

The trainer patch maps adapter slots onto MXFP8's padded expert rows. The
serving patch applies BF16 adapter deltas after the gate/up projection, before
SiLU, and after the down projection, before routing weights are applied.
DeepGEMM retains native MXFP8 base GEMMs. Both masked decode and packed prefill
layouts are covered. The current serving path requires expert parallel size 1
and unclamped, unswizzled SiLU intermediates.

CPU tests compare padded adapter outputs and gradients with an unpadded
reference, including empty experts. Serving layout tests cover invalid routing
entries and check that routing weights are applied once. A B300 native expert
test passed with frozen MXFP8 weights, BF16 adapter gradients matching the
reference, and finite input gradients.

The full Qwen3.5-35B-A3B proof passed in both precisions through Tinker 0.25.0.
It used `create_lora_training_client`, `score_centered_mis`, Adam at 1e-6,
adapter publication, `save_state`, and `load_state_with_optimizer`. Each update
trained on 32 sequences with 1,024 prompt tokens and exactly 4,096 generated
tokens. The shifted trainer input has 5,119 tokens per sequence. Metrics were
finite, adapter gradients were nonzero, and the published update changed served
logprobs. Restored logprobs matched exactly in both modes. A fresh training
client also restored each checkpoint with the stock SDK.

The warm training comparison reuses those replay batches for two passes with
learning rate zero. It measures compute, not learning quality. GPU startup and
first-use compilation are excluded. API times include batch transfer and dispatch;
backend times measure forward/backward inside the trainer.

Sampling uses one B300 and 128 sequences, with 1,024 prompt tokens and exactly
4,096 generated tokens each. Two SDK requests with 64 samples each keep the
per-worker HTTP connection limit from capping concurrency at 100. Both recipes
use target/max concurrency 128, 4K prefill chunks, and the same CUDA graph sizes.
Sampling probabilities are not captured in this comparison. Three repetitions
use a pinned initial adapter with zero B weights; the BF16 adapter kernels still
run. No training or publication runs concurrently on the sampler.

Measured times in seconds, with [individual runs and metrics](mxfp8-lora-results.json):

| Operation | BF16 base | MXFP8 base |
| --- | ---: | ---: |
| Sampling 128 x 4,096 tokens, median of 3 | 119.6 | 112.1 |
| Warm forward/backward, 32 x 4,096 targets, median of 2 | 32.9 | 40.4 |
| Adapter publication after the proof update | 31.1 | 31.2 |

Sampling takes 6.3% less time with MXFP8; warm training takes 22.9% more time.
The three controlled sampling runs were 119.2–120.0 seconds for BF16 and
111.6–112.2 seconds for MXFP8. These results do not establish a large sampling
speedup. The JSON also retains the earlier 32-request diagnostic runs, which
had long admission tails and are not used for this comparison.

Backend forward/backward medians are 23.7 seconds for BF16 and 32.7 seconds
for MXFP8. Adapter publication is unchanged at about 31 seconds because both
modes export the same BF16 adapter format. The frozen sampler base is retained
across updates.

Sampler model allocation after loading was 65.49 GiB for BF16 and 35.51 GiB for
MXFP8, before KV/cache allocation. This is separate from total GPU memory use.

[Run data and source revisions](mxfp8-lora-results.json) include full proof metrics
and each timing repetition. Runtime validation covers the pinned Miles, Bridge,
Megatron-Core, and SGLang versions on B300. The installed host routing hook and
the Bridge fallback both account for MXFP8 padding. Two native trainer tests passed,
including the vision boundary; four serving layout cases passed.

The CPU suite has 1,212 passes and 15 skips, with one CLI parsing assertion that
also fails on unchanged main. Formatting, lint, type checks, and both GitHub CI
workflows passed for runtime commit `5bf7889`. Later documentation-only changes
do not alter that runtime validation. Reward and convergence were not evaluated.

Both validation deployments were stopped, with zero remaining containers.
