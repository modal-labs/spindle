# Online MXFP8 reload validation

Online MXFP8 sampling can quantize a BF16 checkpoint when it starts. Loading a
later BF16 publication also needs to restore the checkpoint dtype before the
quantization kernels run. The previous loader kept FP8 storage and failed its
second quantization on NVIDIA B200.

[SGLang #91](https://github.com/modal-projects/sglang/pull/91) restores the
checkpoint dtype and shape for online MXFP8 linear and expert weights. It keeps
parameter objects and their loader attributes intact. Serialized quantized
checkpoints retain their existing restore path.

Eight tests passed on B200 with the changed source and test files from pinned
SGLang revision `2a3714aa0904566431d5edad2e94ea51a8b9f49e`.
The dense and MoE tests each load two successive BF16 updates and compare every
weight and scale parameter with a freshly quantized layer. Both updates match
exactly. BF16 expert reload tests also cover the packed layouts used by the
ordinary and routed FlashInfer TRT-LLM backends. They preserve parameter storage
and copy the new checkpoint values exactly.

Raw evidence: [failure before the fix](assets/quantized-sampler/b200-reload-before.json),
[fixed dense-layer comparison](assets/quantized-sampler/b200-reload-after.json),
and [eight layer tests](assets/quantized-sampler/b200-reload-tests.json).
The [initial four-test run](assets/quantized-sampler/b200-tests.json) records the
MXFP8 checks before the BF16 layout fix was added.

These checks establish layer reload correctness on B200. Full-model timings
require the separate B300 comparison below.

## Full-parameter MoE optimizer

The first full-parameter Qwen3.5-35B-A3B optimizer update failed with
Transformer Engine 2.17's fixed tensor-handle pool limit. Bounded fused Adam
kernel calls retain the same parameter groups, update equations, and checkpoint
layout. A B200 regression with 6,000 parameter tensors passed three updates
against PyTorch AdamW. Restoring the optimizer checkpoint reproduced the next
update exactly. [Raw test evidence](assets/quantized-sampler/b200-adam-tests.json).

Qwen3.5's 64-channel `in_proj_ba` gate is also below the selected MXFP8 dense
kernel's minimum output width. The MXFP8 experiment keeps that projection in
BF16 with `SGLANG_FP8_IGNORED_LAYERS=in_proj_ba`. Expert weights and supported
linear layers use MXFP8.

## B300 comparison setup

The matched runs use Qwen3.5-35B-A3B at model revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`, four B300 GPUs for each BF16
full-parameter trainer, and one B300 GPU for each sampler. The loss is
`score_centered_mis`. Each update samples 32 sequences with a 128-token cap.
[Resolved configurations](assets/quantized-sampler/b300-comparison-configs.json)
and the [benchmark](../examples/quantized-sampler/benchmark.py) make the workload
reproducible. Generated sequences and token counts can differ across precisions.
These are on-policy timings, not a fixed-token trainer kernel comparison.

The MXFP8 run uses SGLang overlay `a1667d68a4017026ed7f8ce1c19261862aecee19`.
The BF16 sampler was refreshed to `2a3714aa0904566431d5edad2e94ea51a8b9f49e`
to fix its first publication. The only difference between those overlays is
the BF16 expert shape restore and its tests. The MXFP8 source is identical.
The final Spindle pin uses the latter revision. Runtime source hashes are
recorded with the [memory evidence](assets/quantized-sampler/b300-memory.json).
The base image's Git HEAD identifies its upstream checkout; the fork overlay
copies Python and Rust files without replacing that Git metadata.

Both deployments include the bounded Adam calls from local runtime commit
`e932db6`, based on Spindle `93e03ce`. The runtime and benchmark source was
published in [Spindle commit 265fd63](https://github.com/modal-projects/spindle/commit/265fd635ea7d119f0723b8f3ce57c38490edb422). The later merge of `9028a74` adds synthetic
monitoring files and does not change the validated runtime. The public benchmark
matches the remote loop except that zero probability change on an individual
quantized update is allowed; at least one update must change probabilities over
the run. All other publication and training checks remain the same.

The [CPU checks](assets/quantized-sampler/cpu-validation.json) passed 1,201 tests
and skipped 10 GPU-only tests. One CLI argument parsing case also fails on the
unchanged base commit. Ruff formatting and lint checks passed.
GitHub Actions [Core CPU tests](https://github.com/modal-projects/spindle/actions/runs/42693158951)
and [Codegolf example](https://github.com/modal-projects/spindle/actions/runs/42693159417)
both passed on published commit `265fd63`.

## B300 results

Both modes completed ten updates and served every publication. The table uses
medians of updates 3–10, excluding the first two for startup and compilation.
BF16 update 1 also includes recovery from the reload bug and is not a timing
baseline. Later compilation and storage variation remain in the raw records.

| Median seconds per update | BF16 sampler | MXFP8 sampler | MXFP8 / BF16 |
| --- | ---: | ---: | ---: |
| Rollout | 4.07 | 3.00 | 0.74x |
| Forward/backward | 33.50 | 33.53 | 1.00x |
| Optimizer | 0.91 | 1.15 | 1.26x |
| Save publication | 38.95 | 38.41 | 0.99x |
| Publication through serving | 125.51 | 150.78 | 1.20x |
| Complete update | 166.24 | 191.93 | 1.15x |

Publication through serving includes saving the publication; those rows overlap.
Complete update also includes data preparation and fixed-sequence probability
probes used to validate serving. All optimizer updates succeeded with finite
metrics and positive gradient norms. Each publication changed fixed-sequence
log probabilities.

The median generated tokens per update were 3216 for BF16 and 3502.5 for
MXFP8. Both trainers use BF16, so these timings do not measure a quantized
training kernel speedup. MXFP8 reduces sampler weight allocation from 65.492
to 35.357 GiB (46.0%). This is not a total or peak memory measurement.
With five B300 GPUs held during each warm update, the median wall times
correspond to 0.231 allocated GPU-hours for BF16 and 0.267 for MXFP8.
These figures exclude startup and debugging.

Across 320 sampled completions per mode, arithmetic correctness was 97.81%
for BF16 and 95.31% for MXFP8. These are training prompts, not a held-out
evaluation. The short run, different on-policy samples, and high initial
accuracy do not establish convergence or quality parity.

For this batch-32 workload, MXFP8 saves sampler weight memory and rollout
time, but publication costs outweigh the rollout saving. The full loop is
slower. This result does not establish the difference at other batch sizes,
sequence lengths, or model sizes.

Raw runs: [BF16](assets/quantized-sampler/b300-bf16-run.json),
[MXFP8](assets/quantized-sampler/b300-mxfp8-run.json), and
[derived summary](assets/quantized-sampler/b300-summary.json).
