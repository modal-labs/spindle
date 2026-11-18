# Codegolf streamed minibatch measurement

The short-rollout comparison did not demonstrate an end-to-end speedup.
Across six measured updates per variant, the median was 48.0 seconds for
`batch-v11` and 51.0 seconds for `stream-v11`, including weight publication.
The observed streamed time was 6.2% longer. Variation between trials was larger
than this difference, so it does not establish a causal regression either.

## Setup

Measured source: `210d46d68689e427a16516c0ae93f62880e37857` on November 18, 2026.
Qwen3.5-9B full training used 8 H200 GPUs with TP2/CP2/DP2 and four configured
H200 sampler replicas. The trainer stayed allocated across trials. Sampler
processes restarted between trials, and their physical hosts were not pinned.

Each trial started from base weights. Updates used four problems with eight
samples each, a 2,048-token output cap, temperature 1, sequence-mean PPO and
Adam at 1e-6. Both variants used the same task order and dataset. Sampling seeds
were not fixed. The reused reference-validation list had 139 problems; two were
reserved for evaluation. This is a shorter workload than the default recipe.

## Results

Trials ran in batch–stream–stream–batch order. Updates 1–2 were warmups. The
comparison uses updates 3–5 from every trial and excludes checkpoint step 6
entirely. The last trial was stopped after all timed updates were published,
before its unmeasured final save. Earlier final checkpoints were about 193 GiB
each and took several minutes to write.

| Mode | First trial median | Second trial median | Pooled update median | Mean generated tokens per completion |
| --- | ---: | ---: | ---: | ---: |
| Batched | 42.8 s | 51.6 s | 48.0 s | 236.1 |
| Streamed | 52.7 s | 50.6 s | 51.0 s | 259.3 |

Streaming reduced the estimated exposed tail from 5.6 to 3.2 seconds. This
estimate subtracts the slowest recorded group duration from the loop time; it
includes optimizer, persistence and client overhead as well as training.
Publication alone had medians of 16.2 seconds batched and 16.8 seconds streamed.
Generated lengths and rollout tails varied enough to outweigh the small tail
saving. These results support keeping streaming opt-in, without a speedup claim.

[Run records and phase timings](codegolf-streaming-benchmark.json) include every
timed update, aggregate group durations, hardware, package versions and dataset
hashes. Prompts, generated code and credentials are excluded. Twelve timed
optimizer updates succeeded. The experiment does not establish learning quality,
GPU gradient parity, long-rollout performance or multi-tenant LoRA performance.

## Validation

The GPU path used the stock Tinker SDK 0.33.1 pointed at Spindle's protected URL.
It exercised sampling, judging, streamed accumulation, optimizer updates and
publication. No stock cookbook recipe or GPU recovery injection was run.

Local checks passed formatting, Ruff and ty, 1,572 core CPU tests, and 89 example
tests. Core tests requiring the unbuilt Rust router were excluded. An existing
example judge test fails in macOS resource-limit setup and was excluded from the
passing example count. CPU autograd tests compare batched and streamed PPO
gradients for unequal lengths with GRPO and TailRL advantages.

All benchmark apps stopped with zero remaining containers. Timing records were
retained, and the dedicated temporary checkpoint Volume was deleted.
