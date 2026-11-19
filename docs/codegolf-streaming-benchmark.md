# Codegolf streamed minibatch measurement

A later controlled replay measured a conditional 0.78-second mean scheduling
saving across matched recorded arrivals. Its numerical checks failed; see
[Controlled replay](#controlled-replay) for the limits.

The original short-rollout comparison did not demonstrate an end-to-end speedup.
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

## Controlled replay

The replay uses identical tokens, rewards, behavior log probabilities and group
readiness times from the first batched trial's updates 3–5. It pins the model
revision to `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, disables dropout and uses
one 8-H200 trainer with 32 requested CPU cores and 256 GiB of host memory.
No sampler GPUs are allocated. Each update has four groups of eight samples.
The three cases average 83.4, 336.4 and 340.4 completion tokens per sample.
Both modes use stock Tinker SDK 0.33.1 with Spindle's full-training client
factory. No stock cookbook run was performed.

The comparison has four pairs per workload and readiness profile, with
balanced batch-first and stream-first order and a shuffled pair schedule seeded
with 42. The recorded profile releases groups at their original sampling plus
judging durations. The all-ready control releases every group immediately.
Twelve warmup updates cover both modes on all three workloads before timing.
Optimizer calls use learning rate zero and gradient clipping at 1. Optimizer
state still advances. The final cold-versus-post-run probe failed, so policy
stability was not established by this run.

Each timer starts before group arrivals and ends when the optimizer response
returns. It includes datum construction, transport and training. Result writes,
weight publication, checkpoints and actual generation are excluded. These are
replay timings, not an end-to-end RL speedup or a learning-quality measurement.

### Numerical controls

Strict GPU checks on `f4fe041` failed before the timing phase. The same frozen
packed request repeated exactly, but changing its grouping changed target-token
log probabilities. Both training modes started from identical base probes.

| Check | Maximum absolute log-probability difference | Mean absolute difference |
| --- | ---: | ---: |
| Frozen request repeated | 0 | 0 |
| Frozen full batch versus separate groups | 2.941746 | 0.003586 |
| Base probe after model reset | 0 | 0 |
| Probe after one batched versus streamed update | 2.055891 | 0.022789 |

The frozen comparison covers all 32 sequences. The post-update probe covers the
first sequence of each group. These checks include prompt target positions as
well as completion positions. The update used learning rate 1e-5, ten times the
recipe rate, and gradient clipping at 1. Batched and streamed gradient norms
were 1.320160 and 1.319305, a 0.065% difference. The preset limits were 1e-3 maximum
and 1e-4 mean post-update log-probability error, and 5e-4 relative gradient-norm
error. An earlier check on `400cf8a` also failed its post-update limit.

These failures do not establish a learning-quality regression or identify a
root cause. They do establish that this run did not meet the strict numerical
comparison. Timing-only replay retains these failed checks and explicitly sets
`numerical_equivalence_established` to false; it does not relax the limits or
label the update paths numerically equivalent.

### Replay times and limits

Source `e8d519d` completed 48 timed updates after 12 warmups. Every input and
readiness hash matched within its pair, and every optimizer update succeeded.
The following means are conditional wall times; validation did not pass.

| Readiness profile | Batched mean | Streamed mean | Mean paired saving, batched minus streamed |
| --- | ---: | ---: | ---: |
| Recorded arrivals | 26.70 s | 25.92 s | 0.78 s, interval 0.70–0.86 s |
| All groups ready | 3.71 s | 6.31 s | -2.61 s, interval -2.68 to -2.54 s |

The intervals use 10,000 paired bootstrap resamples stratified within the three
fixed workloads. They describe timing variation for these cases and do not
account for numerical-validation uncertainty or generalize to other workloads.
Recorded-arrival savings per case were 0.99, 1.81 and -0.45 seconds. One case
was slower with streaming. Pooled recorded-profile medians were 25.99 seconds
batched and 26.43 seconds streamed; the paired mean reflects the differing
effects across cases. Smaller calls added about 2.6 seconds when there
was no rollout wait to hide them behind.

The final probe compared against a cold reference taken before warmup and
changed by 3.646363 maximum and 0.008350 mean absolute log probability, above
the 1e-6 maximum limit. The driver raised an error after collecting all timings.
The table was computed afterward from all 48 retained records. It is not a
passed frozen-policy benchmark or evidence of equivalent training updates.
Warmup effects were not isolated from changes during the timed phase.

The driver now takes its reference probe after warmup and persists failed
probe diagnostics before raising. That change was not rerun on GPUs. The
numerical limits remain unchanged. The strict update comparison also remains
failed. No learning-quality run, stock cookbook run, LoRA comparison or GPU
recovery injection was performed.

[Replay records and failed checks](codegolf-streaming-replay.json) include every
timed update, the fixed arrival schedules, input hashes, versions and conditional
summary. The replay used stock Tinker SDK 0.33.1, PyTorch 2.11.0+cu130,
Transformers 5.12.1, Megatron Core 0.18.0+2d1fa8d37 and NVIDIA driver 620.70.06.
The numerical checks ran separately on `f4fe041`.

Local validation passed pre-commit, 1,572 core CPU tests with the existing
native-router exclusions, and 107 example tests with the existing macOS judge
test excluded. All eight replay apps, including failed and superseded starts,
stopped with zero containers. The empty temporary checkpoint Volume was deleted;
timing and diagnostic records were retained.
