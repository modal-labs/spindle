# Miles native LoRA: Qwen RL validation

These short runs exercise the stock Tinker SDK against Spindle's
`MilesCommandBackend`, including deployed trainers and SGLang sampler pools.
The backend source is Spindle `6e630f7`, with Miles
`15afe217019b2762290da27a31a92454051a0d34` on its v0.2.0 image.

## Workload

Two concurrent clients per model use ranks 8 and 16. Each takes five GRPO
updates on multiplication questions, with four groups of eight responses and a
64-token generation cap. Exact-answer reward has a 0.001-per-token length
penalty. Group-centered advantages are divided by 32 before training with
`importance_sampling`; Adam learning rate is 1e-4.

Each client then saves optimizer state, takes another update, restores, and
repeats that update. Those checks run separately after concurrent RL finishes.
A fixed trainer probe checks restore behavior; final publication, prompt
scoring, and generation exercise the trained sampler adapter. Metrics are the
per-client JSONL files in this directory. The client is
[`native_lora_rl.py`](../../native_lora_rl.py).

## Results

All six models completed two clients at ranks 8 and 16, five accepted RL
updates each, checkpoint restore, a continued Adam update, final publication,
prompt scoring, and generation. Every client had a nonzero gradient and changed
trainer probe logprobs. Fully correct groups can have zero advantages; this
workload does not require a nonzero gradient on every update.

| Model | Trainer TP × DP | RL updates | Restore max difference | Next-update max difference | Sampler clients |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen25-3b | 1 × 4 | 10 | 0 | 0.156348 | 2/2 |
| qwen25-7b-instruct | 2 × 2 | 10 | 0 | 0.207634 | 2/2 |
| qwen3-4b | 1 × 4 | 10 | 0 | 0.000000 | 2/2 |
| qwen3-8b | 2 × 2 | 10 | 0 | 0.133620 | 2/2 |
| qwen35-4b | 1 × 4 | 10 | 0 | 0.000000 | 2/2 |
| qwen35-9b | 4 × 1 | 10 | 0 | 0.118724 | 2/2 |

Each trainer requests four H100 GPUs; Modal's H100 selector permits H200
upgrades. Samplers request one H100, except Qwen3-8B, which requests two GPUs
with TP2. Each model has one sampler replica. The Qwen3.5 validation samplers
request 128 GiB of host memory. These are validation allocations, not changes
to the bundled recipes or performance measurements.

Both ranks of both Qwen3.5 models have nonzero adapter B tensors for
`in_proj_q`, `in_proj_k`, `in_proj_v`, `in_proj_z`, and `out_proj` in all 24 GDN
layers. See [`gdn-projection-coverage.json`](gdn-projection-coverage.json).

## Numerical replay

Restored trainer probe logprobs match exactly in the completed checks. A new
backward/Adam step is not generally bitwise reproducible, even when isolated
from other clients. The JSONL records retain the measured difference.
`--require-exact-replay` makes that a failing check rather than requiring only
finite, accepted continuation updates.

A separate Qwen2.5-3B diagnostic started from an RL-trained rank-8 adapter,
restored it with the stock SDK, and repeated a fixed cross-entropy update.
All 1,513 saved tensors (FP32 adapter weights, Adam moments, and step) matched
byte for byte immediately after restore. After the repeated update, 1,476
tensors differed and probe logprobs differed by 0.093921. The two gradient
norms were 703.39197 and 703.50421. This establishes checkpoint-state fidelity;
it does not establish bitwise reproducibility of gradient computation.
[`checkpoint-state-comparison.json`](checkpoint-state-comparison.json) records
the complete tensor hash comparison counts.

## Qwen3.5 serving compatibility

The Qwen3.5 variants additionally select `in_proj_qkv`, `in_proj_z`, and
`out_proj`, with `train_unembed=True`, native canonical LoRA, and CP=1.
Miles splits the selected GDN QKV projection into independent Q/K/V adapters.
The other four models use the native recipes selected by the PR.

The Qwen3.5 validation image applied the GDN alias patch over SGLang
`90785332`. The same model source is now pinned at `ca6f7e1a9`
([SGLang #100](https://github.com/modal-projects/sglang/pull/100)). Without the fix,
sampler startup fails with `get_hidden_dim not implemented for in_proj_qkv`. The patch
maps both trainer names and split adapter names onto `in_proj_qkvz` through
SGLang's existing model alias mechanism. It leaves the adapter factors
independent. The aliases cover dense and MoE text and conditional-generation
classes; an isolated CPU normalization check covers eight model/target combinations,
which fail on the parent and pass with the patch. Live runs cover only the dense 4B and 9B models.

## Limits

These are short execution checks, not convergence measurements, official
Tinker parity results, or maximum-context tests. They do not cover MoE expert
adapters, CP>1, multi-node native trainers, FP32 adapter execution, or seeded
initialization parity. The Qwen3.5 variants in these original runs did not change the bundled defaults.

## Original CPU checks

The locked Python 3.12 development environment passes the full pre-commit
formatting, lint, and type checks. With CI's CPU Torch 2.10.0 and the built
router extension, the CPU suite reports 1,340 passed, 8 skipped, and one failure:
`test_add_cli_validation_fails_before_http` for `--=1`. Argparse exits 2 rather
than the expected 1; the PR already records the same failure on its base.

## Config migration audit

After integrating Spindle main `a3fd861`, the bundled set contains 53 LoRA
recipes and five full-parameter recipes. The architecture IDs were read from
the public Hugging Face configs and compared with the native registry in the
current Miles pin `384d73b`. Twenty single-node dense Qwen recipes select
`raw` / `canonical_lora`; their other resolved settings are preserved.

The other recipes cannot all be switched by changing config flags:

| Blocker | Recipes |
| --- | ---: |
| Missing `nemotron_h` native architecture | 9 |
| Missing `gpt_oss` native architecture | 4 |
| Selected routed expert adapters unsupported | 12 |
| Spindle v0.2.0 single-node restriction | 8 |

These are disjoint primary blockers; some recipes have additional restrictions.
[`config-support.json`](config-support.json) records each config and actual HF
architecture ID. Qwen3.8-27B uses `qwen3_5` internally. The 20 candidates are
not covered by the earlier six-model results above on the final merged source.
The new sampler image uses merged SGLang `e003cf9`, preserving older named images.

## Synmon regression on the merged migration

Source: Spindle `913d0c3`, Miles `384d73b`, and SGLang `e003cf9`.
The stock Tinker SDK 0.39.0 runs cookbook 0.6.1's guess-number GRPO workload
through the repository's synmon helpers against temporary checkout deployments.
Deployment isolation uses [`checkout_server` from PR #314](https://github.com/modal-labs/spindle/blob/ae383d32575e156282ca0bd9ee6fc8ef0cd933b5/synthetic_monitoring/checkout.py);
the workload uses this branch's `run_guess_number` helper.
Each model takes three steps at rank 32, with batch size 8, group size 4, a
64-token limit and learning rate 3e-5. The cookbook uses the repository's
`guess_number.patch`. Every run saves final trainer state and sampler weights.

All nine models completed, for 27 accepted updates. These use one bundled
recipe per base model; they do not cover every context-length or topology
variant among the 20 migrated recipes. Sampler settings, target sets and
speculative decoding follow those recipes. H100 requests allow H200 upgrades.

| Model | Trainer TP × DP | Sampler/trainer KL | Prior synmon KL | Reward first → last |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-3B | 1 × 4 | 0.000608 | — | -7.8438 → -7 |
| Qwen2.5-7B-Instruct | 2 × 2 | 0.000400 | — | 0.28125 → 0.625 |
| Qwen3-4B | 1 × 4 | 0.001265 | — | 0.15625 → 0.125 |
| Qwen3-8B | 2 × 2 | 0.000511 | 0.000709 | 0.0625 → 0.34375 |
| Qwen3.5-2B | 1 × 4 | 0.000328 | — | -1 → -0.28125 |
| Qwen3.5-4B | 1 × 4 | 0.000242 | 0.000331 | 0.25 → 0.21875 |
| Qwen3.5-9B | 1 × 1 | 0.000258 | 0.000417 | 0.21875 → 0.3125 |
| Qwen3.5-9B-Base | 4 × 1 | 0.000351 | 0.000456 | -4.0938 → -3.3438 |
| Qwen3.8-27B | 4 × 2 | 0.000041 | 0.000112 | -0.09375 → 1 |

[`synmon-results.json`](synmon-results.json) records the per-model results,
recipes, GPU requests, timing stages and prior metrics. Prior runs are context
for the regression, not a controlled performance or convergence comparison.
The low rewards of base models are retained; completion is an execution check.
[`native-initialization.json`](native-initialization.json) records native factory
initialization and actual trainer GPUs for all nine models.

An additional Qwen3-8B attempt on four H200s failed at its first backward
request, while NCCL reduced the microbatch count over the DP group. It reported
CUDA error 999 with 126 GB free on the affected GPU. The fresh-allocation retry
selects H200 explicitly and enables NCCL diagnostic logging; its outcome is in
the table. The cause of the failed attempt is not established. An earlier H100
run also completed. The failed attempt remains in `failed_attempts`.

Cookbook's model-name registry omitted four of these models. Synmon now selects
existing renderers for them. The three chat renderers match the models' HF
generation-prompt tokens in [`renderer-template-check.json`](renderer-template-check.json).
Qwen2.5-3B uses the cookbook's generic `role_colon` renderer for base models.

The full CPU suite on the committed source reports 1,482 passed, 10 skipped,
and the same known `--=1` CLI validation failure as the base. It uses Python
3.12, locked Tinker 0.26.0, CPU Torch 2.10.0 and the built router extension.
Formatting, lint, types and package build pass.

These runs do not establish Tinker numerical parity, convergence, maximum
context support, native multi-node support or native routed expert support.
The earlier checkpoint/replay results above use their stated older source.

All temporary regression apps and clients stopped, with zero running containers
and no remaining test volumes, secrets or proxy tokens; see
[`synmon-cleanup.json`](synmon-cleanup.json).
