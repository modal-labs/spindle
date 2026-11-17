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
initialization parity. Qwen3.5 variants do not change the bundled defaults.

## CPU checks

The locked Python 3.12 development environment passes the full pre-commit
formatting, lint, and type checks. With CI's CPU Torch 2.10.0 and the built
router extension, the CPU suite reports 1,340 passed, 8 skipped, and one failure:
`test_add_cli_validation_fails_before_http` for `--=1`. Argparse exits 2 rather
than the expected 1; the PR already records the same failure on its base.
