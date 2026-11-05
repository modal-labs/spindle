# Deterministic training (experimental)

Spindle now has an opt-in Miles multi-LoRA path ported from
[Lilo PR #28](https://github.com/modal-projects/lilo/pull/28), which remains open.
The old documentation described that experiment, but its deployment definitions,
runtime hooks, and patched FA3 image were absent from Spindle main.

This is a validation candidate, not a guarantee across all models. Keep fixed
the GPU architecture, dependency revisions, TP layout, ordered training data,
initial weights, optimizer state, and forward/backward request and update boundaries. Increasing precision
alone does not make atomic reductions or asynchronous scheduling reproducible.

The default keeps different clients in separate microbatches, preserving each
client's isolated shapes and accumulation order while sharing the trainer and
base weights. This trades some multi-client batching throughput for repeatability.
Cross-client packing from the Lilo prototype is retained only as an explicit
`miles_cfg.deterministic_pack_clients=True` experiment. It failed exact gradient
parity on Qwen3-8B TP4 in the current stack and is not a deterministic guarantee.

## Enable

Two recipes expose the path through the ordinary Tinker API:

| Recipe | Trainer | Attention |
| --- | --- | --- |
| `qwen3-8b-lora-32k-deterministic` | 4 H100, TP4 / DP1 | native deterministic FA3 for smaller heads |
| `qwen35-9b-lora-16k-deterministic` | 4 H100, TP4 / DP1 | patched FA3 hdim256 plus fixed GDN layouts |

For Qwen3.5, build the artifact against Spindle's pinned Miles base image first:

```bash
PYTHONPATH=src python scripts/export_fa3_test_wheel.py
export SPINDLE_MILES_FA3_ARTIFACTS="$PWD/scripts/results/fa3_deterministic/artifacts"
spindle config validate src/spindle/configs/qwen35_9b_lora_16k_deterministic.py
spindle deploy --app spindle-deterministic-test qwen35-9b-lora-16k-deterministic
```

Select the intended Modal workspace/environment before running these commands.
Use an isolated frontend for validation. The artifact directory contains a wheel
and manifest with pinned source revision, wheel checksum, and CUDA binary hashes.
The image verifies the installed binaries before relaxing TE's hdim256 filter.
This specialization targets SM90 and hdim256 only; it is not a Blackwell build.
Missing artifacts cause deployment to fail instead of silently using another image.

Custom dense-model recipes can set `miles_cfg.deterministic_training=True`,
`tp_reduce_precision="float64"`, TP equal to the trainer GPU count, and CP=1.
Set `deterministic_gdn=True` for the Qwen3.5 recurrence hooks. Set
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
`NCCL_ALGO=Ring`, and `NCCL_PROTO=Simple` in `trainer_env` before CUDA starts.
These settings are forwarded to actors on an existing Ray cluster. The current
candidate rejects GPU architectures other than Hopper SM90 at actor startup.

The mode uses topology-aware seeded adapter initialization, FP64 TP/vocabulary
reductions, deterministic dynamic-shape CE compilation, deterministic attention,
and isolated per-client microbatch boundaries and accumulation order. Each fresh adapter
uses the trainer seed (Miles' default or `miles_cfg.cli_options.seed`), independent
of the selected slot and previously created adapters. Per-model Tinker seeds
remain unsupported. An explicit model-and-optimizer checkpoint is the strongest
starting point for comparing optimizer variants.

The scalar loss hook also removes a rounding artifact: multiplying a summed
Tinker loss by N microbatches and dividing it back out yields different FP32
gradient scales for N=55 and N=173. Both normalization sites use one in this
mode; the scheduler still executes the actual microbatches.

Both recipes request deterministic SGLang inference. The pinned sampler image
preserves FP32 GDN prefix states only when that inference mode is enabled.
Trainer repeatability and sampler batch invariance need separate validation.

## Replay validation

```bash
PYTHONPATH=src modal run scripts/check_deterministic_training_gpu.py \
  --models Qwen/Qwen3-0.6B,Qwen/Qwen3-8B --steps 12

# Exercise the Qwen3-8B recipe's tensor-parallel layout.
SPINDLE_DETERMINISM_TP=4 PYTHONPATH=src \
  modal run scripts/check_deterministic_training_gpu.py \
  --models Qwen/Qwen3-8B --steps 12

# Requires SPINDLE_MILES_FA3_ARTIFACTS from the build above.
PYTHONPATH=src modal run scripts/check_deterministic_training_gpu.py \
  --models Qwen/Qwen3.5-2B --steps 12
```

This launches an isolated trainer (one H100 by default), sequentially for each model. It
executes rank-8 adapters and fixed synthetic cross-entropy inputs with alternating sequence lengths
in four fresh processes: reference, exact replay from the initial checkpoint,
replay sharing work units with another client's changing batches, and continuation from a
midpoint model-and-optimizer checkpoint. Every step compares loss, token
logprobs, gradient norm, and per-rank hashes of adapter weights, gradients,
and optimizer state. The reference process also restores its initial checkpoint
and repeats the loop in that same process. The shared-work-unit case verifies
that the default keeps clients in separate microbatches. With `--pack-clients`,
the harness instead requires actual mixed-client microbatches and tests the
experimental packing path. A fixed final evaluation reports replay NLL difference.
Missing steps and mismatched initial/restored state fail the check. The default
12 steps extend beyond the observed step-3/step-6 divergence.

The harness calls Spindle's real Miles backend and distributed actors directly;
it does not exercise the HTTP/control-plane scheduler or an online RL loop.
The report is `scripts/results/deterministic-training/report.json`. A failing
case records the first divergent step and field; it is not converted into a
tolerance-based pass. Use `--no-deterministic` for a control with the ordinary
trainer path. The scripts create ephemeral apps, not shared deployments.

The independent FA3 kernel checks are also retained:

```bash
PYTHONPATH=src python scripts/run_fa3_deterministic.py
PYTHONPATH=src python scripts/run_fa3_deterministic.py --forward-backward
PYTHONPATH=src python scripts/run_fa3_deterministic.py --upstream-tests
```

## Limits

DP>1, CP>1, pipeline interleaving, multimodal inputs, MoE models, MTP,
dropout, and per-token loss normalization are outside the supported candidate.
Unsupported topology/options are rejected, and MoE providers fail before model
construction. The FA2 fallback for unpatched hdim256 is not evidence of packing
invariance. Full-parameter training and TriMuon optimizer implementations are
not covered by this multi-LoRA path. Changing client update order, asynchronous
policy lag, prompts, or sampled data is a different experiment even with exact
kernels. Forward-only scoring is not generally guaranteed invariant to packing.

Historical Qwen3.5 GSM8K/DAPO parity plots predate the current dependency stack.
The Lilo PR explicitly says its full GPU parity run was not repeated after its
latest integration/loss-scaling changes. Treat those plots as historical evidence,
not certification of current Spindle or another model family.
