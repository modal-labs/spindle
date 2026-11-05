# Deterministic training validation: November 5, 2026

These checks validate the experimental path in [Spindle #214](https://github.com/modal-projects/spindle/pull/214),
ported from unmerged [Lilo #28](https://github.com/modal-projects/lilo/pull/28).
The working tree starts from Spindle `6d63a0e`; no shared deployment was changed.

## Environment and workload

- Modal workspace/environment: `modal-labs/kailash-dev`; H100 80 GB only.
- Miles: `fdd808c008175d4895fd8c57c95ef4fa5bb6a2a0`.
- Megatron-LM: `f148a32b4385b758b66a77c9c3ad1641f1295d4b`.
- Megatron-Bridge: `8cd3466d14d2337c8492827b3712482c2b3e4866`.
- PyTorch: `2.13.0+cu130`; BF16 model parameters, FP32 gradient accumulation.
- Qwen3-8B checkpoint: `b968826d9c46dd6066d109eabc6255188de91218`.
- Qwen3.5-2B checkpoint: `15852e8c16360a2fea060d615a32b45270f8a8fc`.

Each case trains two rank-8 adapter slots with fixed synthetic cross-entropy
inputs, Adam at learning rate 1e-4, 12 updates, and a 768-token microbatch budget.
The target client has three sequences of lengths 127, 257, and 513, with their
order alternating. The neighbor contributes one to five sequences of changing
length. A fixed 257-token evaluation supplies the final mean NLL comparison.

Every trajectory starts with the same model-and-optimizer checkpoint round-trip,
including the reference: loading materializes Adam's initially lazy zero moments.
This prevents a representation-only mismatch between uninitialized and explicit
zero state from being mislabeled numerical drift.

The harness compares loss, every returned token logprob, gradient norm, and
per-rank SHA-256 hashes of every adapter tensor, gradient, and optimizer state
tensor. It checks same-process restore/repeat, fresh-process replay, a shared
work unit with a changing neighbor, and fresh-process continuation from step 6.
All 12 steps count; the continuation comparison covers steps 7–12.

## Strict default

The strict policy preserves the target's two canonical microbatches and runs
the neighbor in separate microbatches. Shared work units contained three or
four microbatches per rank, with zero mixed-client microbatches; the observed
schedule is recorded and checked rather than inferred from co-submission.

| Model | Layout | Same-process | Fresh process | Shared work unit | Midpoint continuation | Final NLL differences |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | TP1 / DP1 / CP1 | Exact | Exact | Exact | Exact | All 0 |
| Qwen3-8B | TP4 / DP1 / CP1 | Exact | Exact | Exact | Exact | All 0 |
| Qwen3.5-2B | TP1 / DP1 / CP1 | Exact | Exact | Exact | Exact | All 0 |

"Exact" includes every recorded loss, logprob, gradient, weight, optimizer state,
and gradient norm, not only rounded scalar metrics or a tolerance check.

Evidence: [Qwen3-8B TP1](https://modal.com/apps/modal-labs/kailash-dev/ap-SWUSSWvbFjU8IU6IHo25ns),
[Qwen3-8B TP4](https://modal.com/apps/modal-labs/kailash-dev/ap-9Kr7ETYy7EvS0dDEGU7kAs),
[Qwen3.5-2B](https://modal.com/apps/modal-labs/kailash-dev/ap-Uq37yqhZM9AbOUExFGZeYw).
Raw reports are generated under `scripts/results/deterministic-training/` as
`qwen3-8b-tp1-strict.json`, `qwen3-8b-tp4-strict.json`, and `qwen35-2b-strict.json`.

## Controls and the packing limit

The ordinary Qwen3-8B TP1 control starts from identical checkpoint state but
diverges during forward/backward execution:

| Comparison | First difference | Final NLL delta |
| --- | --- | ---: |
| Same-process restore/repeat | Step 1, forward | +0.000137809650 |
| Fresh-process replay | Step 1, gradients | +0.000634417924 |
| Shared work unit | Step 1, forward | +0.001320300863 |
| Midpoint continuation | Step 7, forward | +0.000970419279 |

[Ordinary control](https://modal.com/apps/modal-labs/kailash-dev/ap-mVNtT0qGYQwIMS7og8ODaC).
These are fixed-input replay differences, independent of TriMuon or rollout randomness.

The prototype's cross-client packing also fails despite deterministic attention,
FP64 reductions, canonical per-client boundaries, and corrected loss scaling:

| Model/layout | First difference | Final NLL delta |
| --- | --- | ---: |
| Qwen3-8B TP1 | Step 1, gradients | +0.000365789762 |
| Qwen3-8B TP4 | Step 1, gradients | −0.000152836514 |
| Qwen3.5-2B TP1 | Step 1, gradients | −0.001510234195 |

All three prototype runs pass their isolated repeat and midpoint-continuation
controls. The test verifies that the failing case actually mixed clients in
microbatches. This is why strict mode isolates client microbatches by default.
`deterministic_pack_clients=True` / `--pack-clients` retains the failing prototype
as an explicit experiment; it is not a bitwise-invariance promise.

## FA3 artifact and CPU checks

The hdim256 artifact was rebuilt against Spindle's current pinned base image,
from FlashAttention `8d3a3b80d4758ebde5a867c50d24d4351443cf2b` plus the retained patch.
Its installed binary hash is
`7015ab21c5287a546326f3fb4a8596ac88c66b99ebc8223bd24e86be209a4112`.
The guarded Transformer Engine patch verified this binary before enabling it.

- [18,624 fresh-forward/autograd checks](https://modal.com/apps/modal-labs/kailash-dev/ap-RVxIzyHJg0psYblJ7Mh7Tn)
  pass with bitwise equality of outputs, LSE, dQ, dK, and dV across repeated,
  dense, variable-length, and reordered packed execution, using FP16 and BF16.
- [110 reference/repeat/packing cases](https://modal.com/apps/modal-labs/kailash-dev/ap-5JwmyawC286HFWg93gS0oD)
  pass, including explicit FP32-reference accuracy comparisons.
- 282 focused local CPU tests pass; one existing test that attempts a Modal
  lookup was excluded locally. The complete GitHub CPU matrix passes for minimum,
  image-pinned, and latest Tinker SDKs at `c323b7c`.
- Both recipes validate offline; Ruff and source/wheel builds pass. The GDN
  prefix-state patch applies and Python-compiles against the pinned SGLang source.

## Practical limits

This establishes repeatability for these fixed-input, fixed-topology, rank-8
LoRA/Adam workloads. It does not certify every model, LoRA rank, context length,
loss function, GPU, or distributed layout. The bundled Qwen3.5-9B recipe was
validated offline; its full model was not run here. The harness calls the real
Spindle/Miles backend and actors directly, not the HTTP/control-plane scheduler.

TriMuon, full-parameter training, online RL trajectories, stochastic rollouts,
sampler/trainer parity, and model-quality convergence remain outside this result.
No throughput improvement is claimed: client isolation and FP64 reductions have
a cost, and the diagnostic tensor hashing adds synchronization overhead.

Cleanup was verified after completion: all 22 temporary validation/build apps
were stopped with zero remaining tasks. Existing shared apps were untouched.
