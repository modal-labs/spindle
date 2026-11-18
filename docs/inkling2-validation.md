# Inkling-2-Small validation

The full released checkpoint passed training and sampling on two separate eight-B300 nodes, with `exclude_reserved_capacity=True`. One node trained two attention and unembedding adapters; the other served them. This checks adapter mechanics on text, not convergence.

## Versions

- Model: `thinkingmachines/Inkling-2-Small`, revision `1df2d97d45210d94f781f42a6521392e5a539e46`.
- Spindle runtime: `3e0139d56b726eef8c860e639c0518150dab6bd1`.
- Miles runtime: `e186d97710dcc319da659f8bc25dbb674e227909` from [Miles PR 45](https://github.com/modal-projects/miles/pull/45), based on `spindle-miles-v0.2.0`.
- Release image: `radixark/miles:inkling-2-small@sha256:a2e3c49d5a266d9374d67ca4b706767b7bdf7d798958dd51520f61c29597e9a5`.
- Trainer image: `miles-v0.2.0-e186d97-inkling2-attn`. Sampler image: `sglang-inkling2-a2e3c49-head`.
- Torch: `2.14.1+cu130`. Megatron: `0.19.0+bea0ed74a`.
- Stock Tinker SDK: `0.26.0` for the two-client check.

The `inkling2-small-lora-8k-b300` recipe uses rank 32 and two slots. Training uses BF16, TP=CP=PP=1 and EP=8. Sampling uses the checkpoint's NVFP4/MXFP8 precision, TP=EP=8, and one adapter per batch. Experts and attention sinks stay frozen.

The sampler selects per-parameter precision metadata instead of the embedded legacy ModelOpt tag. Full attention target paths identify fused projections. A small sampler patch maps the HF unembedding target to `lm_head` during compatibility checks; checkpoint tensors retain their HF names.

## Full-model results

The [stock SDK check](../tests/manual/inkling2_full_e2e.py) created two clients, updated A twice, and published and sampled both adapters. It saved A's weights and optimizer, updated A again, then restored the exact saved loss. It also updated and served B independently.

| Check | Mean loss or logprob change |
| --- | ---: |
| Initial training loss | 2.8942715327 |
| A after two updates | 2.6545491219 |
| B after those A updates | 2.8942715327 |
| A after another update | 2.4234318733 |
| A after weights and optimizer restore | 2.6545491219 |
| Maximum A serving logprob change | 0.8667951226 nats |
| Frozen B serving change | Less than 0.00001 nats |

An 8,192-token forward/backward, optimizer step, publication, and adapter sample passed. Sampling from that adapter with an 8,185-token prompt returned five tokens with finite logprobs. The pinned sampler requires prompts strictly shorter than its stated 8,186-token bound.

[Machine-readable SDK results](assets/validation/inkling2-small-e2e.json) retain stage timings and numerical results without deployment credentials or checkpoint identifiers.

## Stock cookbook

Unmodified `tinker-cookbook==0.7.0` with stock `tinker==0.39.1` restored the attention-only checkpoint and completed one `recipes.chat_sl.train` step on a conversation dataset. It processed 32 tokens, reported finite mean NLL `3.4183804989`, and saved final training and sampler checkpoints. The client used Torch `2.14.1+cpu`, `tml-renderers==0.4.2`, and the repository's existing authentication helper to supply Modal proxy headers.

Use `load_checkpoint_path` for this workflow, with `renderer_name=tml_v0`. The cookbook CLI creates MLP adapters by default; loading an attention-only checkpoint preserves the supported target selection. This check does not validate a fresh cookbook run with its default MLP targets. The Inkling renderer requires Torch 2.11 or newer on the client.

## Observed times

These are individual SDK calls including request overhead, with compilation caches warm. They are not before/after performance measurements.

| Operation | Seconds |
| --- | ---: |
| Short backward | 2.62 |
| Short optimizer step | 0.49 |
| Adapter publication | 7.57 |
| Adapter sample, up to 16 tokens | 3.23 |
| Save weights and optimizer | 6.95 |
| Restore weights and optimizer | 4.02 |
| 8,192-token backward | 29.64 |
| 8,192-token optimizer step | 0.47 |
| Long-prompt sample | 0.94 |

Cold costs are substantial. Earlier runs took about 3,046 seconds to download the checkpoint, 3,054 seconds to start a trainer, and 1,218 seconds to start the corrected sampler. The first short backward took 327.61 seconds and the first 8K backward took 716.61 seconds including compilation. These cold measurements came from the earlier full-model run, before the final CPU thread limits and merged runtime.

## Other checks and limits

After the GPU run, main through `0d8f611` was merged to resolve a changelog conflict. Those later profiling, scoring, deployment typing, and usage changes have CPU coverage; the full-model run remains tied to `3e0139d`.

The Spindle CPU suite passed 1,572 tests and skipped 11 on Python 3.12.16. All pre-commit checks passed. SDK compatibility CI passed with minimum, image-pinned, and latest SDK versions in [run 44211487616](https://github.com/modal-labs/spindle/actions/runs/44211487616).

The merged Miles PR head `d23367105d09833dad3c91f490c90aefa18b80c2` separately passed 251 native-LoRA tests and 10 Inkling launcher snapshots in the matched image on B300. Full-model results apply to the frozen `e186d977` runtime, not that newer merged head. The full launcher suites had the same 23 failures and four errors on that head and its unchanged `e8142459` base, with no new failing cases (679 versus 669 passing cases). The full launcher suite is not green. Separate CUDA checks cover mixed-slot forward and gradient parity.

B200, trainer/sampler colocation, image inputs, engine restart, convergence, and full-model mixed-client batches were not validated. Training and serving used two resident clients with sequential requests. Expert adapters are unsupported by this recipe.

The temporary deployment and keepalive were stopped after validation. A container listing confirmed no remaining validation GPU containers. Shared model assets and saved checkpoints were retained.
