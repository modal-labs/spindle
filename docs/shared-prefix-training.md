# Experimental shared-prefix training

Shared-prefix training is disabled by default. Set
`miles_cfg.shared_prefix_training = True` and explicitly select an image built
with `spindle.providers.modal.shared_prefix_image.with_shared_prefix(base)`.
The provided `glm53-flash-lora-shared-prefix-b300` config selects that image
and excludes reserved capacity. Existing model configs and images keep their
current behavior. This is an experimental feature, not a default rollout.

The supported Miles integration currently covers GLM-5.3 Flash's native
KDA/DSA provider with one node, one adapter slot, CP=PP=ETP=1, THD packing,
zero dropout, and no routing replay or MTP. Other provider specifications fail
at startup. It does not imply that the separate Transformers 9B/27B experiments
are integrated into the Miles actor.

The common Miles package owns tree planning, adapter-isolated row ownership,
original-occurrence restoration, and the shape-stable kernel policy. The
architecture adapter owns KDA state/convolution sharing and DSA ancestor
indexing. Spindle selects the actor and retains native loss and optimizer
execution. There is no model-name dispatch in the trainer integration.

The opt-in policy enables native deterministic mode, uses explicit RMS math
and common fixed-row LoRA/frozen-expert kernels, disables fused MoE routing
and mutable router bias, and applies the same policy to native reference and
shared execution. These numerical controls can cost throughput. Before **every**
forward/backward call, the actor compares expanded and compact forward results
with `atol=rtol=1e-4`. Any rank's failure aborts all ranks before backward.
There is no switch to bypass this check in this initial integration.
A forward check does **not** establish gradient or optimizer equivalence.

The configured 49,152-token budget splits native dynamic microbatches while
preserving the client's effective batch. Expanding original-occurrence
vocabulary logits remains memory-intensive; large trajectory lengths and
batches still need an appropriate token budget.

## Measured evidence

The corrected full GLM SFT diagnostic measured maximum-rank gradient relative
L2 of **0.7343%** for compact sharing, **0.6965%** for expanded repeat, and
**0.5862%** for the tree adapter without sharing. That comparison used the
corrected kernel policy at one checkpoint/microstep; it does not establish a
statistical tolerance. The separate 9B/27B runs retained approximately 1.1%
gradient error against exact repeats. See the detailed record in
[Miles #35](https://github.com/modal-projects/miles/pull/35).

The subsequent [async SWE-Gym run](https://modal.com/apps/modal-labs/kailash-dev/ap-2bJXRM4hkdUnA9U16WQicL)
completed three 64-trajectory LoRA updates and published three checkpoints;
updated adapters generated real subsequent rollouts. It reused 64 exact graded
initial-policy trajectories from an attempt that had made zero optimizer
updates, then trained on 128 fresh selected trajectories. All eight ranks
passed the unchanged forward tolerance before each update. Rank 0 logged
zero maximum and relative-L2 forward error for all three checks. A separate
corrected full-model replay matched forward outputs exactly on all eight ranks.
**No new gradient-equivalence measurement was made in this RL run.**

Warm shared forward/backward compute was 1,587 and 1,320 original input tokens/s
across eight B300 trainer GPUs, excluding the parity checks, optimizer and I/O.
Rank-0 row savings were approximately 11% and 5%. Paired forward-only times were
43.3/41.9 seconds and 66.4/89.1 seconds (expanded/shared). A matched unshared full
training baseline was **not** run. These data establish neither a consistent
speedup nor a convergence/reward improvement.

Earlier long-document experiments showed substantial potential savings, but
these timings must not be advertised as a speed guarantee for this corrected
policy. Long shared prefixes are a prerequisite for substantial savings, not
sufficient evidence to ship by default.

The successful GPU experiment predates promotion into this clean Spindle
integration and used its frozen image/source manifest. This PR has focused CPU
regression coverage; the new image/config combination still needs a full GPU
acceptance run before removing the experimental gate.
