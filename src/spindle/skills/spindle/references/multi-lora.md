# Multi-LoRA

- `service.create_lora_training_client(base_model, rank)`: each client gets its own adapter, gradients, optimizer state. Ranks can differ per client, up to 32; alpha and target modules are deployment-side (SDK defaults `train_attn`/`train_mlp`/`train_unembed` true).
- Clients sharing a definition may share an engine depending on slots/deployment limits.
- Engine batches `forward_backward` by compatible `loss_fn` + `loss_fn_config` up to a token budget: large batches interleave per client, small batches pack into shared passes. Greedy scheduler, no fairness guarantee.
- Keep a steady async stream of `forward_backward`; submit futures before awaiting. Lockstep sync clients cause utilization bursts and troughs.
- Pricing: `GPU $/output token = GPU $/s ÷ output TPS`. Raising useful TPS is the only lever.
- Routing: all `num_samples` of one request go to one replica for prefix reuse. Multi-turn: set `cache_affinity_key` (≤256 chars, e.g. per trajectory) so each turn hits the same replica; absent it, the route derives from the prompt after the first reply.
- `target_tokens` accepts shape `[seq]` or `[seq, choices]`; logprobs return in the same shape and weights/advantages/logprobs must match. Dense and CSR both work. Scoring cost grows with `num_choices`. Megatron backends reject 2D targets.
- Deterministic training (experimental): use `qwen3-8b-lora-32k-deterministic` or `qwen35-9b-lora-16k-deterministic`. The Qwen3.5 recipe requires `SPINDLE_MILES_FA3_ARTIFACTS`. Requires Hopper SM90, DP1/CP1, and fixed topology and inputs; keep the default separate client microbatches. See `docs/deterministic-training.md` for setup.
