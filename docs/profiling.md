# Profiling and tuning a deployment

How to find where your training loop spends its time and make it faster by
changing your deployment config. Work in this order:

1. Measure where a step's time goes, with the timings Spindle already reports.
2. Trace the slow part.
3. Decide what limits it: overhead, memory, compute or communication.
4. Change one config field, measure again, and keep the change only if it helps.

Prefer config changes. If the fix needs code, see
[When the fix is not a config change](#when-the-fix-is-not-a-config-change).

## 1. Measure the step

Time each part of one step in your own script, around the stock Tinker SDK calls:
sampling (all `sample` calls for a batch, including tools and the slowest
trajectory), training (`forward_backward` and `optim_step`), and publishing
(`save_weights_for_sampler` or `save_weights_and_get_sampling_client`).

Every Miles LoRA `optim_step` result includes `timing/*` metrics for the step:

| Metric | Meaning |
| --- | --- |
| `timing/forward_backward_s`, `timing/optim_step_s`, `timing/trainer_step_s` | Trainer time in each call, and their sum. |
| `timing/prepare_batch_s`, `timing/build_outputs_s` | Converting your datums for the GPUs, and building the results. |
| `timing/idle_wait_s` | Trainer time spent waiting between your calls. |
| `timing/save_sampler_weights_s`, `timing/publish_weights_s` | Sampler-weight export and publish, reported one step late. |
| `timing/<phase>_calls` | Calls per phase. |

The trainer log (`modal app logs <frontend>-trainer-<config>`) has the same
numbers as `{"event": "spindle_step_timing", "phase": ..., "step": ..., "seconds": ...}`
lines, and `spindle_memory` lines with `max_allocated_gb`.

Your client's training time minus `timing/trainer_step_s` is time outside the
trainer. If it is large, see [Time outside the trainer](#time-outside-the-trainer).

Expected compute time: about 4 × parameters × tokens FLOPs per step for LoRA
(6× for full fine-tuning), divided by the GPU's dense BF16 peak (about 1 PFLOP/s
for H100 or H200, 2.2 for B200). Training usually reaches 15–40% of peak. A step
much slower than that is not limited by compute.

Read a field's current value before changing it:

```bash
uv run python -c "from spindle.configs.qwen35_4b_lora_32k import config; print(config.trainer_gpu, config.trainer_gpus_per_node); print(config.miles_cfg); print(config.sglang_cfg)"
```

## 2. Capture a trainer trace

Spindle can record a [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html)
trace of one whole training step on the Miles LoRA trainer's GPUs: every
`forward_backward` call for the step, the `optim_step` and the LoRA export,
including idle gaps between your calls. It doesn't cover sampling, your script,
or sampler-weight save and publish (use `timing/*` for those). Off by default.

```bash
SPINDLE_TORCH_PROFILE_STEP=2 uv run spindle deploy deployment.py --refresh-trainer <config name>
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPINDLE_TORCH_PROFILE_STEP` | unset (off) | Step to trace. Steps count prior `optim_step` calls from 0; avoid 0, which includes kernel compilation. |
| `SPINDLE_TORCH_PROFILE_RANKS` | `0` | Ranks to trace, such as `0,4` or `all`. About 250 MB each. |
| `SPINDLE_TORCH_PROFILE_DIR` | `/checkpoints/torch-profile/<config name>` | Output directory on the checkpoint volume. |

- A running trainer keeps its old settings until it idles out. To be sure the
  trace is taken, profile a [variant](deployment-configs.md#variants) with its
  own name. To turn profiling off, deploy without the variable and with
  `--refresh-trainer`.
- Recording starts at the step's first `forward_backward` and stops when its
  sampler weights are published or the next step starts. On a shared trainer,
  the step belongs to the first client to reach it.
- Phases are labelled `spindle/forward_backward`, `spindle/forward_only`,
  `spindle/optim_step` and `spindle/export_slot_peft`.
- Files: `rank<N>.trace.json.gz` (timeline) and `rank<N>.key_averages.txt`
  (top 60 operators by GPU time).
- Writing the trace takes minutes and delays the next update. Leave the profiled
  step and the next one out of timing comparisons.

Download from the checkpoint volume (`platform.storage.checkpoints`, default
`spindle-checkpoints`):

```bash
uv run modal volume get <checkpoint volume> /torch-profile/<config name>/rank0.trace.json.gz .
```

Open it at <https://ui.perfetto.dev>. CPU threads are on top, GPU streams are
the `stream N` rows. Search `spindle/` to jump between phases. For numbers, use
the DuckDB queries in the `spindle-experiments` skill (phase durations, top
kernels, GPU busy time, NCCL share). `scripts/upload_torch_profile.py` attaches
a trace to a W&B run.

## 3. Decide what limits the step

Start with `spindle/forward_backward` on rank 0. Thresholds are rough. "Idle"
means no kernel on any GPU stream; the skill's busy-fraction query sums streams,
so it can exceed 1.

| Limit | What the trace shows | Config to try |
| --- | --- | --- |
| **Overhead** (CPU, Python, launches) | GPU idle for more than ~20% of the phase in many short gaps; kernels under ~20 µs; CPU in dispatch or `cudaStreamSynchronize`. | Raise `miles_cfg.max_tokens_per_gpu` for fewer, larger micro-batches. Send the batch in one `forward_backward` (check `timing/forward_backward_calls`). Check for fused kernels turned off in `miles_cfg.cli_options` (`apply_rope_fusion`, `bias_swiglu_fusion`, `moe_permute_fusion`); turn one back on only with a parity check. The trainer has no `torch.compile` or CUDA graph option: packed batches change shape every call. |
| **Memory capacity** | Out-of-memory errors, or `max_allocated_gb` near the GPU's memory. | Turn on recompute (`miles_cfg.cli_options.recompute_granularity: "full"`, `recompute_method: "uniform"`, `recompute_num_layers: 1`). Lower `miles_cfg.cli_options.log_probs_chunk_size`. For very long sequences, raise `miles_cfg.context_parallel_size`. Or use a GPU with more memory. |
| **Compute** | GPU busy over ~85%; top kernels are matrix multiplies (`gemm`, `cutlass`, `nvjet`) and attention (`flash`, and Triton `chunk_*` for Qwen3.5 linear attention). | Turn recompute off if memory allows. Then add data-parallel GPUs (`trainer_gpus_per_node`) or a faster GPU. |
| **Memory bandwidth** | GPU busy, but top kernels are elementwise, softmax, cross-entropy, norm or copies. | If kernels are also short, treat as overhead. Otherwise it needs kernel work. |
| **Communication** | `ncclKernel_*` over ~15% of GPU time while compute streams idle. | Lower `tensor_model_parallel_size` (and `expert_model_parallel_size` for MoE) if the model fits, so more GPUs do data parallelism. |
| **A slow rank** | With all ranks traced, one finishes `forward_backward` late and the others wait in their first collective. | `miles_cfg.cli_options.balance_data: True`. |

### Rules of thumb

Several follow [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198).

- **Prefer data parallelism.** Use the smallest tensor and context parallel sizes
  that fit, often 1 for LoRA on models up to ~30B. Tensor parallelism
  communicates in every layer and splits matrix multiplies into less efficient
  pieces. Data parallelism exchanges LoRA gradients, which are small, once per
  step.
- **Use tensor or context parallelism only to fit memory**, within one machine.
  Spindle enables sequence parallelism whenever tensor parallelism is above 1.
  Context parallelism costs extra in attention; use it only when one sequence
  doesn't fit.
- **Recompute as little as possible.** Full recompute reruns each layer's
  forward: about +33% compute for full fine-tuning, +50% for LoRA. If you need
  some, `recompute_method: "block"` with `recompute_num_layers: N` recomputes
  only the first N layers; raise N until the step fits. Selective recompute
  saves little here because flash attention already doesn't store attention
  scores.
- **Estimate activation memory before raising `max_tokens_per_gpu`.** Without
  recompute, about 34 bytes × hidden size × layers per token: ~2.8 MB per token
  for Qwen3.5-4B (2,560 × 32), ~23 GB for 8,192 tokens. Add base weights
  (2 bytes per parameter) and logits (4 bytes × vocabulary per token in each
  `log_probs_chunk_size` chunk), then compare with `max_allocated_gb`.
- **Balance data-parallel ranks.** Every rank waits for the slowest. LoRA
  recipes: `miles_cfg.cli_options.balance_data: True` splits rows by token
  count, and `miles_cfg.dynamic_batch_size` (on by default) packs several rows
  per micro-batch. Full fine-tuning balances rows automatically; check
  `packing_padding_fraction` and `packing_utilization` in `forward_backward`
  metrics. One row much longer than the rest still leaves ranks idle.
- **Spend spare memory on bigger micro-batches.** It is usually the largest win
  for an overhead-limited step.
- **Scale GPUs only once one GPU is busy.** Measure 1, 2 and more GPUs and fit
  step time ≈ a + b / GPUs. `a` is per-step overhead that more GPUs won't
  reduce. Reduce it first, and pick the GPU count by cost per epoch.
- **Retest dropped changes after a big win.** A change that did nothing while
  overhead-limited can help once compute-limited.

To remove recompute settings a bundled config sets, replace its `cli_options`
without them:

```python
from spindle.configs.qwen35_9b_lora_16k import Config as Parent
from spindle.configs.qwen35_9b_lora_16k import config as parent


class Config(Parent):
    name = "my-9b-no-recompute"
    overrides = {
        "miles_cfg.cli_options": {
            key: value
            for key, value in parent.miles_cfg["cli_options"].items()
            if not key.startswith("recompute")
        }
    }


config = Config()
```

### Time outside the trainer

If client training time is well above `timing/trainer_step_s`:

- **Queueing.** `GET /api/v1/spindle/queues` shows a shared trainer's backlog.
  For RL, set `trainer_max_clients_per_batch = 1` so each client gets its update
  as soon as its own step finishes.
- **Many small calls.** Send one `forward_backward` per batch, and submit
  `optim_step` without waiting.
- **Distance.** Run your script in the deployment's region.
- **Request timing.** Deploy with `SPINDLE_REQUEST_TIMING=1` for
  `spindle_request_mark` lines (upload, decode, result delivery) in the frontend
  and trainer logs.

## 4. Tune the sampler

Signals:

- **Sampler logs** (`modal app logs <frontend>-lora-<config>`): SGLang decode
  lines with `#running-req`, `#queue-req`, `token usage` (KV cache in use) and
  `gen throughput`.
- **Startup marks:** `sglang_healthy` and `first_adapter_loaded` mid-round mean
  you waited on cold starts.
- **Usage:** `spindle usage` reports `cold_start` and `idle` cost
  ([cost-attribution.md](cost-attribution.md)).
- **Tracing:** with OpenTelemetry export, `spindle.sample` spans carry
  `sglang.queue_s`, `sglang.prefill_s`, `cached_tokens` and `prompt_tokens`
  ([observability.md](observability.md)).

Compare `gen throughput` with `#running-req` over a round. Throughput rising with
running requests means each replica needs more requests at once. Throughput
flat while requests wait means you need more replicas. `token usage` near 1
means the KV cache is the limit.

| Limit | Signs | Config to try |
| --- | --- | --- |
| **Cold starts** | Mean time per turn far above the median; replicas start mid-round. | `inference_min_replicas` at what one round needs, or `inference_scaledown_window_s` longer than your training step. |
| **Capacity** | `#queue-req` above 0, `token usage` above 0.9, or 503s. | Raise `inference_max_replicas`. Lower `inference_target_concurrency` to scale out earlier. Raise `sglang_cfg.max_queued_requests` if requests are rejected. |
| **KV cache** | `token usage` near 1. | Raise `sglang_cfg.mem_fraction_static`, or give each replica more GPUs (`inference_gpus_per_node` and `sglang_cfg.tp_size`, set equal). |
| **Too little batching** | `#running-req` well below `sglang_cfg.max_running_requests` while your loop waits. | Send a round's requests at once. Raise `max_running_requests` if memory allows. |
| **Decode overhead** | Throughput per request drops once more requests run than CUDA graphs cover. | Keep `sglang_cfg.cuda_graph_max_bs_decode` ≥ `max_running_requests`. For small models at small batch sizes, try `sglang_cfg.enable_torch_compile: True` (slower replica start). |
| **Prefill** | Long prompts, short outputs, low `cached_tokens` share. | `sglang_cfg.enable_mixed_chunk: True`. Spindle already routes each conversation to one replica. |
| **Many adapters** | Throughput drops as LoRA clients are added. | `sglang_cfg.max_loras_per_batch` ≥ adapters sampling at once. |
| **Speculative decoding** | Recipe sets `sglang_cfg.speculative_algorithm`; large batches or high temperature. | Compare a variant without the `speculative_*` keys (same pattern as the recompute example). |
| **Slowest trajectories** | Round lasts far longer than the p90 trajectory; sampler idle at the end. | Not a sampler setting. Start the next round early, or cap turns and output length. |
| **Tools** | Tool calls take much of each trajectory. | Not a Spindle setting. Speed up or parallelize tools. |

## 5. Compare before and after

- One change per named [variant](deployment-configs.md#variants).
- Compare the median of 10 or more steady-state steps, excluding warm-up, the
  profiled step and the step after it.
- Pin the GPU type with `!` (`trainer_gpu = "H100!"`); Modal can otherwise place
  `H100` on an H200.
- Repeat surprising results once; machines differ by 5–10%.
- Check parity: per-token logprobs and loss on a fixed batch should match.
- For RL with long steps, tune the sampler with a rollout-only benchmark: save
  sampler weights once, replay the same prompts with fixed seeds, run one copy
  per training run, and compare round time and seconds per 1k generated tokens.
  Confirm with a few real steps.
- To move a running job to a variant: `save_state`, then `load_state` in a
  training client created on the variant.
- Stop when changes save less than a few percent, or what remains needs code.

If your config beats a bundled recipe for a common workload, open a pull request
updating the recipe with before and after step times.

## Past tuning changes

Spindle pull requests that fixed a measured bottleneck. Their descriptions give
before and after times; their diffs show the fields. Read the closest one first.

| Pull request | Bottleneck | Change |
| --- | --- | --- |
| [#36](https://github.com/modal-labs/spindle/pull/36) | Tensor and expert parallel communication | Data parallelism on the same GPUs (20B MoE) |
| [#40](https://github.com/modal-labs/spindle/pull/40) | Uneven tokens per rank | `balance_data` |
| [#150](https://github.com/modal-labs/spindle/pull/150), [#163](https://github.com/modal-labs/spindle/pull/163) | More tensor parallelism than memory needs | Less tensor parallelism, per GPU type |
| [#193](https://github.com/modal-labs/spindle/pull/193) | One sequence per micro-batch | Several per micro-batch |
| [#203](https://github.com/modal-labs/spindle/pull/203), [#247](https://github.com/modal-labs/spindle/pull/247) | Out of memory on packed batches | Token budget per micro-batch |
| [#246](https://github.com/modal-labs/spindle/pull/246) | Expert-parallel communication | All experts on every GPU |
| [#327](https://github.com/modal-labs/spindle/pull/327), [#337](https://github.com/modal-labs/spindle/pull/337), [#338](https://github.com/modal-labs/spindle/pull/338) | Full fine-tuning layout | Data parallelism, larger micro-batches, B200/B300 |
| [#37](https://github.com/modal-labs/spindle/pull/37) | Sampler scaled to zero during training | Warm pool between steps |
| [#45](https://github.com/modal-labs/spindle/pull/45), [#236](https://github.com/modal-labs/spindle/pull/236), [#243](https://github.com/modal-labs/spindle/pull/243) | Rejected or queued requests | Higher running and queued limits |
| [#49](https://github.com/modal-labs/spindle/pull/49) | Prefill pausing decode | `enable_mixed_chunk` |
| [#237](https://github.com/modal-labs/spindle/pull/237) | Small KV cache | More KV cache memory |
| [#241](https://github.com/modal-labs/spindle/pull/241) | Replicas added for short bursts | Scale out only when all replicas are full |
| [#254](https://github.com/modal-labs/spindle/pull/254) | Decode speed on Hopper | B200 samplers with Hopper fallback |
| [#165](https://github.com/modal-labs/spindle/pull/165), [#323](https://github.com/modal-labs/spindle/pull/323) | Long decodes | Speculative decoding with built-in draft layers |

Code changes, as examples of a non-config fix:

| Pull request | Bottleneck | Change |
| --- | --- | --- |
| [#42](https://github.com/modal-labs/spindle/pull/42), [#152](https://github.com/modal-labs/spindle/pull/152), [#153](https://github.com/modal-labs/spindle/pull/153) | Time outside the trainer | Binary transport, one frontend region |
| [#158](https://github.com/modal-labs/spindle/pull/158) | Host time moving rows between processes | Arrays instead of Python lists |
| [#157](https://github.com/modal-labs/spindle/pull/157), [#317](https://github.com/modal-labs/spindle/pull/317), [#322](https://github.com/modal-labs/spindle/pull/322) | Vocabulary softmax memory and time | One softmax and logits gradient per micro-batch |
| [#333](https://github.com/modal-labs/spindle/pull/333), [#334](https://github.com/modal-labs/spindle/pull/334) | Recompiling kernels per batch length | Compile once |
| [#336](https://github.com/modal-labs/spindle/pull/336) | Unbalanced ranks, full-size padding | Split rows by length, short padding |
| [#124](https://github.com/modal-labs/spindle/pull/124) | LoRA clients blocking each other | One client at a time, short operations first |
| [#46](https://github.com/modal-labs/spindle/pull/46) | Prefix cache misses across turns | One replica per conversation |

## When the fix is not a config change

Keep the best config meanwhile, then:

1. Open an issue on [modal-labs/spindle](https://github.com/modal-labs/spindle/issues)
   with your config, step-time table and the relevant trace excerpt.
2. Say where the fix belongs: Spindle (request path, scheduling), Miles
   (training execution) or SGLang (sampling). Link any matching upstream issue;
   Spindle keeps its forks close to upstream.
3. Optionally contribute: test a patched backend by pointing `trainer_image` or
   `inference_image` at your own `modal.Image` (see [Images](deployment-configs.md#images)),
   then open a pull request with before and after times for maintainers to
   review.

## Where a cold start spends its time

Each container prints one JSON line per startup phase, with no configuration:

```json
{"event": "spindle_startup_mark", "component": "trainer-backend", "mark": "trainer_actors_initialized", "ts": 1792012345.1, "since_start_s": 212.4, "phase_s": 148.9}
```

`since_start_s` is time since the container's process started. `phase_s` is the
cost of the phase ending at this mark.

| Mark (component) | Phase that ends here |
| --- | --- |
| `trainer_function_entered` (trainer) | Container boot and Modal runtime start. |
| `assets_reloaded` / `ray_cluster_joined` (trainer) | Volume reload, or Ray cluster formation for multi-node trainers. |
| `backend_spawned`, `engine_record_starting` (trainer) | Backend process launched. `queue_s` is the wait from the control plane's request to the process starting. |
| `backend_imports_done` (trainer-backend) | Importing Torch, Megatron and Miles. |
| `miles_args_parsed`, `ray_started`, `ray_workers_launched` (trainer-backend) | Ray startup and per-GPU workers. |
| `actor_model_loaded` (trainer-rank0) | Model built and base weights loaded. |
| `trainer_actors_initialized`, `miles_backend_ready` (trainer-backend) | All ranks loaded. |
| `backend_ready`, `engine_serving` (trainer) | Engine reachable; `since_spawn_s` is the total from request to ready. |
| `first_slot_loaded`, `first_forward_backward_done`, `first_optim_step_done`, `first_sampler_export_done` (trainer-backend) | First client's work, including kernel compilation; `duration_s` is the call. |

Sampler replicas print `server_enter`, `sglang_healthy` (base model loaded, CUDA
graphs captured), `sidecar_healthy` and `first_adapter_loaded`. Pool deployment
prints `pool_deployed`, and the model download prints `assets_ready`. Trainer
instance records also keep `spawned_at`, `started_at`, `ready_at` and the phase
breakdown.

```bash
uv run modal app logs spindle-trainer-<config> \
  | grep -o '{"event": "spindle_startup_mark".*}' \
  | jq -r '[.component, .mark, .since_start_s, .phase_s] | @tsv'
```
