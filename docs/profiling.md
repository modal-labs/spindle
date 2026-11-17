# Profiling and tuning a deployment

This guide shows how to find where your training loop spends its time and how to
make it faster by changing your deployment config. It covers the trainer and the
sampler. The bundled configs are tuned for common workloads, so a few config
changes often matter more for your workload than any code change.

Work in this order:

1. Measure where a step's time goes, with the timings Spindle already logs.
2. Trace the slow part.
3. Decide what limits it: overhead, compute, memory or communication.
4. Change one config field, measure again, and keep the change only if it helps.

If the fix turns out to need a change inside Spindle, Miles or SGLang rather than
a config change, see [When the fix is not a config change](#when-the-fix-is-not-a-config-change).

## 1. Measure the step first

Before you trace anything, split one step of your training loop into its parts.
Time these in your own script, around the stock Tinker SDK calls:

- **Sampling**: all `sample` calls for one batch, including tool calls and the
  wait for the slowest trajectory.
- **Training**: `forward_backward` and `optim_step`.
- **Publishing**: `save_weights_and_get_sampling_client` or `save_weights_for_sampler`.

Then compare the training part with what the trainer reports. Every Miles LoRA
`optim_step` result includes `timing/*` metrics for that step:

| Metric | Meaning |
| --- | --- |
| `timing/forward_backward_s`, `timing/optim_step_s` | Time the trainer spent in each call. |
| `timing/trainer_step_s` | The two above added together. |
| `timing/prepare_batch_s`, `timing/build_outputs_s` | Converting your datums for the GPUs, and building the results you get back. |
| `timing/idle_wait_s` | Time the trainer waited between your calls. |
| `timing/save_sampler_weights_s`, `timing/publish_weights_s` | Exporting and publishing sampler weights. These are reported one step late. |
| `timing/<phase>_calls` | How many calls each phase took. |

The trainer also prints one line per phase to its log, which you can read with
`modal app logs <frontend>-trainer-<config>`:

```json
{"event": "spindle_step_timing", "phase": "forward_backward", "step": 12, "seconds": 4.81, "model_id": "..."}
```

Your client's training time minus `timing/trainer_step_s` is time spent outside
the trainer: uploading datums, queueing behind other clients, and downloading
results. If that gap is large, a trace of the GPUs will not explain it. See
[Time outside the trainer](#time-outside-the-trainer).

A rough expected compute time helps you judge whether a step is slow. A forward
and backward pass costs about 4 × parameters × tokens FLOPs with LoRA, and
6 × parameters × tokens with full fine-tuning. Divide by the GPU's dense BF16
peak, about 1 PFLOP/s for an H100 or H200 and about 2.2 PFLOP/s for a B200, to
get the time at 100% utilization. Training usually runs at 15–40% of that. A
step far slower than that range is probably not limited by compute.

Before you change a field, read its current value. Import your config, or a
bundled one, and print its backend settings:

```bash
uv run python -c "from spindle.configs.qwen35_4b_lora_32k import config; print(config.trainer_gpu, config.trainer_gpus_per_node); print(config.miles_cfg); print(config.sglang_cfg)"
```

## 2. Capture a trainer trace

Spindle can record a [`torch.profiler`](https://pytorch.org/docs/stable/profiler.html)
trace of one whole training step on the Miles LoRA trainer's GPUs: every
`forward_backward` call for that step, the `optim_step`, and the weight export
that follows. The trace shows what the GPU and CPU did during that step,
including idle gaps between your calls. It doesn't cover sampling, your own
script, or saving and publishing sampler weights; use the `timing/*` metrics
for those. Profiling is off by default and costs nothing until you turn it on.

Set the step to trace in the shell you deploy from, and rebuild the trainer:

```bash
SPINDLE_TORCH_PROFILE_STEP=2 uv run spindle deploy deployment.py --refresh-trainer <config name>
```

A trainer that is already running keeps its old settings until it shuts down
after going idle. To be sure the trace is taken, profile a variant of your config
with its own name (see [Variants](deployment-configs.md#variants)), which always
starts a fresh trainer. To turn profiling off, deploy again without the variable
and with `--refresh-trainer`.

A step is numbered by how many `optim_step` calls came before it, counting from
0, so `2` traces the third training step. Avoid step 0, which includes one-time
work such as compiling kernels.

| Variable | Default | What it does |
| --- | --- | --- |
| `SPINDLE_TORCH_PROFILE_STEP` | unset (off) | Which step to trace. Setting it turns profiling on. |
| `SPINDLE_TORCH_PROFILE_RANKS` | `0` | Which trainer GPUs (ranks) to trace, such as `0,4` or `all`. Each trace is about 250 MB. |
| `SPINDLE_TORCH_PROFILE_DIR` | `/checkpoints/torch-profile/<config name>` | Where traces are written, on the checkpoint volume. |

What you get:

- **One training step per trainer process.** Recording starts at the first
  `forward_backward` of the chosen step. It stops once that step's sampler
  weights are published, or when the next step's first `forward_backward`
  arrives, whichever comes first. On a trainer shared by several clients, the
  step belongs to whichever client reaches it first.
- **Labelled phases.** On each GPU rank: `spindle/forward_backward`,
  `spindle/forward_only`, `spindle/optim_step` and `spindle/export_slot_peft`.
- **Files:** `rank<N>.trace.json.gz` (the timeline) and `rank<N>.key_averages.txt`
  (the top 60 operators by GPU time, readable without a UI).

Writing a trace takes several minutes and delays the next update. Leave the
profiled step and the step after it out of any timing comparison.

Download a trace from your checkpoint volume. Its name is
`platform.storage.checkpoints` in your config, `spindle-checkpoints` by default:

```bash
uv run modal volume get <checkpoint volume> /torch-profile/<config name>/rank0.trace.json.gz .
```

You can also attach it to a W&B run with
`uv run scripts/upload_torch_profile.py --dir <trace dir> --entity <entity> --project <project> --run-id <run id>`.

Open <https://ui.perfetto.dev> and load the `.json.gz` file directly. The top
rows are CPU threads. The rows named `stream N` are GPU streams, where kernels
run. Press `W`/`S` to zoom and `A`/`D` to pan, and search for `spindle/` to jump
between phases. Drag across a range to get a time breakdown of everything in it.

For numbers rather than a picture, query the trace with DuckDB. The
`spindle-experiments` skill has ready-made queries for phase durations, top
kernels, GPU busy time and communication share.

## 3. Decide what limits the step

Look at the `spindle/forward_backward` phase on rank 0 first, since it is
usually most of the step. Find the pattern below that matches best. The numbers
are rough guides, not hard cut-offs. "Idle" means no kernel running on any GPU
stream. The skill's busy-fraction query adds overlapping streams together, so
its result can go above 1 when streams run in parallel.

| What limits it | What the trace shows | Config to try |
| --- | --- | --- |
| **Overhead** (CPU, Python, kernel launches) | GPU streams are idle for more than about 20% of the phase, in many short gaps. Kernels are short, often under 20 µs. CPU rows are busy with dispatch or `cudaStreamSynchronize`. | Raise `miles_cfg.max_tokens_per_gpu` so each call runs fewer, larger micro-batches. Check `timing/forward_backward_calls`: many small calls per step cost more than one large one. Check whether the recipe turns off fused kernels in `miles_cfg.cli_options` (such as `apply_rope_fusion`, `bias_swiglu_fusion` or `moe_permute_fusion` set to `False`). Some recipes turn one off for a specific model, so turn it back on only with a parity check. The trainer doesn't offer `torch.compile` or CUDA graphs as config: packed batches change shape every call, so they would recompile or fall back. |
| **Memory capacity** | Out-of-memory errors, or you had to lower `max_tokens_per_gpu` to fit. The `spindle_memory` log line shows `max_allocated_gb` close to the GPU's memory. | Turn on recompute (`miles_cfg.cli_options.recompute_granularity: "full"`, `recompute_method: "uniform"`, `recompute_num_layers: 1`). Lower `miles_cfg.cli_options.log_probs_chunk_size`. Raise `miles_cfg.context_parallel_size` for very long sequences, or use a GPU with more memory. |
| **Compute** | GPUs are busy for more than about 85% of the phase, and the top kernels are matrix multiplies (`gemm`, `cutlass`, `nvjet`) and attention (`flash`, and for Qwen3.5's linear-attention layers, Triton kernels such as `chunk_*`). | Spend memory headroom on speed: turn recompute off if it is on. Then add GPUs for data parallelism (`trainer_gpus_per_node`), or use a faster GPU type. |
| **Memory bandwidth** | GPUs are busy, but the top kernels are elementwise, softmax, cross-entropy, norm or copy kernels rather than matrix multiplies. | Few config fixes. If the kernels are also short, treat it as overhead and raise `max_tokens_per_gpu`. Otherwise it usually needs kernel work; see [When the fix is not a config change](#when-the-fix-is-not-a-config-change). |
| **Communication** | `ncclKernel_*` kernels take more than about 15% of GPU time while compute streams sit idle. | Lower `tensor_model_parallel_size` (and `expert_model_parallel_size` for MoE models) if the model fits on fewer GPUs, so more GPUs do data parallelism instead. LoRA gradients are small, so data parallelism adds little communication. |
| **A slow rank** | Trace all ranks. One rank finishes `forward_backward` well after the others, and the rest wait in their first collective. | Set `miles_cfg.cli_options.balance_data: True` so ranks get similar token counts. |

### Rules of thumb for the trainer

These follow from how memory and work split across GPUs. Several come from
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198).

- **Prefer data parallelism.** Use the smallest `tensor_model_parallel_size` and
  `context_parallel_size` your model and longest sequence fit in, often 1 for
  models up to about 30B parameters with LoRA. Tensor parallelism adds
  communication inside every layer, in both the forward and backward pass, and
  splits each matrix multiply into smaller, less efficient pieces. Data
  parallelism only exchanges gradients once per step, and LoRA gradients are
  small.
- **Use tensor or context parallelism only to fit memory.** Keep it within one
  machine, where GPUs share fast links. Spindle turns on sequence parallelism
  whenever tensor parallelism is above 1, which spreads the remaining
  activations across those GPUs too. Context parallelism splits long sequences
  and pays for it in the attention layers, so use it only when one sequence
  doesn't fit.
- **Recompute as little as you can.** Full recompute (`recompute_granularity:
  "full"`) runs each layer's forward pass twice. That is about a third more
  compute for full fine-tuning and about half more for LoRA. Turn it off if
  memory allows. If the step doesn't fit without it, recompute only some layers:
  `recompute_method: "block"` with `recompute_num_layers: N` recomputes the
  first N layers and keeps the activations of the rest. Raise N until the step
  fits. The paper's cheaper "selective" recompute targets attention scores,
  which the flash-attention kernels Spindle uses already don't store, so it
  saves little here.
- **Estimate activation memory before you raise the token budget.** Without
  recompute, activations grow linearly with `max_tokens_per_gpu`, at roughly
  34 bytes × hidden size × layers per token. For Qwen3.5-4B (hidden size 2,560,
  32 layers) that is about 2.8 MB per token, or about 23 GB for 8,192 tokens.
  Add the base weights (2 bytes per parameter) and the logits (4 bytes ×
  vocabulary size per token in each `log_probs_chunk_size` chunk), then compare
  with `max_allocated_gb` in the `spindle_memory` log line.
- **Balance work across data-parallel GPUs.** Every GPU waits for the slowest
  one at the end of a step. On LoRA recipes, `miles_cfg.cli_options.balance_data:
  True` splits rows across GPUs by token count instead of in arrival order; the
  Qwen3.5 32K recipes already set it. Keep `miles_cfg.dynamic_batch_size` on
  (the default), so each GPU packs several rows into micro-batches of up to
  `max_tokens_per_gpu`; turned off, every row is its own micro-batch. Full
  fine-tuning recipes balance rows by length on their own. Their
  `forward_backward` results report `packing_padding_fraction` and
  `packing_utilization`, which show how much of each micro-batch was padding.
  Balancing can't fix a batch whose single longest row outweighs everything
  else, so mixing very long and very short rows in one step leaves GPUs idle.
- **Spend memory on bigger micro-batches.** With memory left over, a larger
  `max_tokens_per_gpu` gives fewer, larger matrix multiplies and fewer launches,
  which usually helps more than anything else in an overhead-limited step.
- **Scale GPUs only after one GPU is busy.** If one GPU's trace shows a lot of
  idle time, adding data-parallel GPUs adds cost without much speed. Measure the
  step on 1, 2 and more GPUs and fit step time ≈ a + b / GPUs. The part `a`
  doesn't shrink with more GPUs: it is per-step overhead, such as preparing the
  batch or time outside the trainer. Reduce `a` before you add GPUs, and pick
  the GPU count by cost per epoch, not by step time alone.

To remove recompute options a bundled config sets, replace its `cli_options`
with a copy that leaves them out:

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

Two worked examples, both found this way:

- **A 4B LoRA classifier** with several candidate labels per position spent
  most of `forward_backward` in many small micro-batches. Raising
  `max_tokens_per_gpu` from 2,048 to 8,192 cut the step from about 35 s to 26 s.
  Later, once the trainer was no longer overhead-limited, turning recompute off
  cut `forward_backward` by another 24%.
- **A 20B MoE model** trained with tensor and expert parallelism across 8 GPUs
  spent 35% of `forward_backward` in communication. The model fit on one GPU,
  so moving to plain data parallelism on the same 8 GPUs nearly halved
  `forward_backward` (89 s to 47 s). Setting `balance_data` then removed a
  7% straggler cost.

Retest a change you dropped earlier after any large win. A change that did
nothing while the step was limited by overhead can help once it is limited by
compute.

### Time outside the trainer

If your client's training time is much larger than `timing/trainer_step_s`:

- **Queueing behind other clients.** On a shared trainer, `GET /api/v1/spindle/queues`
  shows the backlog and recent waits. For RL, where clients sample at different
  times, set `trainer_max_clients_per_batch = 1` so each client gets its update
  back as soon as its own step finishes.
- **Many small calls.** Send one `forward_backward` with the whole batch instead
  of many small ones, and submit `optim_step` without waiting in between.
- **Distance.** Run your training script in the same region as the deployment.
- **Request timing.** Deploy with `SPINDLE_REQUEST_TIMING=1` to print
  `spindle_request_mark` lines with timestamps for upload, decode and result
  delivery in the frontend and trainer logs.

## 4. Tune the sampler

The sampler (SGLang) has no trace capture through Spindle. Use these signals
instead:

- **Sampler logs.** `modal app logs <frontend>-lora-<config>` shows SGLang's
  decode lines with `#running-req`, `#queue-req`, `token usage` (the share of
  KV cache in use) and `gen throughput`.
- **Startup marks.** `sglang_healthy` and `first_adapter_loaded` lines show
  when a replica started. Replicas starting in the middle of a sampling round
  mean you waited on cold starts.
- **Usage.** `spindle usage` reports how much of your sampler cost was
  `cold_start` and `idle` time. See [cost-attribution.md](cost-attribution.md).
- **Tracing.** With OpenTelemetry export on, `spindle.sample` spans carry
  `sglang.queue_s`, `sglang.prefill_s`, `cached_tokens` and `prompt_tokens`.
  See [observability.md](observability.md).

To tell what limits decoding, compare `gen throughput` with `#running-req` across
a round. If throughput rises with the number of running requests, each replica
needs more requests at once: send more, or raise `max_running_requests`. If it
levels off while requests keep arriving, the replica is full and you need more
replicas. If `token usage` is near 1, the KV cache is the limit: raise
`sglang_cfg.mem_fraction_static`, or give each replica more GPUs by setting
`inference_gpus_per_node` and `sglang_cfg.tp_size` to the same value.

| What limits sampling | Signs | Config to try |
| --- | --- | --- |
| **Cold starts** | Mean time per turn far above the median. The first turns of each round are slow, and replicas start mid-round. | Set `inference_min_replicas` to the replicas one round needs, or set `inference_scaledown_window_s` longer than your training step so replicas survive it. |
| **Not enough capacity** | `#queue-req` stays above 0, `token usage` is above 0.9, or requests fail with 503. | Raise `inference_max_replicas`. Lower `inference_target_concurrency` so replicas are added earlier. Raise `sglang_cfg.max_queued_requests` if requests are rejected. |
| **Too little batching** | `#running-req` stays well below `sglang_cfg.max_running_requests` while your loop waits on samples. | Send all of a round's requests at once rather than a few at a time. Raise `max_running_requests` if memory allows. |
| **Decode overhead** | `gen throughput` per request drops sharply once more requests run than the CUDA graphs cover. Decode steps above that batch size run without CUDA graphs, one kernel launch at a time. | Keep `sglang_cfg.cuda_graph_max_bs_decode` at least as large as `max_running_requests`. For small models at small batch sizes, try `sglang_cfg.enable_torch_compile: True`; it adds start-up time to every replica. |
| **Prefill** | Long prompts with short outputs, and a low `cached_tokens` share. | Spindle already sends each multi-turn conversation to the same replica, so later turns reuse the cached prompt. Try `sglang_cfg.enable_mixed_chunk: True` so prefill doesn't pause running decodes. |
| **Many adapters** | Several LoRA clients sample at once and throughput drops as clients are added. | Make sure `sglang_cfg.max_loras_per_batch` is at least the number of adapters sampling at once. |
| **Speculative decoding** | The recipe sets `sglang_cfg.speculative_algorithm`, and many requests run at once. Its speedup shrinks at large batch sizes and high temperature. | Compare a variant whose `sglang_cfg` leaves out the `speculative_*` keys, using the same pattern as for recompute above. |
| **The slowest trajectories** | A round lasts much longer than the 90th-percentile trajectory, and the sampler is nearly idle at the end of the round. | Not a sampler setting. Start the next round's requests before the slowest ones finish, or cap turns and output length in your loop. |
| **Your tools** | Tool calls take a large share of each trajectory's time. | Not a Spindle setting. Speed up or parallelize the tools in your environment. |

A 20B MoE agent RL workload with a 6-minute training step lost its sampler
replicas between rounds, because the step outlasted the 300-second scale-down
window. Keeping a minimum number of replicas cut the mean time per turn from
31.5 s to 8.1 s for the same GPU-hours.

## 5. Compare before and after

- Change one thing per run. Override fields in a subclass of your config (see
  [Variants](deployment-configs.md#variants)) so each run is a named config.
- Skip the first few steps, the profiled step and the step after it. Compare the
  median of at least 10 steady-state steps.
- Pin the GPU type. Modal can place a plain `H100` request on a faster H200. Add
  `!`, as in `trainer_gpu = "H100!"`, to get exactly that type.
- Repeat any surprising result once. Steps can differ by 5–10% between machines.
- Check that training still matches: per-token logprobs and loss on a fixed
  batch should agree with the previous run.
- For RL with long steps, tune the sampler with a rollout-only benchmark rather
  than full training steps. Save sampler weights once, then replay the same
  prompts with fixed seeds against each variant, running as many copies at once
  as you have training runs. Compare the round's wall time and the time per
  thousand generated tokens. Confirm the winner with a few real steps.
- A variant is a new config, so a running job moves to it through a checkpoint:
  `save_state` on the old config, then create a training client on the variant
  and call `load_state`.
- Keep a change only if it saves at least a few percent. Stop when what is left
  needs code changes, or when the remaining time is in your own loop.

If your config beats a bundled recipe for a common workload, open a pull request
that updates the recipe, with before and after step times.

## Past tuning changes to learn from

Each of these pull requests changed a bundled recipe or Spindle itself after a
measurement like the ones above. Their descriptions explain the bottleneck and
give before and after step times, and their diffs show the exact fields. Read
the one closest to your problem before you change a field.

Config changes to the trainer:

| Pull request | What limited the step | Change |
| --- | --- | --- |
| [#36](https://github.com/modal-labs/spindle/pull/36) | Communication from tensor and expert parallelism | Plain data parallelism on the same GPUs for a 20B MoE model |
| [#40](https://github.com/modal-labs/spindle/pull/40) | One rank getting more tokens than the others | `balance_data` |
| [#150](https://github.com/modal-labs/spindle/pull/150), [#163](https://github.com/modal-labs/spindle/pull/163) | Tensor parallelism larger than memory needs | Less tensor parallelism, chosen per GPU type |
| [#193](https://github.com/modal-labs/spindle/pull/193) | One sequence per micro-batch | Several sequences per micro-batch |
| [#203](https://github.com/modal-labs/spindle/pull/203), [#247](https://github.com/modal-labs/spindle/pull/247) | Out of memory on packed batches | A token budget per micro-batch that fits |
| [#246](https://github.com/modal-labs/spindle/pull/246) | Expert-parallel communication on a model that fits | All experts on every GPU |
| [#327](https://github.com/modal-labs/spindle/pull/327), [#337](https://github.com/modal-labs/spindle/pull/337), [#338](https://github.com/modal-labs/spindle/pull/338) | Full fine-tuning layout | Data parallelism and larger packed micro-batches on B200 and B300 |

Config changes to the sampler:

| Pull request | What limited sampling | Change |
| --- | --- | --- |
| [#37](https://github.com/modal-labs/spindle/pull/37) | Replicas scaling to zero during each training step | Keep the pool warm between steps |
| [#45](https://github.com/modal-labs/spindle/pull/45), [#243](https://github.com/modal-labs/spindle/pull/243), [#236](https://github.com/modal-labs/spindle/pull/236) | Requests rejected or queued at low concurrency | Higher running and queued request limits |
| [#49](https://github.com/modal-labs/spindle/pull/49) | Prefill pausing decodes | `enable_mixed_chunk` |
| [#237](https://github.com/modal-labs/spindle/pull/237) | KV cache too small | More memory for the KV cache |
| [#241](https://github.com/modal-labs/spindle/pull/241) | Replicas added for short bursts | Add a replica only when every replica is full |
| [#254](https://github.com/modal-labs/spindle/pull/254) | Decode speed on Hopper GPUs | B200 samplers with a Hopper fallback |
| [#165](https://github.com/modal-labs/spindle/pull/165), [#323](https://github.com/modal-labs/spindle/pull/323) | Long decodes | Speculative decoding with the model's own draft layers |

Changes inside Spindle, as examples of what an issue and pull request for a
non-config fix look like:

| Pull request | What limited the step | Change |
| --- | --- | --- |
| [#152](https://github.com/modal-labs/spindle/pull/152), [#153](https://github.com/modal-labs/spindle/pull/153), [#42](https://github.com/modal-labs/spindle/pull/42) | Time outside the trainer: distance and JSON payloads | One frontend region and a binary transport |
| [#158](https://github.com/modal-labs/spindle/pull/158) | Host time converting rows between processes | Arrays instead of Python lists |
| [#157](https://github.com/modal-labs/spindle/pull/157), [#317](https://github.com/modal-labs/spindle/pull/317), [#322](https://github.com/modal-labs/spindle/pull/322) | Memory and time in the vocabulary softmax | One softmax and one logits gradient per micro-batch |
| [#333](https://github.com/modal-labs/spindle/pull/333), [#334](https://github.com/modal-labs/spindle/pull/334) | Kernels recompiling for every new batch length | Compile once and reuse |
| [#336](https://github.com/modal-labs/spindle/pull/336) | Data-parallel GPUs waiting on the busiest one, and full-size padding micro-batches | Split rows by length before packing, and pad with short micro-batches |
| [#124](https://github.com/modal-labs/spindle/pull/124) | Several LoRA clients waiting on each other | Serve clients one at a time, short operations first |
| [#46](https://github.com/modal-labs/spindle/pull/46) | Prefix cache misses on multi-turn conversations | Send each conversation to one replica |

## When the fix is not a config change

Sometimes the trace points at Spindle itself or at one of its backends: a
Python loop in the request path, a slow kernel, or a missing feature in Miles or
SGLang. Keep tuning with config in the meantime, and:

1. **Open an issue** on [modal-labs/spindle](https://github.com/modal-labs/spindle/issues)
   with your config, the step-time table, and the trace or the part of it that
   shows the problem.
2. **Say where the fix belongs.** Problems in training execution usually belong
   in Miles, and problems in sampling in SGLang. If the same problem exists
   upstream, say so and link the upstream issue. Spindle keeps its Miles and
   SGLang forks close to upstream, so fixes that upstream would accept are the
   easiest to take.
3. **Optionally, contribute the change.** You can test a patched backend without
   changing Spindle: build your own `modal.Image` and set `trainer_image` or
   `inference_image` to its `module:attribute` path (see [Images](deployment-configs.md#images)).
   Then open a pull request with before and after step times. The Spindle
   maintainers review it.

## Where a cold start spends its time

A trace explains steady-state speed. A cold start is different: the first
`forward_backward` on a fresh deployment waits for a GPU to be scheduled, the
container to boot, the model to load, and kernels to compile. Spindle times each
of these phases on every container, with no configuration needed, and prints one
JSON line per phase boundary to the container's log:

```json
{"event": "spindle_startup_mark", "component": "trainer-backend", "mark": "trainer_actors_initialized", "ts": 1792012345.1, "since_start_s": 212.4, "phase_s": 148.9}
```

- `since_start_s` is the time since that container's process started.
- `phase_s` is the time since the previous mark in the same component, so it is
  the cost of the phase that ends at this mark.

Trainer marks, in order (`component` in parentheses):

| Mark | Phase that ends here |
| --- | --- |
| `trainer_function_entered` (trainer) | Container boot and Modal runtime start. |
| `assets_reloaded` / `ray_cluster_joined` (trainer) | Volume reload, or Ray cluster formation for multi-node trainers. |
| `backend_spawned`, `engine_record_starting` (trainer) | Backend process launched. `engine_record_starting` carries `queue_s`, the wait between the control plane requesting the trainer and the container's process starting. |
| `backend_imports_done` (trainer-backend) | Importing Torch, Megatron and Miles. |
| `miles_args_parsed`, `ray_started`, `ray_workers_launched` (trainer-backend) | Ray startup and per-GPU worker actors. |
| `actor_model_loaded` (trainer-rank0) | Rank 0 built the Megatron model and loaded the base weights from the assets volume. |
| `trainer_actors_initialized`, `miles_backend_ready` (trainer-backend) | All ranks finished loading. |
| `backend_ready`, `engine_serving` (trainer) | The engine answered its health check and is reachable; `since_spawn_s` is the total from request to ready. |
| `first_slot_loaded`, `first_forward_backward_done`, `first_optim_step_done`, `first_sampler_export_done` (trainer-backend) | The first client's work, including kernel compilation. `duration_s` is the call itself. |

Sampler replica marks: `server_enter`, `sglang_healthy` (base model loaded and
CUDA graphs captured), `sidecar_healthy`, then `first_adapter_loaded`
(`sampler-sidecar`) when the first LoRA adapter is applied. The one-time pool
deployment prints `pool_deployed` (`sampler-pool`), and the model asset
download prints `assets_ready` (`model-assets`).

The control plane also keeps `spawned_at`, `started_at`, `ready_at` and the
phase breakdown on each trainer instance record, next to its state.

To list a deployment's marks as a table:

```bash
uv run modal app logs spindle-trainer-<config> \
  | grep -o '{"event": "spindle_startup_mark".*}' \
  | jq -r '[.component, .mark, .since_start_s, .phase_s] | @tsv'
```
