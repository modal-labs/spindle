# Profiling a step

Miles LoRA trainer only; off by default. For what to measure first, how to classify a bottleneck (overhead, compute, memory, communication, slow rank), sampler signals, and which config fields to change, follow [docs/profiling.md](https://github.com/modal-labs/spindle/blob/main/docs/profiling.md). Prefer config changes; a backend change goes to a GitHub issue first.

## Capture

- `SPINDLE_TORCH_PROFILE_STEP=2 uv run spindle deploy <config> --refresh-trainer <config name>`: traces the third training step (steps count `optim_step` calls from 0; use 1 or 2 so compile work is out). A running trainer keeps its old env until it idles out, so profile a variant with its own name to be sure. Redeploy without the variable (and `--refresh-trainer`) to disable.
- `SPINDLE_TORCH_PROFILE_RANKS`: default `0` (rank 0); e.g. `0,4` or `all`. ~250 MB per trace.
- `SPINDLE_TORCH_PROFILE_DIR`: default `/checkpoints/torch-profile/<config name>`.

Captures one whole training step per trainer process on the GPU ranks: all its `forward_backward` calls, `optim_step` and LoRA export (first `forward_backward` of the step until its sampler weights are published or the next step starts; on a shared trainer, the first client to reach it). Rank phases: `spindle/forward_backward`, `spindle/optim_step`, `spindle/forward_only`, `spindle/export_slot_peft`. Sampler-weight save and publish are not traced; read `timing/save_sampler_weights_s` and `timing/publish_weights_s`. `rank<N>.key_averages.txt` is a no-UI top-ops table. Writing the trace delays the next update; exclude both steps from timings.

Always-on timing without a trace: `optim_step` result metrics `timing/*_s` and `timing/*_calls`, and `spindle_step_timing` / `spindle_memory` lines in the trainer log.

## Download and view

```bash
uv run modal volume get <platform.storage.checkpoints, default spindle-checkpoints> /torch-profile/<config name>/rank0.trace.json.gz .
```

Open at https://ui.perfetto.dev (loads `.json.gz` directly). Search `spindle/` to jump between phases. Look for idle GPU streams inside `spindle/forward_backward` (CPU/comms stalls), long `ncclKernel_*`, and time outside labelled phases.

## DuckDB queries

The `.json.gz` is one JSON object; raise the object-size cap before unnesting:

```sql
CREATE VIEW ev AS
SELECT unnest(traceEvents, recursive := false) AS e
FROM read_json('rank0.trace.json.gz', format='unstructured',
               maximum_object_size := 536870912);
```

Duration of each `spindle/*` phase:

```sql
SELECT e->>'name' AS phase, sum((e->>'dur')::DOUBLE) / 1e6 AS dur_s
FROM ev
WHERE e->>'cat' = 'user_annotation' AND e->>'name' LIKE 'spindle/%'
GROUP BY 1 ORDER BY 2 DESC;
```

Top GPU kernels inside one phase's window (`kernel`/`gpu_memcpy`/`gpu_memset` are the GPU-stream categories; `gpu_user_annotation` marks phases on GPU rows):

```sql
WITH w AS (
  SELECT (e->>'ts')::DOUBLE ts, (e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE ts_end
  FROM ev
  WHERE e->>'name' = 'spindle/forward_backward' AND e->>'cat' = 'user_annotation'
  LIMIT 1
)
SELECT e->>'name' AS kernel, sum((e->>'dur')::DOUBLE) / 1e3 AS ms
FROM ev, w
WHERE e->>'cat' = 'kernel'
  AND (e->>'ts')::DOUBLE >= w.ts AND (e->>'ts')::DOUBLE < w.ts_end
GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
```

GPU busy fraction of the phase window (sums overlapping streams, so >1 means parallel streams, not >100% of one GPU):

```sql
WITH w AS (
  SELECT (e->>'ts')::DOUBLE ts, (e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE ts_end
  FROM ev
  WHERE e->>'name' = 'spindle/forward_backward' AND e->>'cat' = 'user_annotation'
  LIMIT 1
), k AS (
  SELECT greatest((e->>'ts')::DOUBLE, w.ts) s,
         least((e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE, w.ts_end) f,
         w.ts_end - w.ts AS win
  FROM ev, w
  WHERE e->>'cat' IN ('kernel', 'gpu_memcpy', 'gpu_memset')
    AND (e->>'ts')::DOUBLE < w.ts_end
    AND (e->>'ts')::DOUBLE + (e->>'dur')::DOUBLE > w.ts
)
SELECT coalesce(sum(greatest(f - s, 0)) / max(win), 0) AS busy_frac FROM k;
```

NCCL share of kernel time:

```sql
SELECT sum((e->>'dur')::DOUBLE) FILTER (e->>'name' LIKE 'nccl%')
     / nullif(sum((e->>'dur')::DOUBLE), 0) AS nccl_frac
FROM ev WHERE e->>'cat' = 'kernel';
```
