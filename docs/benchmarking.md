# Benchmarks

Spindle includes inference, eval, trainer, weight-sync, async-RL, and control-plane benchmark suites.
Results go to a local SQLite store and a portable JSON bundle. The results browser
discovers every recorded numeric metric, including new backend metrics, without
requiring a chart definition. It also shows labels, raw records, run configuration,
collection errors, and OTLP trace documents.

For Frontier-style training curves and sweep planning, upload the same version 1
bundles into a Frontier instance with training ingestion enabled. The controller
can publish directly to a compatible collector using `--collector-app`.

## Start the results browser

Install the repository's locked dependencies, then run:

```bash
uv sync --locked --python 3.12
uv run python -m spindle.benchmarks --directory ./benchmark-results serve
```

Open `http://127.0.0.1:8765`. Select runs, search the metric list, and choose elapsed
time, step, context length, or client count for the X axis. Raw histogram buckets,
sums, counts, and bounds are separate metrics; the original histogram is also
available in raw records. Sample means are not histogram latency percentiles.

Use **Upload results** to import a result bundle. For a benchmark produced elsewhere:

```bash
uv run python -m spindle.benchmarks --directory ./benchmark-results ingest results.json
uv run python -m spindle.benchmarks --directory ./benchmark-results import-json old-results.json
```

Legacy JSON imports expose all nested numeric fields but are marked as uncontrolled
imports. Download a run's complete bundle from its details panel, or use `export`.
Repeated ingestion of the same records does not duplicate samples.

The server binds to loopback by default. Set `SPINDLE_BENCHMARK_TOKEN` before using
`serve --host 0.0.0.0`. The UI's **Access token** button supplies the token for API
calls; it stays in session storage. External clients send `Authorization: Bearer …`.

## Collect remote telemetry

The browser's server also receives OTLP HTTP/protobuf at `/v1/metrics` and
`/v1/traces`. Modal containers need a collector URL they can reach; a laptop's
loopback URL does not work from a remote container.

For a collector in your own Modal environment, deploy:

```bash
uv run modal deploy --strategy recreate -e YOUR_ENV -m spindle.benchmarks.modal_app
```

This app uses its own `spindle-benchmark-results` volume and requires Modal proxy
authentication. Its single container owns the SQLite writer. It does not reuse
a training API key for dashboard authentication. To use it for uploads, supply
your Modal proxy credentials through the environment and opt in explicitly:

```bash
uv run python -m spindle.benchmarks upload results.json \
  --url https://YOUR_COLLECTOR.modal.run --modal-proxy-auth
```

Use `recreate` for updates so old and new versions cannot write the SQLite store
concurrently. Requests during the short restart may retry.
The collector batches volume snapshots every five seconds and flushes on graceful
shutdown. An abrupt container loss can lose records since the last snapshot;
client bundles remain independently downloadable.

Set the runner's OTLP authentication through environment variables:

```bash
export OTEL_EXPORTER_OTLP_HEADERS='Modal-Key=YOUR_PROXY_ID,Modal-Secret=YOUR_PROXY_SECRET'
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

Use OTLP percent-encoding for header values. An additional benchmark-only bearer
token is optional: set `SPINDLE_BENCHMARK_AUTH_SECRET` to the name of a secret
containing `SPINDLE_BENCHMARK_TOKEN` when deploying the collector. The proxy
authentication requirement remains enabled.

To forward the collector's saved telemetry to another OTLP destination, set
`SPINDLE_BENCHMARK_FORWARD_OTLP_ENDPOINT` and
`SPINDLE_BENCHMARK_FORWARD_HEADERS` (a JSON object) on the collector. For example,
the latter can contain a `dd-api-key` header. Forward failures are retained as
collection errors and return a retryable response. Keep credentials in environment
variables or secrets, not benchmark config files.
For the Modal collector, put these variables in a secret and set
`SPINDLE_BENCHMARK_FORWARD_SECRET` to its name when deploying. Keep run attribution
under `service.name=spindle-benchmarking` and `spindle.run_id`; the ingestion key
does not select the service name.

## Run a revision

If deployment credentials already live in Modal secrets, run the CPU controller
in the same environment. It uses the existing proxy-auth helper for RL scripts,
uploads results to the collector, and downloads the combined bundle locally:

```bash
uv run modal run -e YOUR_ENV -m spindle.benchmarks.controller \
  --config benchmarks/configs/trainer.json --revision HEAD \
  --environment YOUR_ENV --directory ./benchmark-results
```

The controller uses `spindle-api`, `spindle-proxy`, and `huggingface-secret` by
default. Override the API and proxy secret names with its command options. An
explicit deployment credential secret can be selected with `--modal-secret`.
Deployment credentials stay in Modal containers.
The archive's recorded Git commit must match the requested revision.

The **GPU benchmarks** manual GitHub Actions workflow accepts a recipe, revision,
and Modal environment. It uses the repository's `gpu-ci` environment credentials,
publishes to the collector, saves a downloadable artifact, and cleans up apps
owned by that workflow attempt. Deploy the collector before dispatching it. The
workflow is opt-in; it does not add GPU jobs to every pull request.

`weight-sync-native-smoke.json` is a small full-parameter Megatron smoke workload:
one H200 trainer, one H100 sampler, pinned Qwen3-4B weights, one warm-up update,
and two measured updates. The other recipes exercise larger LoRA and RL workloads.
`timeout_s` covers deployment and workload execution; cleanup has a separate
five-minute budget.

List central runs or sync one into a local browser store with the same controller:

```bash
uv run modal run -e YOUR_ENV -m spindle.benchmarks.controller \
  --environment YOUR_ENV --list-runs
uv run modal run -e YOUR_ENV -m spindle.benchmarks.controller \
  --environment YOUR_ENV --run-id RUN_ID --directory ./benchmark-results
```

Copy a config from `benchmarks/configs/`, choose a Python deployment recipe, and
adjust its sweep and deployment overrides. Validate the workload:

```bash
uv run python -m spindle.benchmarks validate benchmarks/configs/trainer.json
```

Commit the Spindle change you want to measure. Start an isolated deployment:

```bash
export TINKER_API_KEY='the-key-in-your-deployment-api-secret'
export MODAL_PROXY_TOKEN_ID='your-proxy-token-id'
export MODAL_PROXY_TOKEN_SECRET='your-proxy-token-secret'

uv run python -m spindle.benchmarks --directory ./benchmark-results run \
  benchmarks/configs/trainer.json --revision HEAD --env YOUR_ENV \
  --otlp-endpoint https://YOUR_COLLECTOR
```

The runner archives the exact requested Git commit, deploys it behind a unique
frontend, runs the workload, and stops its owned deployment in cleanup. It uses
your selected deployment recipe's secret names in the explicitly selected Modal
environment. Failed and interrupted runs retain measurements and failure records.
Cleanup failures are visible in the bundle and `cleanup.log`; use `spindle stop
--app FRONTEND --env YOUR_ENV` if cleanup could not complete.

An existing server can be used with `--base-url URL`. Its revision is recorded as
an assertion, not verified. The runner does not redeploy, instrument, or stop that
server. Deploy it with benchmark telemetry enabled before testing it. Requests
use the stock Tinker SDK; proxy headers come from the environment variables above.

The benchmark harness and RL recipe code come from the checkout running this
command. The deployed Spindle comes from `--revision`. This lets the same harness
drive both revisions. Manifests retain the workload fingerprint, source lock hash,
resolved deployment, local package versions, and recipe file hashes. Per-client
prompt hashes identify the fixed-data workload. Adapter initialization is not
seeded because the Miles LoRA backend does not support per-adapter seeds.

For a controlled comparison, pin model weights in `deployment_overrides` using
`weights_revision` and use fixed hardware rather than a list of fallback GPUs.
Without these pins, a matching requested config does not guarantee the same
downloaded weights or GPU allocation. Worker telemetry reports actual GPU names,
memory, and available runtime package versions and source commits.

## Suites

| Suite | Workload and measurements |
| --- | --- |
| `inference` | Distinct LoRA clients publish adapters and sample concurrently at each context/client count. Records per-client latency, actual input/output token counts, and observed output rate. |
| `eval` | Fixed-token forward-only cross-entropy requests, with latency, tokens per second, and all backend result metrics. No backward pass or optimizer update. |
| `control-plane` | Concurrent authenticated capabilities requests, retaining individual latencies, error counts, mean/p95 latency, and request rate. Includes transport time. |
| `trainer` | Repeated fixed-token cross-entropy forward/backward and optimizer updates. Records all returned result metrics, forward/backward and optimizer latency, step time, and processed input tokens. |
| `weight-sync` | An optimizer update, weight publication, then a one-token request through the returned version-specific sampling client. Records publication receipt, probe time, and update-to-first-response latency. |
| `async-rl` | Existing DAPO or SWE-Gym recipes, or an argv-based external recipe. Streams all numeric event/metric fields into the result store, including nested rollout, update, queue, reward, and policy-lag values when the recipe emits them. |

Warm-up observations are retained with a separate `phase` label. Trainer timings
include host transport and synchronization; they are not GPU kernel durations.
Weight-sync's first-response timer is an upper bound on readiness and includes
probe execution. The existing capture/publication spans provide additional phase
boundaries. Buffered SDK responses do not establish streaming TTFT or ITL.

DAPO uses the checked-in dataset and recipe from
`examples/dapo-27b-4h200-12b200`. Install its tokenizer dependencies before use.
SWE-Gym needs a compatible pinned worker checkout and its sandbox dependencies:

```bash
export SWE_GYM_WORKER=/path/to/modal-examples/06_gpu_and_ml/swe_gym/swe_gym.py
```

The adapter copies the worker and helpers into the run artifacts, applies `MODEL`
and `TRAINING` config values, and records file hashes. It invokes the local worker
entrypoint, bypassing that worker's example deployment and secret selection. The
supported worker layout has top-level `MODEL` and literal `TRAINING` assignments.
Each RL config selects one client count and one context length. External command
recipes accept `{output}`, `{base_url}`, and `{model}` substitutions and must write
`events.jsonl` or `metrics.jsonl` under `{output}`, with `update_done` or `step_done`
events for completed updates. Process groups are stopped on failure or timeout.

## Metric coverage

With `SPINDLE_BENCHMARK_METRICS=1` and an OTLP endpoint in the trainer and inference
environments, the collectors export:

- Every numeric sample exposed by the SGLang Prometheus endpoint, including
  cache/pool occupancy, running/queued requests, counters, and histogram buckets
  provided by that SGLang build. Original names, labels, and Prometheus types are
  retained. Native Prometheus values are exported as raw snapshot gauges; counters
  and histogram buckets retain their cumulative meaning.
- New benchmark deployments enable all-scheduler metrics, forward-pass metrics,
  native MFU metrics, cache reports, and request timing reports on the pinned
  SGLang build. Explicit `sglang_cfg` settings take precedence. Native MFU uses
  that engine's estimates; it is not computed from device utilization.
- All numeric trainer `metrics` response fields, backend phase measurements, and
  Miles phase timers. Existing Spindle OTLP traces retain command and batch links.
- Per-rank Torch CUDA allocator statistics and free/total device memory in Miles
  actors and native Megatron workers, without adding CUDA synchronization.
- Device utilization, memory, power, temperature, and clocks through `nvidia-smi`.
  These are periodic device observations, not modeled MFU or per-kernel profiles.

The scrape interval is two seconds. Additional native Miles/Megatron Prometheus
endpoints can be selected with `SPINDLE_BENCHMARK_PROMETHEUS_URLS`, a JSON URL list.
Scrape failures and unavailable device queries have explicit health metrics.
Nonfinite samples are preserved in raw records and excluded from numerical plots.
Internals that upstream does not expose are not fabricated. Profiling remains
available separately through [the profiling guide](profiling.md).

Spindle revisions that predate these collectors still run, with an explicit
coverage warning. Only their pre-existing OTLP signals can be received. Without
an OTLP destination, the run retains client measurements and reports the missing
server telemetry. A run ID on the OTLP resource is required for correlation.

## Compare runs

Run the same config on the base and candidate commits, then upload both bundles
to the collector or import both into one local store. Native telemetry received
by a remote collector is already in that collector's store; a local runner's
bundle contains the records available in its local store. Download from the
collector to obtain the combined bundle.

```bash
uv run python -m spindle.benchmarks --directory ./benchmark-results compare BASE_ID CANDIDATE_ID
```

The comparison reports median measurement values and changes for matching label
groups, excluding warm-up. It refuses controlled comparisons when workload or
configuration fingerprints differ, recipe sources differ, either run failed, or
the server revision was not verified. `--exploratory` shows values and limitations
without percentage claims. Inspect resolved config and actual hardware in the
dashboard before interpreting any result. Sample counts and individual values
remain visible; small or noisy runs should not become merge gates.


## CI workloads

Select `training-ci` in the GPU benchmark workflow to run small dense
(Qwen3.5-9B), small MoE (Qwen3.6-35B-A3B), and large MoE (GLM-5.3), with
eval and SFT workloads at 1,024 and 8,192 tokens and 1, 2, and 4 clients. These
recipes pin model snapshots and trainer hardware. Context and client counts can
be changed in the JSON recipes. The matrix also includes weight sync and a
control-plane load test at 1, 8, 32, and 128 concurrent clients.

The nightly schedule runs DAPO on the same pinned MoE checkpoint with replay off,
router replay, score centering, and both enabled, followed by SWE-Gym. All four
DAPO variants use the same sampling parameters and input dataset. They retain
rollout wait, generation, training, publication, policy-lag, reward, token counts,
and every numeric recipe/backend field. Score centering uses its own loss;
reward and sequence lengths must be checked before attributing timing changes
only to implementation overhead. These are performance observations, not automatic
merge gates.

The workflow publishes into a separately deployed collector named
`frontier-training-dev`. Deploy that collector before enabling the schedule.
Use `--collector-app NAME` on the controller to select another compatible
collector; use the same value for result downloads and owned-app cleanup.
The schedule is defined in source and does not run until this workflow reaches
the repository's default branch.

Opt-in telemetry emits capture-start and sampler-ready boundaries for each
publication. The results store joins these by model and version and reports
`weight_sync.capture_to_ready_s` separately for each sampler. It excludes probe
generation and requires synchronized clocks across hosts. Missing or negative
intervals are omitted; their raw boundary records remain available. The broader
`weight_sync.update_to_first_response_s` client observation is retained.
