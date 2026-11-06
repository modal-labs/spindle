# Trainer runtime memory snapshots

**Result:** a separate four-H100 container restored the warmed 9B Miles trainer
with **GPU P2P/CUMEM NCCL**, matched the saved model/gradient/optimizer/RNG state,
passed all process-group checks and completed another backward/Adam update.
This uses explicit per-actor NCCL suspend/resume, `NCCL_CUMEM_ENABLE=1`, NVLS
disabled, and stable loopback addresses for Ray and communication bootstrap.
The first forward matches exactly; numerical training-continuation parity
remains unproven. Earlier socket-NCCL results are retained below.

This experiment asks whether a new container can resume an initialized Spindle
trainer from a Modal CPU/GPU memory snapshot, without loading model or optimizer
checkpoints. The first target is `Qwen/Qwen3.5-9B-Base`, Miles LoRA, TP=4 on
4 H100s in one container, using the existing 16K recipe and a 128-token probe.
It is an investigation, not a production trainer configuration.

## Exactly what was saved and restored

The experiment did **not** export a model-plus-runtime file to a `modal.Volume`
and then load that file. It used Modal's deployed Function/Cls memory-snapshot
lifecycle. Modal persisted the snapshot in platform-managed storage and selected
it when starting a compatible new container. The harness did not obtain a
portable snapshot file, snapshot ID, or user-controlled save/load path.

There were two separate kinds of persisted data:

| Storage | Contents | How the experiment used it |
| --- | --- | --- |
| Dedicated Modal Volume, mounted at `/assets` | Original Hugging Face `Qwen/Qwen3.5-9B-Base` files | Downloaded once by `prepare()` and read during the initial trainer construction |
| Modal-managed CPU/GPU memory snapshot | Warmed container process tree and CUDA state, including the already-loaded base-model shards, trained LoRA adapter and Adam state | Captured after `@modal.enter(snap=True)` and restored automatically for a later container |
| Local JSON evidence files | Hashes, group inventory, IDs, timings and metrics | Used to verify the experiment; these files cannot restore a trainer |

The capture and restore sequence was:

1. **Build the initial runtime.** In `@modal.enter(snap=True)`, construct a real
   `MilesCommandBackend`. This starts its background event-loop thread, local Ray
   cluster and four trainer actors, creates process groups, reads the base weights
   from `/assets`, and places the TP=4 shards on the GPUs.
2. **Materialize training state.** Create one rank-32 LoRA job, perform a synthetic
   128-token backward/Adam update, run a reference forward, and record per-rank
   state hashes and collective results. The adapter is already trained for one
   step at this boundary; the saved state is more than frozen base weights.
   In suspend mode, verify a communicator suspend/resume round trip, then suspend
   every registered NCCL group again before returning from the snapshot hook.
3. **Let Modal capture the container.** The class enables
   `enable_memory_snapshot=True` and
   `experimental_options={"enable_gpu_snapshot": True}`. After the snapshot hook
   returns, Modal handles CPU/process and CUDA checkpointing. Its documented GPU
   mechanism drains outstanding GPU work and moves GPU allocations/state through
   host memory for the container snapshot. The original container then resumes;
   passing checks in that original container alone does not establish cloning.
4. **Start a different container from that snapshot.** Invoke the same deployed
   class again after publication. On a compatible worker, Modal restores the
   process tree and CUDA state before `@modal.enter(snap=False)` runs. The
   `MilesCommandBackend` constructor, `ray.init()`, actor/model initialization,
   LoRA slot loading and warmup hook are skipped. Ray is retained, not restarted.
   In suspend mode, the post-restore hook resumes the saved NCCL backends on
   every actor before submitting any training work.
   The harness makes no application-level model or optimizer checkpoint-load call
   on this path.
5. **Verify the restored runtime.** Generate a fresh boot ID after restore, run
   the saved actors' collectives and hash comparisons, reproduce reference log
   probabilities, and execute the next backward/Adam update. The saved
   initialization ID and actor PIDs match, while the Modal task ID changes.

This is a reusable initialization snapshot associated with a deployed function,
its configuration and compatible worker hardware. It is not a demonstration of
periodically saving an arbitrary live training session to a volume and explicitly
restoring it by checkpoint path. The image and referenced asset volume remain
dependencies; the experiment did not test restoration after deleting them.

Only the trainer backend and its dependencies were instantiated. The Spindle
Engine HTTP frontend, tunnels, control-plane registration and sampler were not
started or restored. A production integration would need to initialize those
per-container services and identities after restoring the backend.

The quoted timings describe different intervals in the successful experiment:

- **333.51 s:** the original initialization hook, starting after Python imports
  and ending after model loading, first training step, reference forward and
  initial fingerprints. This is not just checkpoint-file I/O.
- **About 41 s:** the new container's `Restoring Function from memory snapshot.`
  log to its post-restore hook, using second-resolution logs. This is a combined
  platform resume interval, not a separately measured CPU, GPU or storage stage,
  and excludes earlier scheduling/queue time.
- **18.06 s:** the `verify()` method after restore, including full tensor hashes,
  collectives, forward comparisons and another training step. This validation
  overhead is not a required 18-second production startup phase.

## What must be captured

| State | Expected mechanism | Open compatibility question |
| --- | --- | --- |
| Sharded weights, buffers, gradients, Adam state | GPU allocations plus CPU process memory | Allocator and driver allocation types must support checkpoint/restore |
| CUDA contexts, streams, events, loaded kernels, captured graphs | CUDA checkpoint/restore | Driver, GPU topology, IPC and shared GPU mappings |
| PyTorch/Megatron process groups | CPU state plus all underlying NCCL/Gloo resources | Communicator objects alone do not restore IPC handles, NVLS multicast or RDMA resources |
| Ray driver, runtime loop, actors, GCS, raylet, object store | Entire container process tree, threads, shared memory and local connections | Whether all local connections, identities and timeouts survive cloning |
| Python/NumPy/PyTorch RNG and training counters | CPU/GPU memory | Clones intentionally inherit RNG state; new instance identities must be generated after restore |
| Engine HTTP server, Modal tunnels, control-plane registrations | Recreate after restore | External peers, URLs, tokens and leases belong to the new instance |
| Files and mounted volumes referenced by the runtime | Existing image/volume files | Memory snapshots do not version mutable external volumes |

The current trainer builder in `src/spindle/providers/modal/deployment_apps.py`
calls `run_trainer()` inside the function body. That starts the backend HTTP
subprocess and initializes Miles/Ray after Modal's initialization snapshot point.
Turning on memory snapshots there would not capture the initialized trainer.
The expensive trainer initialization must move into `@modal.enter(snap=True)`.

`MilesRuntime` owns a background asyncio thread, a local Ray cluster (for
single-node deployments), worker-manager and trainer handles, and an object store.
Retaining only the Python driver object cannot retain its dependencies. Restarting
Ray after restore would invalidate the existing actors and their resident state.

## Reproduction

The manual harness is `tests/manual/benchmark_qwen9b_memory_snapshot.py`; its
test-only actor subclass is `tests/manual/qwen9b_snapshot_probe.py`.
The app and asset volume use dedicated names and do not register a production
engine, start a sampler, or access production checkpoints.

For the successful configuration, from the repository root using the intended
Modal profile/environment:

```bash
export SPINDLE_SNAPSHOT_APP=spindle-qwen9b-memory-snapshot-loopback-1105
export SPINDLE_SNAPSHOT_TRANSPORT=socket
export SPINDLE_SNAPSHOT_RAY_IP=127.0.0.1
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py --mode prepare
uv run modal deploy tests/manual/benchmark_qwen9b_memory_snapshot.py
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode baseline --repeats 1 --output /tmp/qwen9b-baseline.json
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode snapshot --repeats 3 --require-restore --output /tmp/qwen9b-snapshot.json
```

Use a distinct `SPINDLE_SNAPSHOT_APP` for independent experiments. Omitting the
transport/Ray overrides exercises the default configuration, which stalled
capture in the recorded runs. The script reports the snapshot creator task ID
and whether a result came from a different container; `--require-restore` fails
if all successful calls only initialized or resumed their original containers.
The historical JSON records predate these explicit fields, so their proof uses
the matching initialization IDs, differing task IDs and platform restore log.

The local entrypoint calls the **deployed** class with `Cls.from_name()`. The
warning about disabled snapshots in an ephemeral `modal run` app does not apply
to that deployed class. Snapshots are not exercised by invoking an ephemeral
class directly.

Each verification runs in a single-use container. Before snapshotting, the harness:

1. Initializes the real Miles backend, Ray actors and TP=4 model.
2. Loads a rank-32 adapter and runs backward plus Adam, materializing lazy state.
3. Records reference forward log probabilities.
4. Runs an all-reduce with a known answer on every existing process group.
5. Hashes all model parameters/buffers, optimizer state, and CPU/CUDA/Python/NumPy
   RNG state in each actor.

After restore, it repeats the group collectives and hashes, checks the reference
log probabilities, then requires another successful backward/Adam update and
changed weight and optimizer hashes. No model or optimizer checkpoint is loaded
on this path. A successful health response alone is not a pass.

An `initialization_id` is generated before capture and a `boot_id` after restore.
Multiple results sharing the former but having different boot/task IDs, together
with Modal's restore log, establish reuse of an actual snapshot. Otherwise a
successful result might be a cold initialization or snapshot fallback.

The first snapshot-building call includes capture overhead. `call_to_result_s`
includes scheduling and verification; `initialization_s` is the work bypassed by
restore; `verification_s` measures the `verify()` method. These are not isolated
platform restore latency measurements. Use Modal container startup metrics to
separate queueing, image startup, capture, and restore.
The harness waits 60 seconds between repeat calls for asynchronous snapshot
publication; this wait is outside `call_to_result_s`. A worker with a different
compatibility fingerprint may still need its own initial snapshot.

For a transport diagnostic, deploy a separate app with
`SPINDLE_SNAPSHOT_APP=spindle-qwen9b-memory-snapshot-socket-1105` and
`SPINDLE_SNAPSHOT_TRANSPORT=socket`, using those variables for subsequent commands
as well. This disables NCCL peer-to-peer, shared-memory, NVLS and IB transports
and confines sockets to loopback. Success there would establish a compatibility
workaround, not support for the default high-performance communication path.

`SPINDLE_SNAPSHOT_TRANSPORT=no-nvls` disables only NVLS and retains GPU P2P/IPC.
Use a separate `SPINDLE_SNAPSHOT_APP` for each variant to preserve the evidence
and avoid invalidating snapshots from another configuration.

For explicit communicator suspension, set `SPINDLE_SNAPSHOT_COMMS=suspend`.
After warmup and the initial collective probes, the driver stops submitting
training commands. Each actor barriers its NCCL groups in consistent order,
synchronizes CUDA, then calls `group._get_backend(device).suspend()` on each
initialized group, including the default group. PyTorch 2.13 exposes these
backend methods; they call `ncclCommSuspend(comm, NCCL_SUSPEND_MEM)` and
`ncclCommResume(comm)` with NCCL 2.29.7 or newer. They are not the vLLM-specific
`suspend_device_comms()` / `resume_device_comms()` helpers.

The harness first performs a suspend/resume round trip without a snapshot and
checks all state fingerprints, group collectives and reference log probabilities.
It then suspends again for capture. The post-restore hook resumes every saved
backend before verification runs, in both the original and cloned containers.
This mode also selects loopback for NCCL/Gloo bootstrap sockets and explicitly
sets `NCCL_CUMEM_ENABLE=1`: Miles otherwise defaults it to `0`, which selects the
legacy P2P/IPC path rather than the P2P/CUMEM path used by the successful controls.
Selecting loopback does not disable GPU P2P communication.
Missing APIs and suspension failures raise errors instead of silently skipping
communicators. This mode only handles the idle, one-GPU-per-actor workload in
this harness; it does not coordinate arbitrary concurrent training submissions.

```bash
export SPINDLE_SNAPSHOT_APP=spindle-qwen9b-nccl-suspend-cumem-1106
export SPINDLE_SNAPSHOT_TRANSPORT=no-nvls
export SPINDLE_SNAPSHOT_RAY_IP=127.0.0.1
export SPINDLE_SNAPSHOT_COMMS=suspend
uv run modal deploy tests/manual/benchmark_qwen9b_memory_snapshot.py
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode snapshot --repeats 3 --require-restore \
  --output /tmp/qwen9b-nccl-suspend.json
```

`tests/manual/benchmark_ray_memory_snapshot.py` is a CPU-only control using the
same Miles image. Deploy it before running it. It snapshots a local Ray cluster
and four stateful actors, then verifies actor identities and mutations after
restore. Its `--node-ip 127.0.0.1` option isolates the effect of Ray advertising a
container IP instead of loopback.

## Observed results

The baseline passed on 2026-11-05 using 4 NVIDIA H100 80GB HBM3 GPUs, driver
610.57.04, PyTorch 2.13.0+cu130, CUDA runtime 13.0, and NCCL 2.29.7.
The live experiment used source commit
`e29c8d5` (see the recorded repository revision
when reproducing); the PR adds the harness and evidence rather than modifying
the production trainer path.
The result is retained in
[`qwen9b-baseline.json`](assets/trainer-memory-snapshots/qwen9b-baseline.json).

- Initialization, slot loading, first backward/Adam, reference forward and initial
  fingerprints: **268.62 s**. First backward alone took **152.31 s**, including
  compilation and lazy initialization.
- Verification: **14.12 s**, including full model/optimizer/gradient fingerprints.
  The warmed second backward plus optimizer took about **0.57 s**.
- Client call to completed verification: **334.71 s**. This includes imports and
  container startup outside the initialization timer.
- All four sharded model hashes differed, as expected. Combined live allocations
  reported by PyTorch were **20.61 GiB**, excluding CUDA/NCCL allocations outside
  PyTorch's allocator.
- The collective checks covered **58 distinct groups: 48 NCCL and 10 Gloo**
  (22 memberships per rank). Log probabilities matched exactly; optimizer state
  and model hashes changed after the second successful update.

This baseline used container-local compilation caches, whereas production
`run_trainer()` mounts Spindle's persistent kernel cache. It establishes
correctness of the workload and probes, not a production startup speedup.

### Ray network identity

The CPU-only control reproduced a failure independently of CUDA/NCCL. Ray 2.58.0
created a valid CPU snapshot, but the next container's actor calls failed with
`ActorUnavailableError` / `No route to host`, targeting the original container's
`172.20.x.x` actor address. The restore log explicitly said
`Restoring Function from memory snapshot.`

Passing `_node_ip_address="127.0.0.1"` alone was insufficient: Ray's
`services.resolve_ip_for_localhost()` rewrote it to the external container IP.
The test-only workaround preserves loopback during that normalization and
asserts the actual `NodeManagerAddress` is `127.0.0.1`.

With that workaround, two independent restored containers successfully accessed
the same saved actor PIDs and counter values, then incremented each counter.
The saved snapshot's counters remained unchanged for the next clone. Measured
client call times were **5.85 s** and **7.38 s**, versus approximately **20–21 s**
for snapshot-creating calls. Results are in
[`ray-forced-loopback.json`](assets/trainer-memory-snapshots/ray-forced-loopback.json).
This demonstrates local Ray process-tree restoration, not reconnection to a
remote Ray cluster.

### GPU communication capture

The default 9B configuration reached the capture boundary after a successful
backward, Adam step and all group probes, then made no progress for **10 minutes**
before the experiment was stopped. Platform logs confirmed
`memory checkpoint starting; container liveness counting stopped`, with no
corresponding resume/save-complete event during that window. This is a bounded
capture stall, not a reported CUDA error or proof of one particular driver bug.

Disabling **only NVLS** still stalled for more than **5 minutes** at capture and
was stopped. Logs verified `NCCL_NVLS_ENABLE=0`, while NCCL channels still used
`P2P/IPC`. Consequently, disabling NVLS alone is not a sufficient workaround for
this trainer. The experiment does not isolate GPU IPC from all other shared
allocation mechanisms individually.

The socket-only NCCL configuration completed capture and resumed the original
container approximately **24 s** after the capture boundary. Full fingerprints,
log probabilities and another backward/Adam update passed. This initial capture
ran on **4 H200s**, because Modal upgraded the requested H100s. The harness pins
`H100!:4` for subsequent experiments to avoid mixing GPU types; override with
`SPINDLE_SNAPSHOT_GPU` if needed. The capture result is in
[`qwen9b-socket-capture.json`](assets/trainer-memory-snapshots/qwen9b-socket-capture.json).
It is not evidence of a separate restored container by itself.

### Explicit NCCL suspension (2026-11-06)

The follow-up uses PyTorch 2.13's backend `suspend()` / `resume()` methods on
every initialized NCCL group, with NVLS disabled and GPU P2P retained. A
pre-capture control checks that suspension alone preserves the initialized
trainer before testing platform checkpoint/restore.

The initial 9B run successfully suspended and resumed all **48 distinct NCCL
groups** (18 memberships per rank). All 58 NCCL/Gloo collective checks and
model/gradient/optimizer/RNG fingerprints matched, and the reference forward
had maximum log-probability error **0.0**. A second suspension completed in
roughly **17 ms per rank**, but capture then stalled for more than five minutes
and was cancelled. No post-capture hook or restored trainer ran in this attempt.
The transport logs show **P2P/IPC**, with `NCCL_CUMEM_ENABLE=0` set by Miles.
Evidence is in [`qwen9b-nccl-suspend.json`](assets/trainer-memory-snapshots/qwen9b-nccl-suspend.json).

Standalone subprocess controls separated NCCL behavior from the model and Ray:

| Control | Capture/restore result |
| --- | --- |
| 2 H100s, one NCCL group, NVLS off, P2P/CUMEM | Two different containers restored the saved worker PIDs and tensor sums, resumed NCCL, and passed all-reduce |
| 4 H100s, the trainer's 48 NCCL / 10 Gloo group definitions, NVLS off, P2P/CUMEM | A different container restored and passed collectives on every group; each rank resumed 18 NCCL memberships |
| 2 H100s, same single-group control with only `NCCL_CUMEM_ENABLE=0` added | Pre-capture suspend/resume passed, but capture stalled for more than two minutes and was cancelled |

The successful controls log **P2P/CUMEM**, not socket data transport. They
demonstrate functioning NCCL collectives after real snapshot restoration, but
only check a small tensor's sum and the saved processes; they do not contain a
model, gradients, optimizer, Ray cluster or training step. A second four-GPU
call initialized a new snapshot rather than restoring the first; the later
successful clone explicitly matches the first snapshot's initialization ID,
creator and worker state.

Control results and bounded transport excerpts are retained in
[`nccl-suspend-minimal.json`](assets/trainer-memory-snapshots/nccl-suspend-minimal.json),
[`nccl-suspend-groups.json`](assets/trainer-memory-snapshots/nccl-suspend-groups.json)
and [`nccl-suspend-legacy.json`](assets/trainer-memory-snapshots/nccl-suspend-legacy.json).
The exact diagnostic scripts and group layout are archived in
[`nccl-suspend-control-sources.json`](assets/trainer-memory-snapshots/nccl-suspend-control-sources.json);
extract its `source_files` under `/tmp` and run the corresponding Modal deploy/run
commands with the repository's `src` on `PYTHONPATH` to reproduce them.

### Full trainer restore with P2P/CUMEM (2026-11-06)

With `NCCL_CUMEM_ENABLE=1`, the full 9B trainer captured successfully and a
**different four-H100 container restored it**. The existing communicator objects
were retained; the hooks suspended and resumed their NCCL-managed resources.
NCCL channel logs show GPU **P2P/CUMEM**. The successful settings retain GPU P2P
and shared-memory transport support while disabling NVLS; no throughput
comparison against normal NCCL or the socket workaround was performed.

| Check | Result |
| --- | --- |
| Snapshot initialization ID | `5cf46e87aebe4de0b096c607364b4dae` in producer and clone |
| Producer container | `ta-01M6RRFDC2B3G7FKX5PBKR5SER` |
| Restored container | `ta-01M6RRVVXD6Q07N9H3QRV6S8FR` |
| Saved actor PIDs, parameters/buffers/gradients, Adam and RNG fingerprints | Exact match |
| Group checks after resume | All 48 NCCL / 10 Gloo definitions passed; 18 NCCL memberships resumed per rank |
| Reference forward | Maximum log-probability error **0.0** |
| Continued training | Backward/Adam succeeded; optimizer step advanced from 1 to 2; weights and optimizer changed |
| Producer initialization, after imports and before suspension control | **254.38 s** |
| Platform restore log to post-restore hook | Approximately **13 s**, excluding scheduling |
| NCCL resume on restored actors | Approximately **1.71 s per rank**, running concurrently |
| Verification after communicator resume | **12.79 s**, including hashes, all groups and another training step |

The platform restore log appeared at 15:50:32 UTC, the post-restore hook at
15:50:45 UTC, and communicator recovery completed at 15:50:47 UTC. These
second-resolution logs measure combined platform stages; they are not isolated
storage/GPU timings or a production persistent-cache benchmark. The earlier
snapshot-building run took 302.43 s to initialize. Another call waited for H100
capacity and built the second snapshot; only the third call actually restored.

**Numerical continuation parity remains unproven.** The clone started from
identical saved state and reproduced the first forward exactly. After the next
update, model and optimizer hashes differed from the producer's; RNG hashes
still matched. The maximum post-update log-probability difference was
**0.2948008**, and gradient norms were 570.7933 versus 570.8847. Two independent
fresh initializations also differed in model/optimizer fingerprints before
capture and in their reference outputs (maximum difference 0.1976738), despite
matching RNG fingerprints. That shows variation already exists before cloning;
it does not establish that the restored continuation's numerical differences
are acceptable or entirely explained by ordinary nondeterminism.

Raw results, per-rank suspend/resume inventories, reference/post-update outputs,
transport excerpts, source hashes and continuation comparisons are in
[`qwen9b-nccl-cumem-restored.json`](assets/trainer-memory-snapshots/qwen9b-nccl-cumem-restored.json).
This establishes a functional, single-container, TP=4 LoRA trainer restored with
GPU P2P communication. NVLS-enabled snapshots, multi-node/RDMA restoration,
full-parameter optimizer state and production frontend integration remain
untested. All experiment apps were stopped afterward.

### Full trainer restore with socket NCCL (2026-11-05)

The combined configuration passed in a **separate container on four H100s**:

| Check | Result |
| --- | --- |
| Saved initialization ID | `ee53ae3787f2494b9ed13665ab3ba1c4` in both containers |
| Original container | `ta-01M6Q0MPTF3FAA5YBASHBJ2WTR` |
| Restored container | `ta-01M6Q1QWZF3ZQMXNVGESBM688R` |
| Worker processes and group membership | Same saved PIDs and 58 group definitions; every collective passed |
| Restored parameters, buffers, gradients, Adam state and RNG | Exact per-rank hash match |
| Reference forward log probabilities | Maximum absolute error **0.0** |
| Continued training | Backward/Adam succeeded; optimizer step advanced from 1 to 2; model and optimizer hashes changed |
| Saved initialization and warmup work | **333.51 s**, excluding initial imports |
| Restore log to post-restore hook | Approximately **41 s** |
| Post-restore verification | **18.06 s**, including full tensor hashes and continued training |

The restored container logged `Restoring Function from memory snapshot.` at
23:47:11 UTC and entered the post-restore hook at 23:47:52 UTC. It did not execute
the trainer initialization hook. Evidence, complete rank inventories and hashes
are retained in [`qwen9b-restored.json`](assets/trainer-memory-snapshots/qwen9b-restored.json).
That call's client-to-result duration includes waiting behind other startup work
and must not be used as an isolated restore latency.

**Continuation parity remains unproven.** The first forward and all saved state
matched exactly, but the original container's and restored container's subsequent
updates were not byte-identical. Gradient norms were 572.6663 and 572.5220 (about
0.025% apart); post-update model and optimizer hashes differed on all ranks,
while RNG hashes remained equal. The experiment establishes a functional restored
trainer, not bitwise-equivalent training continuation. A repeated non-snapshot
control and numerical parameter/logit comparisons are needed to distinguish
normal backward/reduction nondeterminism from a restore-specific effect. The
harness now returns post-update log probabilities for that follow-up comparison.

The observed ~41 s resume versus ~334 s initialization supports the cold-start
idea for this probe. It is not a production speedup claim: the baseline used local
compilation caches, and production persistent kernel caches will change that
comparison. Socket NCCL also has a different training-throughput tradeoff from
the normal GPU P2P/NVLS configuration.

Only a short 128-token workload, rank-32 LoRA and an idle single-node capture
boundary were exercised. Engine HTTP serving, sampler publication, full-parameter
Adam state, long-context shapes and multi-node restoration were not validated.
The test apps and their remaining containers were stopped after verification;
the dedicated weights volume remains available for reproduction.

For the final 9B experiment combining the Ray fix and socket NCCL:

```bash
export SPINDLE_SNAPSHOT_APP=spindle-qwen9b-memory-snapshot-loopback-1105
export SPINDLE_SNAPSHOT_TRANSPORT=socket
export SPINDLE_SNAPSHOT_RAY_IP=127.0.0.1
uv run modal deploy tests/manual/benchmark_qwen9b_memory_snapshot.py
uv run modal run tests/manual/benchmark_qwen9b_memory_snapshot.py \
  --mode snapshot --repeats 3 --output /tmp/qwen9b-loopback.json
```

## Platform evidence and interpretation

As checked on 2026-11-05:

- [Modal's memory snapshot guide](https://modal.com/docs/guide/memory-snapshots)
  describes multi-GPU programs as generally incompatible and warns that
  snapshots do not inherently improve storage-bound model loading. Expected
  wins are skipped initialization, conversion, compilation and warmup.
- [Modal's GPU snapshot implementation overview](https://modal.com/blog/gpu-mem-snapshots)
  describes capturing VRAM, CUDA objects, kernels and memory mappings after
  draining GPU work. It does not mean an in-flight kernel resumes mid-execution.
- [NVIDIA's cuda-checkpoint documentation](https://github.com/NVIDIA/cuda-checkpoint)
  distinguishes driver generations. Version 610 adds a coordinated job mechanism
  for `cuIpcGetMemHandle` IPC, while the documented limitations still include
  UVM and IPC using `cuMemExportToShareableHandle`. The actual Modal driver matters.
- [NCCL's communicator APIs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)
  expose suspend/resume. Spindle currently has no lifecycle calls to suspend and
  resume every NCCL communicator before/after a memory snapshot. Installing a
  newer NCCL library alone does not add that integration to PyTorch.
- [NVIDIA NCCL issue 2337](https://github.com/NVIDIA/nccl/issues/2337) documents
  the NVLS multicast resource problem for CUDA checkpoint/restore. It remained
  open when checked. This is a specific compatibility concern, not proof that
  every Modal multi-GPU workload fails for that reason.

The first scope is an idle, warmed, single-container trainer. Restoring a
multi-node trainer adds coordinated capture of all containers, remote transport
and membership restoration, and rendezvous across new node addresses. Restoring
an arbitrary active training job also requires a consistent boundary for pending
commands, accumulated gradients, publication and durable control-plane state.
Modal startup snapshots are initialization templates, not automatic periodic
checkpoints of the latest training step.
