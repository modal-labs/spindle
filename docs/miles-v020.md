# Miles v0.2.0 integration

Select `trainer_image = "miles-v0.2.0"` to use the new runtime with an existing
Miles LoRA configuration. The default image and existing model recipes have not
changed. Stock Tinker SDK checks on four H100 trainer GPUs cover training,
cold restart with exact optimizer restore, and weights-only restore. The full
20-update reference workload completed. An initial fresh-slot numerical discrepancy remains
unexplained.

The new path uses Miles' typed configuration, trainer initialization request,
per-slot optimizers, and native adapter-directory checkpoint API. It supports the
upstream THD configuration path. The older raw-model and BSHD recipes retain
their existing images; this change does not establish their compatibility with
v0.2.0. No new models or model-specific recipes are included.

## Runtime identity

- Miles release: `v0.2.0`, commit `8e419cc739fb09b5f21085aeacc7f6fd951084e1`.
- Fork: `modal-projects/miles`, branch `spindle-v020-regression-tests`, pinned commit
  `e6c491ace23353e32da863c05470f72f52293a9e`.
- Container: `radixark/miles:v0.2.0@sha256:c5e9ebe1dbb02faa094cf5036b57c4f3aadac9a53992f4d143823a6c20fb377c`.
- Bundled Megatron-LM: `809217f40d60ea671aa9958fcdbc754b33758e42`.
- Bundled Megatron-Bridge: `a0c6d6e09b0bd4d208c2d936608319d5ab456d87`.

The image retains the release's model dependencies. It adds Spindle's server,
Tinker, telemetry, and publication dependencies. The build checks the Miles
release revision, checks out the exact fork commit, and imports the runtime and
actor on a GPU. Checkpoint metadata records the installed fork commit. The branch
is used to maintain fixes; builds fetch the commit rather than its moving head.

## Downstream patch inventory

The comparison uses the pinned fork commits, not branch heads:

- v0.1.1: `fdd808c008175d4895fd8c57c95ef4fa5bb6a2a0`, relative to
  `2806267d060d51b1d3b62f85a1f9b145047aeef9`.
- v0.1.2: `f2f8447c33ac0755a0988c30d413d334dea0a26a`, relative to
  `a22fd824191c64ba7c5d6c531f125406e4ce9cc6`.
- GLM-specific image: `464efc9781bbf8edf2135f0857f5aa417685eae2`.

| Downstream behavior | v0.2.0 treatment |
| --- | --- |
| Non-gated MoE targets do not require `gate_proj` | Retained in the fork; upstream still requires all three expert targets. |
| Bridge receives `cp_comm_type` in both model builders | Retained in the fork; upstream sets CP size but omits its communication type. |
| Adapter materialization stages tensors on CPU before collecting the full list | Retained at the new iterator boundary. |
| Tinker `MASK_LOGPROB` import and renamed top-k protobuf length | Upstream uses the newer SDK names. The trainer image installs Spindle's SDK pin; legacy images keep the compatibility helper. Spindle owns the public HTTP API. |
| DeepSeek-V4.1 raw multi-LoRA targets, builders, export, TP gradient synchronization, eager CUDA loading, and fused-target rejection | Retained on the existing v0.1.2 image. Upstream has a different native adapter implementation; equivalence is not claimed. |
| DeepSeek-V4.1 FP8/FP4 loading and malformed-checkpoint errors | Retained on the existing v0.1.2 image; not migrated or claimed for the new image. |
| DeepSeek-V4.1 THD sample boundaries in attention, compression, indexing, and Engram | Retained on the existing v0.1.2 image; not migrated or claimed for the new image. |
| Engram prefetch and dynamic-shape compilation defaults | Retained on the existing v0.1.2 image. No tuning is carried into the new image. |
| GLM provider installer and model-specific dependency requirements | Existing GLM image remains unchanged. |
| Bridge grouped-import cache patch and extra DSA kernels | Existing images retain them. The new image uses bundled dependencies; GLM/DSA behavior is unverified. |

Spindle's replay and score-centering hooks remain in use. The wrapper accepts the
new adapter-slot argument and supplies vocabulary size from the typed trainer
backend. Upstream now disables LoRA MTP heads itself. Legacy expert checkpoint
repair is bypassed: the new actor uses native checkpoint members. Optional host
synchronization patches are disabled for the new dependency versions.

## Checkpoint compatibility

| Source | Destination | Policy |
| --- | --- | --- |
| `miles_torch_dist` from an existing image | Same Miles revision, model, topology, rank, alpha, and target flags | Existing restore path. |
| `miles_torch_dist` | v0.2.0 | Rejected. Keep the original image to restore it. |
| `miles_adapter_state` | Same v0.2.0 fork revision, model, topology, rank, alpha, dropout, exact targets, and target flags | Native restore path; exact trained-state restore checked on Qwen3.5-4B with four H100 GPUs. |
| Different revision, model, topology, rank, alpha, or target flags | Any runtime | Rejected before slot loading. |
| Weights-only checkpoint | Optimizer restore | Rejected. |

The existing metadata schema remains version 1. `checkpoint_format` distinguishes
the native `miles_adapter_state` layout from `miles_torch_dist`; no metadata
schema bump is needed. Miles owns the native tensor and optimizer serialization.
A weights-only save omits moments through Miles' checkpoint-member interface.
A weights-only load passes `load_optimizer=False` and resets Spindle's step count.

Native captures use unique directories on the checkpoint volume.
Every node commits its native files before capture assembly. The driver reads
the committed volume listing, copies complete local files or downloads missing
files into a unique staging directory, and checks every shard
referenced by DCP metadata, including its byte extent. Sampler captures require
`adapter_config.json` and final `adapter_model.safetensors`. Unrelated files and
temporary files cannot satisfy these checks. Assembly does not reload the
volume while other operations may have files open.

The capture is exposed only after validation. Native checkpoint persistence
installs this complete local copy through a staging-directory rename before
committing the volume. Cleanup is limited to the native
operation directory and its staging directory. A failed assembly leaves the
native source available for inspection and invalidates the runtime. Checkpoint
capture and persistence serialize volume access; native sampler publication
also holds that lock so a restore cannot reload its source during publication.

## Validation

Spindle `0edd81bf982088864fb0a6179352c366dcb8be8b`, Miles fork `e6c491a`,
Qwen/Qwen3.5-4B, rank 32, four H100 trainer GPUs (DP 4, TP 1), stock Tinker
SDK 0.26.0:

- A trained checkpoint restored in a new trainer container with all 775 tensors
  bitwise equal, including nonzero Adam moments and optimizer step 2.
- The next update was accepted and saved optimizer step 3.
- The full reference run's trained checkpoint also restored in a new container
  with all 775 tensors bitwise equal, including moments and optimizer step 20.
- Weights-only restore preserved every weight exactly and reset moments and step
  to zero.
- Two interleaved updates on each of two restored slots matched the serial
  controls bitwise across all 775 tensors per slot.
- The original cold run differed from its serial control for one slot: 128
  weights and 258 moment tensors in layers 0–19. The second slot matched.
  Warm replay did not reproduce the difference. Restoring the same initial state
  before any backward and comparing two cold versus warm serial updates also
  matched bitwise after both steps. The original cause remains unresolved;
  cross-slot contamination was not demonstrated.

The [full DAPO-Math reference run](../reference_runs/results/dapo-math-qwen35-4b-miles-v020/README.md)
completed 20 accepted updates, 21 sampler publications, five held-out evaluations,
and three checkpoint saves. Held-out accuracy went from 61/256 (23.8%) to 92/256
(35.9%). Median SDK sampling, backward, and publication times were 48.5, 39.9,
and 19.9 seconds; median trainer-internal backward was 10.4 seconds. The fresh
v0.1.1 reference finished at 107/256 (41.8%), with 16.6-second median SDK backward
and 7.9-second publication. The new path was slower in this pair. Different
warm-up and autoscaling settings, and only one run per runtime, limit attribution;
learning and performance parity are not established.

An earlier two-update
smoke run used Spindle `75b3e49` and Miles `949b877`; it completed training,
sampler publication, generation, and a trained checkpoint save. Those results
are not a full reference-run comparison or evidence for the current image pin.

CPU checks after rebasing onto the fixed-hardware reference recipe passed:
1,305 tests, 8 skipped, and all pre-commit checks. Tests cover
legacy runtime behavior, native API translation, capture ownership, required
files, missing remote shards, and a synthetic DCP round trip.

Multi-node shard assembly, routing-replay training, other models, and other GPU
topologies remain unverified on the new image. Exact checkpoint identity includes
the fork commit, so even a test-only fork commit requires matching revisions at
save and restore.
