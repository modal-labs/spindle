# Miles v0.2.0 integration

Select `trainer_image = "miles-v0.2.0"` to use the new runtime with an existing
Miles LoRA configuration. The default image and existing model recipes have not
changed. The new image supports single-node trainers in `megatron_to_hf_mode`
bridge; Spindle rejects it with several nodes or raw mode.

The new path uses Miles' typed configuration, trainer initialization request,
per-slot optimizers, and native adapter-directory checkpoint API. It supports the
upstream THD configuration path. The older raw-model and BSHD recipes retain
their existing images. No new models or model-specific recipes are included.

Spindle passes `--lora-type lora` to every Miles image, so v0.2.0 trains the same
adapter as the existing images: one `lora_A` shared across each fused projection
(q/k/v, and gate/up). Miles v0.2.0 would otherwise default to `canonical_lora`,
which gives every projection its own `lora_A`. Sampler exports are standard PEFT
adapters on both images, with the shared `lora_A` copied into each projection.

## Runtime identity

- Miles release: `v0.2.0`, commit `8e419cc739fb09b5f21085aeacc7f6fd951084e1`.
- Fork: `modal-projects/miles`, branch `spindle-v020-regression-tests`, pinned commit
  `a94f7ae2ab58e6f05ba6747a48a8c43f6f4ef855`.
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

The new actor installs only Spindle's replay, score-centering, distribution
sharding and Qwen3-VL context-parallel position patches, through
`actor_shared.install_replay_patches`. It does not import the legacy actor, so
the host synchronization patches, BSHD context-parallel LoRA batching,
cross-node `write_checkpoint_dir` publishing and expert checkpoint repair stay
on the existing images. The wrapper accepts the new adapter-slot argument and
supplies vocabulary size from the typed trainer backend. Upstream now disables
LoRA MTP heads itself.

## Checkpoint compatibility

| Source | Destination | Policy |
| --- | --- | --- |
| `miles_torch_dist` from an existing image | Same Miles revision, model, topology, rank, alpha, and target flags | Existing restore path. |
| `miles_torch_dist` | v0.2.0 | Rejected. Keep the original image to restore it. |
| `miles_adapter_state` | Same v0.2.0 fork revision, model, topology, rank, alpha, exact targets, and target flags | Native restore path; exact trained-state restore checked on Qwen3.5-4B with four H100 GPUs. |
| Different revision, model, topology, rank, alpha, or target flags | Any runtime | Rejected before slot loading. |
| Weights-only checkpoint | Optimizer restore | Rejected. |

Rejecting `miles_torch_dist` on v0.2.0 is Spindle's policy, not a format limit:
Miles v0.2.0 ships a loader for the older torch_dist slot format. It needs a
`META.json` that Spindle's saves never wrote, so restoring older checkpoints on
v0.2.0 would need a tested migration step.

The existing metadata schema remains version 1. `checkpoint_format` distinguishes
the native `miles_adapter_state` layout from `miles_torch_dist`; no metadata
schema bump is needed. Miles owns the native tensor and optimizer serialization.
A weights-only save omits moments through Miles' checkpoint-member interface.
A weights-only load passes `load_optimizer=False` and resets Spindle's step count.

Captures stay on the trainer's local disk, as on the existing images, and
persisting a checkpoint copies it to the checkpoint volume with one commit.

Miles writes each capture to an uncommitted directory. Spindle checks it before
renaming it into place: a checkpoint must have every shard its DCP metadata
references, at its full byte extent, and a sampler export must have
`adapter_config.json` and a non-empty `adapter_model.safetensors`. A capture that
fails these checks fails only its request and stays in place for inspection;
the trainer remains available.

## Validation

Stock Tinker SDK 0.26.0, Qwen/Qwen3.5-4B, rank-32 LoRA, Spindle `016debf`, Miles
fork `a94f7ae`, four H100 trainer GPUs (DP 4, TP 1):

- The [20-update DAPO-Math reference run](../reference_runs/results/dapo-math-qwen35-4b-miles-v020/)
  accepted every update. Held-out accuracy went from 51/256 (19.9%) to 103/256
  (40.2%); the [Miles v0.1.1 reference](../reference_runs/results/dapo-math-qwen35-4b/README.md)
  went from 53/256 (20.7%) to 107/256 (41.8%). One run per image does not
  establish learning parity. Median trainer backward was 9.9 seconds, against
  8.9 on v0.1.1.
- A trained step-20 checkpoint restored with optimizer state into a cold
  trainer, and saved again unchanged, matched the original bitwise in all 210
  weights, all 420 Adam moments and the step.
- A weights-only restore (`load_state`) kept every fp32 weight bitwise, reset
  all moments to zero and reset the step to 0. On v0.1.1 the same restore rounds
  the fp32 master weights to bf16.
- Restoring and then training continued from step 20 to step 21.
- Two models trained with interleaved requests on one trainer ended as close to
  their serial controls as two serial runs of the same work end to each other.
- Training is not bitwise reproducible on either image: two clients restoring
  the same checkpoint and taking the same update differed by up to about 40% of
  the update's size, largest in the first layers and zero in the last. Qwen3.5
  recipes do not request deterministic kernels.
- The sampler export loaded in SGLang with the same 258 bf16 tensor names and
  shapes as the v0.1.1 export, and identical `lora_A` copies within each fused
  group.
- The same recipe on the existing v0.1.1 image completed a smoke run on the final
  runtime code, with its legacy patches installed.

CPU checks passed: 1,324 tests, 8 skipped, and all pre-commit checks.

The run kept the frontend, trainer and samplers in `us-east`
(`platform.modal.region`). An earlier run with the trainer region unpinned saw
request uploads from the frontend to the trainer stall for 20 to 50 seconds on 7
of 20 updates; that path does not depend on the Miles version.

Multi-node trainers, raw and BSHD recipes, routing replay, other models and
other GPU topologies are unverified on the new image. Checkpoint identity
includes the fork commit, so any fork bump requires matching revisions at save
and restore.
