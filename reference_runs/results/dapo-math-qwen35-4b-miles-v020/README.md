# Qwen3.5-4B with Miles v0.2.0

This run used Spindle `619dc8b`, Miles fork `a94f7ae2ab58e6f05ba6747a48a8c43f6f4ef855`
on release `v0.2.0`, stock Tinker SDK 0.26.0, and Qwen/Qwen3.5-4B with rank-32
LoRA. The unchanged [`dapo_math.py`](../../dapo_math.py) completed all 20 updates.
[`metrics.jsonl`](metrics.jsonl) is the unedited client output.

The deployment is the [reference recipe](../../dapo_math_config.py) with
`trainer_image = "miles-v0.2.0"` ([`deployment.py`](deployment.py)): four H100
trainer GPUs (DP 4, TP 1) with one client slot, and two fixed single-H100
sampler replicas. A two-update smoke run warmed it first. The client recorded
this hardware contract; GPU models were not separately checked with `nvidia-smi`.

| Held-out evaluation | Accuracy | Mean response tokens | Truncated |
| --- | ---: | ---: | ---: |
| Before training | 48/256 (18.8%) | 1,899 | 78.9% |
| Update 5 | 67/256 (26.2%) | 1,858 | 71.1% |
| Update 10 | 67/256 (26.2%) | 1,828 | 68.4% |
| Update 15 | 87/256 (34.0%) | 1,783 | 60.5% |
| Update 20 | 95/256 (37.1%) | 1,725 | 54.7% |

All 20 updates were accepted. The run took 2,197 seconds from its first record
to its last.

## Comparison with Miles v0.1.1

The [Miles v0.1.1 reference](../dapo-math-qwen35-4b/README.md) used the same
recipe, client, SDK, model, data, seed and LoRA layout: one `lora_A` shared
across each fused projection (q/k/v, and gate/up).

| Measurement | Miles v0.1.1 | Miles v0.2.0 |
| --- | ---: | ---: |
| Held-out accuracy, before / update 20 | 20.7% / 41.8% | 18.8% / 37.1% |
| Median trainer backward seconds | 8.9 | 9.6 |
| Median trainer sampler export seconds | 0.55 | 0.16 |
| Median sampling seconds | 45.3 | 48.0 |
| Median SDK backward seconds | 16.6 | 19.7 |
| Median publication seconds | 7.9 | 13.4 |
| Median seconds between updates | 73.5 | 90.9 |
| Checkpoint save seconds, updates 0 / 10 / 20 | 4.8 / 14.3 / 10.0 | 19.4 / 12.9 / 14.2 |

Each evaluation is 256 samples at temperature 1.0, so a single point varies by
about three percentage points, and each runtime has one run. The final
accuracy gap is within what two runs can show; learning parity is not
established.

The trainer's own timings are close. Most of the wall-clock difference is
outside the trainer:

- On 7 of 20 updates, the SDK waited 33 to 53 seconds for a `forward_backward`
  that took about 10 seconds on the trainer. The request upload from the
  frontend to the trainer stalled; the trainer was idle meanwhile.
- Copying each sampler export to the bulletin volume and committing it took a
  median 12.0 seconds, against 4 to 7 seconds in other runs of the same file
  size.

Neither path depends on the Miles version. A run of the same build in
`canonical_lora` layout on this recipe showed neither effect (14.0 seconds SDK
backward, 5.9 seconds publication), so treat these timings as one sample of
infrastructure variance.

## What changed after this run

The runtime code changed after `619dc8b`: the actors share helpers through
`actor_shared.py` without importing the legacy actor, captures are validated
outside the runtime call, and multi-node and raw-mode recipes are rejected.
Two-update smoke runs on those commits completed training, sampler publication
and checkpoint saves on the same deployment. They do not repeat this 20-update
run.
