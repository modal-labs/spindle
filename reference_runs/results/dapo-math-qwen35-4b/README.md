# DAPO-Math GRPO on Qwen3.5-4B

One complete 20-update run of [`dapo_math.py`](../../dapo_math.py) with its
default settings and the fixed [`deployment recipe`](../../dapo_math_config.py).
[`metrics.jsonl`](metrics.jsonl) is the unedited output.

| Held-out evaluation | Accuracy | Mean response tokens | Truncated |
| --- | ---: | ---: | ---: |
| Before training | 53/256 (20.7%) | 1,896 | 78.1% |
| Update 5 | 63/256 (24.6%) | 1,866 | 72.7% |
| Update 10 | 73/256 (28.5%) | 1,786 | 66.0% |
| Update 15 | 104/256 (40.6%) | 1,722 | 52.3% |
| Update 20 | 107/256 (41.8%) | 1,648 | 47.3% |

Each evaluation samples two responses to each of 128 held-out problems with a
2,048-token limit.

| Client-side time | Median seconds |
| --- | ---: |
| Between accepted updates | 73.5 |
| Sampling 256 responses | 45.3 |
| `forward_backward` | 16.6 |
| `optim_step` | 1.2 |
| Publishing sampler weights | 7.9 |

All 20 optimizer updates were accepted. The run took 1,766 seconds from
its first record to its last. Checkpoint saves at updates 0, 10 and 20 took
4.8, 14.3, 10.0 seconds, respectively.

## Provenance

- Client and deployment recipe: `e24a4fd`, stock Tinker SDK 0.26.0.
- Server: Spindle 0.1.6, deployed from `e24a4fd`. The trainer's runtime Python
  files were checked against that source; authoring configs are passed as JSON.
- Trainer image: Miles v0.1.1 with the Spindle fork at
  `fdd808c008175d4895fd8c57c95ef4fa5bb6a2a0`, verified in the executing container.
- Requested trainer hardware: `H100!:4`, one instance and one client slot,
  with tensor parallelism 1 and data parallelism 4.
- Requested sampler hardware: two fixed replicas, each requesting `H100!:1`.
- Verified allocation: four NVIDIA H100 80GB HBM3 trainer GPUs and two sampler
  replicas with one NVIDIA H100 80GB HBM3 GPU each.
- GPU model and memory were checked with `nvidia-smi` in the trainer and both
  sampler containers: 81,559 MiB per GPU. The recipe uses `H100!` to disable
  Modal's automatic H200 upgrades and fixes the sampler replica count at two.
- Platform-only overrides selected the app and environment, enabled Modal proxy
  authentication, and disabled the optional Hugging Face secret for this public
  model. Storage used the recipe's default volumes. These overrides did not
  change trainer or sampler settings.
- A two-update smoke run warmed the deployment before this run. The deployment
  was dedicated to this workload, with no other training or sampling clients.

## Limits

- One run with one seed. Evaluations reuse the same problems and sampling seed,
  so they are not independent across updates.
- Timings exclude deployment, model loading, tokenizer and dataset loading, and
  training-client creation. Warm kernels may still compile for new input shapes.
- Times between updates include publication, evaluation and checkpoint saves.
- Nothing restores the saved states; this run verifies saving, not recovery.
- Hardware metadata describes the fixed recipe. For comparisons, verify actual
  trainer and sampler hardware and replica counts as well as workload settings.
