# DAPO-Math GRPO on Qwen3.5-4B

One complete 20-update run of [`dapo_math.py`](../../dapo_math.py) with its
default settings. [`metrics.jsonl`](metrics.jsonl) is the unedited output.

| Held-out evaluation | Accuracy | Mean response tokens | Truncated |
| --- | ---: | ---: | ---: |
| Before training | 41/256 (16.0%) | 1,903 | 82.0% |
| Update 5 | 57/256 (22.3%) | 1,891 | 77.0% |
| Update 10 | 63/256 (24.6%) | 1,858 | 71.9% |
| Update 15 | 72/256 (28.1%) | 1,809 | 65.2% |
| Update 20 | 91/256 (35.5%) | 1,762 | 59.4% |

Each evaluation samples 2 responses to each of 128 held-out problems with a
2,048-token limit. Most of the gain comes from the model finishing its answer
within the limit more often.

| Client-side time | Median seconds |
| --- | ---: |
| Between accepted updates | 38.1 |
| Sampling 256 responses | 22.0 |
| `forward_backward` | 11.1 |
| `optim_step` | 0.6 |
| Publishing sampler weights | 3.5 |

All 20 optimizer updates were accepted. The run took 950 seconds from its first
record to its last. The three `save_state` calls took 3.5, 8.4 and 10.8 seconds.

## Provenance

- Client: this repository at `821b648`, tinker 0.26.0.
- Server: Spindle 0.1.6 at `b5d04c8`, the `qwen35-4b-lora-32k` recipe (four H100
  trainer GPUs). The server reported this commit shortly before the run; the
  script does not record it.

## Limits

- One run with one seed. One evaluation response is 0.4 percentage points, and
  the evaluations are not independent across updates because they reuse the
  sampling seed.
- The trainer and sampler were already warm, so the times exclude cold starts.
  The server is shared; other clients may have been using it.
- Nothing restores from the saved states, so the run shows that `save_state`
  succeeds, not that a restore works.
