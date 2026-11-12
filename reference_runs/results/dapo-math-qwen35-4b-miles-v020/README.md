# Qwen3.5-4B with Miles v0.2.0

Validation uses Spindle `0edd81bf982088864fb0a6179352c366dcb8be8b`, Miles
`e6c491ace23353e32da863c05470f72f52293a9e`, stock Tinker SDK 0.26.0,
and Qwen/Qwen3.5-4B with rank-32 LoRA on four H100 trainer GPUs (DP 4, TP 1).
The unchanged [`dapo_math.py`](../../dapo_math.py) completed all 20 updates.
[`metrics.jsonl`](metrics.jsonl) is the unedited client output.

| Held-out evaluation | Accuracy | Mean response tokens | Truncated |
| --- | ---: | ---: | ---: |
| Before training | 61/256 (23.8%) | 1,898 | 75.0% |
| Update 5 | 57/256 (22.3%) | 1,858 | 74.2% |
| Update 10 | 70/256 (27.3%) | 1,842 | 68.0% |
| Update 15 | 76/256 (29.7%) | 1,796 | 64.1% |
| Update 20 | 92/256 (35.9%) | 1,723 | 56.6% |

All 20 optimizer updates were accepted. The run took 2,897.4 seconds from its
first record to its last, including sampler startup. The three checkpoint saves
took 19.6, 22.2, and 23.2 seconds.

| Time | Median seconds |
| --- | ---: |
| Between accepted updates | 121.6 |
| SDK sampling, 256 responses | 48.5 |
| SDK `forward_backward` | 39.9 |
| Trainer-internal backward | 10.4 |
| SDK `optim_step` | 1.8 |
| Publishing sampler weights | 19.9 |

The trainer was initialized and had run short synthetic updates before this
workload. Its first full-size backward took 107.5 seconds at the SDK. SDK backward
times varied from about 19 to 108 seconds; they include more than GPU computation.
The internal timing comes from `optimizer_metrics.timing/forward_backward_s`.

The recipe requested H100 trainer and sampler GPUs. Live `nvidia-smi` confirmed
four NVIDIA H100 80GB HBM3 trainer GPUs and two single-H100 sampler replicas,
each reporting 81,559 MiB. Samplers scaled from cold rather than starting as two
warm replicas. The driver used two CPU cores and 8 GiB memory with Tinker 0.26.0.
No deployment settings changed during the run.

## Checkpoint and slot checks

[Checkpoint results](checkpoint-validation.json) record independent CPU reads
of the native DCP tensors, plus the stock SDK losses and optimizer metrics.
Each checkpoint contains 258 weights, 258 first moments, 258 second moments,
and one optimizer-step tensor.

| Check | Result |
| --- | --- |
| Trained state restored in a new trainer container | All 775 tensors bitwise equal; nonzero moments and step 2 retained |
| Next update after restore | Accepted; saved step 3 |
| Weights-only restore | All weights bitwise equal; moments zero and step zero |
| Warm interleaved A against serial A | All 775 tensors bitwise equal |
| Warm interleaved B against serial B | All 775 tensors bitwise equal |
| Original cold interleaved B against serial B | All 775 tensors bitwise equal |
| Original cold interleaved A against serial A | 128 weights and 258 moment tensors differ; cause unresolved |

The slot probe uses one four-token cross-entropy sequence per update. A uses
input token IDs `[100, 200, 300, 400]` and shifted targets ending in `500`; B
uses `[600, 700, 800, 900]` and shifted targets ending in `1000`. All token
weights are 1. Adam uses learning rate `4e-5` and SDK defaults.

The original probe saves each fresh slot, then performs two rounds of
`backward(A), backward(B), optimizer(A), optimizer(B)`. It saves the trained
states, restores each initial state with its optimizer, and repeats two updates
serially per slot. The warm probe restores the same initial states after a cold
restart and a continuation update, repeats the interleaved sequence, and compares
against the original serial controls. It does not compare freshly randomized
adapters against one another.

The original A mismatch is confined to layers 0–19. Its maximum FP32 weight
error is `6.85557e-5`; weight relative L2 error is `5.41504e-5` over all adapter
weights. BF16 casting leaves 111 weight tensors different. These numbers do not
establish an effect on output quality or identify a cause. The maximum error
across all state tensors, `0.00194091`, belongs to second moments, not weights.

Hashes in the JSON cover the source checkpoint of each comparison. The
`continued-state` entry is a self-read used to verify saved step 3; it is not
an independent equivalence check.

## Limits

One model, topology, and SDK version were exercised. Multi-node assembly,
other models, MoE routing replay, and other GPU types were not validated.
Warm slot equality does not establish cold execution determinism. The original
serial replay did not retain losses or gradient norms; the warm replay did.
