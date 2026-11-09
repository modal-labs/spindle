# Quantized sampler comparison

`qwen35_4b_fft_fp8_64k` uses the Qwen3.5-4B full-parameter BF16 trainer
with an FP8 sampler. The BF16 control is `qwen35_4b_fft_64k`.
Full-parameter publications remain BF16 deltas. CPU staging applies those
deltas before SGLang compiles a new quantized serving image.
This recipe does not enable FP8 training compute. The 4B FP8 recipe has config
coverage but has not been validated on GPUs; the measured comparison below uses
MXFP8 on the 35B MoE.

```bash
uv run spindle config init --preset qwen35_4b_fft_fp8_64k > fp8.py
uv run spindle config validate fp8.py
```

After deploying, [the score-centering example](../score-centering/train.py)
checks sampling and one update. It does not publish the updated weights or
measure learning quality, so it is insufficient to validate quantized sync.

## Matched comparison

Use separate isolated deployments for BF16 and FP8, with identical model
revision, trainer precision, GPU type/count, parallelism, context length,
sampler replicas, batch size, prompts, generation limits, optimizer and loss.
Use the same fixed evaluation prompts and evaluation sampling settings.
Record the actual GPU model and source revisions. Measure cold start separately.

First check both modes at version zero, then update, publish, and sample again
for multiple weight versions. Require finite losses and gradients, successful
activation of each publication, and changed probabilities after updates.
Capture sampling probabilities with a bounded support, for example `top_k=64`,
and pass the recorded support and temperature to `score_centered_mis`.
Never truncate supports after sampling.

| Measurement | Reporting |
| --- | --- |
| Rollout | Seconds, generated tokens, and completion lengths |
| Forward/backward | Seconds and trained tokens |
| Optimizer | Seconds |
| Publication | Seconds until the first successful sample on the new weights |
| Complete update | Wall seconds including replay capture and data preparation |
| Quality | Held-out reward versus both updates and cumulative wall seconds |
| Distribution mismatch | Replay ratio and mask diagnostics from training |
| Memory | Peak GPU memory for trainer and sampler separately |

Use identical fixed token batches to isolate trainer execution cost; on-policy
rollouts can produce different lengths and supports. Report both fixed-work
and on-policy measurements. Keep replay capture enabled for both modes in the
primary comparison. A BF16 ordinary-PPO run can separately measure replay-loss
overhead, but changes the objective as well as the capture workload.

Exclude startup and compilation warmup from steady-state medians, retaining
the raw per-step measurements. Report FP8/BF16 seconds ratios for each phase
and total update time. A short smoke run verifies operation and timing; it
does not establish convergence or quality parity.

Other SGLang quantization modes, including `nvfp4_online`, pass through config
resolution but need their own hardware and publication validation. LoRA uses a
quantized frozen base and BF16 adapters, so its results are a separate comparison.

## MoE on Blackwell

The [matched Qwen3.5-35B-A3B configurations](../../docs/assets/quantized-sampler/b300-comparison-configs.json)
use four B300 trainer GPUs and one B300 sampler GPU per run. Both trainers use
BF16. The MXFP8 sampler leaves the small `in_proj_ba` gate in BF16 because its
64 output channels are below the selected MXFP8 kernel's minimum width.

Create a Python deployment file from either configuration, giving it a unique
name and your usual platform settings:

```python
import json
from pathlib import Path

from spindle.configuration import BaseConfig

settings = json.loads(
    Path("docs/assets/quantized-sampler/b300-comparison-configs.json").read_text()
)
config = BaseConfig(name="my-moe-mxfp8", **settings["mxfp8"])
```

Deploy the BF16 and MXFP8 configurations behind separate frontends. Point
`TINKER_BASE_URL` and `TINKER_API_KEY` at each matching frontend, then run:

```bash
uv run --with tinker==0.25.0 --with transformers --with jinja2 \
  python examples/quantized-sampler/benchmark.py \
  --mode mxfp8 --steps 10 --output results/mxfp8.json
```

Repeat against the BF16 frontend with `--mode bf16`. `--mode` labels the result;
it does not change server precision. Proxy-protected frontends also need
`MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET`.

The benchmark uses batch 32, a 128-token generation cap, temperature 1, top-p
0.95, top-k 64, `score_centered_mis`, and Adam learning rate 1e-6. It excludes
the first two updates from steady-state medians. Each update checks finite
metrics, positive gradients, a successful optimizer update, and serving the
new publication. Fixed-sequence log probabilities check that published updates
change the policy over the run.

Generated lengths can differ between precisions; compare the recorded token
counts as well as seconds. Total time includes fixed-sequence probability
probes used to validate publication. Arithmetic rewards are an operation check,
not a convergence evaluation. See [validation evidence](../../docs/quantized-sampler-validation.md).
