# Tested speculative serving configurations

These SGLang configurations preserve the model revisions and serving settings
used in the comparisons below. They target SGLang v0.5.23 with
[modal-projects/sglang commit 90785332b8e3](https://github.com/modal-projects/sglang/commit/90785332b8e3e1202ae280d85ecec197473bd5a8).
Numerical parity remains unresolved;
these measurements do not certify equivalent sampling distributions.

| Config | Allocation | Tested scope |
| --- | --- | --- |
| `qwen3-8b-dflash-b300.yaml` | 1 B300 | DAPO inference with four trained adapters, 128 running requests |
| `qwen35-9b-dflash-b200.yaml` | 1 B200 sampler | Ten DAPO LoRA training updates, paired with a separate B200 trainer |
| `qwen38-27b-dflash-b200.yaml` | 2 B200 | DAPO inference with two trained adapters, 64 running requests |
| `glm53-flash-dflash-b200.yaml` | 4 B200 | Native FP8 DAPO inference with two trained adapters, 64 running requests |
| `qwen35-4b-mtp-b200.yaml` | 1 B200 | Native MTP numerical/IFEval controls; no trained-adapter workload |

Run a configuration inside the pinned sampler environment:

```bash
python -m sglang.launch_server --config examples/speculative-decoding/qwen38-27b-dflash-b200.yaml
```

Load your adapters through SGLang's adapter endpoint after startup. Adapter paths,
datasets, generation limits, and client concurrency belong to the request workload;
the YAML files configure the server. The DAPO load comparisons allowed 16,384
generated tokens. The MTP control used a different workload and is not a DFlash run.

For Spindle deployments, the bundled recipes
`qwen35_9b_instruct_lora_32k_dflash_b200.py` and
`qwen38_27b_lora_32k_dflash_b200.py` download their fixed draft checkpoints
automatically. The GLM YAML preserves the tested native FP8 target; it is not a
validation of GLM training on FP8 weights or of the BF16 training recipe.

The Qwen 27B preset reserves 384 recurrent-state slots. Without this explicit
budget, the tested default capped the requested 64 running requests at 36.
The GLM preset uses FA4 draft attention and a 256-token logical page size;
TRT-LLM draft attention triggered an incompatible page-size override in this fork.

Run reports and plots are posted in the conversation on
[PR #261](https://github.com/modal-projects/spindle/pull/261). Experiment results
and validation artifacts are not included in these configuration files.
