# Deterministic training (experimental)

Use a deterministic recipe to enable the Miles multi-LoRA training path. Tinker
training code stays the same.

| Recipe | Trainer | Attention |
| --- | --- | --- |
| `qwen3-8b-lora-32k-deterministic` | 4 H100, TP4 / DP1 | native deterministic FA3 |
| `qwen35-9b-lora-16k-deterministic` | 4 H100, TP4 / DP1 | patched FA3 hdim256 and deterministic GDN |

## Qwen3-8B

Select the intended Modal workspace and environment, then validate and deploy:

```bash
spindle config validate src/spindle/configs/qwen3_8b_lora_32k_deterministic.py
spindle deploy --app spindle-deterministic qwen3-8b-lora-32k-deterministic
```

## Qwen3.5-9B

Build the patched FA3 artifacts against Spindle's pinned Miles image, then set
their directory before validation and deployment:

```bash
PYTHONPATH=src python scripts/export_fa3_test_wheel.py
export SPINDLE_MILES_FA3_ARTIFACTS="$PWD/scripts/results/fa3_deterministic/artifacts"
spindle config validate src/spindle/configs/qwen35_9b_lora_16k_deterministic.py
spindle deploy --app spindle-deterministic qwen35-9b-lora-16k-deterministic
```

Keep the exported wheel and manifest together. The trainer image verifies the
artifact revision and checksums. Missing artifacts cause deployment to fail.
The patched attention build targets Hopper SM90 and head dimension 256.

## Custom recipes

Set these Miles options, with TP equal to the trainer GPU count and CP=1:

```python
miles_cfg = {
    # Include the model's other Miles settings.
    "tensor_model_parallel_size": 4,
    "context_parallel_size": 1,
    "deterministic_training": True,
    "tp_reduce_precision": "float64",
}
```

For Qwen3.5, also set `deterministic_gdn=True`, use
`trainer_image="miles-deterministic-fa3"`, and supply the FA3 artifacts above.
Set these environment variables in `trainer_env` before CUDA starts:

```python
trainer_env = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
}
```

Keep `deterministic_pack_clients=False` (the default) so each client uses separate
microbatches. Client isolation and FP64 reductions have a throughput cost.

## Requirements

The mode requires Hopper SM90, DP1/CP1, dynamic batching, and zero LoRA dropout.
MoE, expert parallelism, pipeline interleaving, multimodal inputs, MTP, and
per-token loss normalization are unsupported. This mode applies to Miles LoRA
training.

For repeatable runs, keep hardware, dependency versions, TP layout, ordered data,
initial weights, optimizer state, and forward/backward and update boundaries
fixed. Fresh adapters use the trainer seed (`miles_cfg.cli_options["seed"]`);
per-model Tinker seeds are unsupported. Restore a model-and-optimizer checkpoint
to replay an existing run.

Both recipes enable deterministic SGLang inference. Custom recipes should set
`sglang_cfg["enable_deterministic_inference"]=True` when reproducible sampling is
also needed; fixed training inputs are still required for training replay.
