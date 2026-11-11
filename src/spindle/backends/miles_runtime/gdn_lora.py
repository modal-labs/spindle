"""LoRA on Qwen3.5 linear-attention (Gated DeltaNet) input projections.

Megatron fuses the q, k, v, z, b and a projections into one ``in_proj``, so Miles
trains one adapter over all six. The sampler serves q, k, v and z as one
``in_proj_qkvz`` adapter and has no LoRA for the b and a gates, so the trainer
keeps the gate rows of every adapter at zero and the export drops them.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_ADAPTER_CONFIG = "adapter_config.json"


def freeze_gdn_gate_adapters(model, tensor_parallel_size: int) -> int:
    """Zero the b and a rows of each GDN in_proj adapter's gradient.

    Megatron lays out each tensor-parallel shard of in_proj as [q, k, v, z, b, a],
    so the gates are the last rows of every local adapter output.
    """
    wrapped = 0
    for chunk in model:
        for module in chunk.modules():
            if type(module).__name__ != "MultiLoRALinear" or not module.base_linear_name.endswith(
                "self_attention.in_proj"
            ):
                continue
            gate_rows = 2 * module.to_wrap.config.linear_num_value_heads // tensor_parallel_size
            for adapter in module.adapters:
                adapter.linear_out.weight.register_hook(partial(_zero_last_rows, rows=gate_rows))
            wrapped += 1
    return wrapped


def _zero_last_rows(grad: torch.Tensor, *, rows: int) -> torch.Tensor:
    grad = grad.clone()
    grad[-rows:] = 0
    return grad


def merge_gdn_input_adapters(path: Path) -> None:
    """Rewrite an exported adapter's GDN q/k/v, z, b, a tensors as one in_proj_qkvz adapter."""
    weights = load_file(str(path / _ADAPTER_WEIGHTS))
    qkv_a_names = [name for name in weights if name.endswith(".in_proj_qkv.lora_A.weight")]
    if not qkv_a_names:
        return
    for a_name in qkv_a_names:
        prefix = a_name.removesuffix(".in_proj_qkv.lora_A.weight")
        for gate in ("in_proj_b", "in_proj_a"):
            del weights[f"{prefix}.{gate}.lora_A.weight"]
            gate_b = weights.pop(f"{prefix}.{gate}.lora_B.weight")
            if gate_b.any():
                raise ValueError(f"{prefix}.{gate} adapter is not zero; the sampler cannot serve it")
        del weights[f"{prefix}.in_proj_z.lora_A.weight"]
        weights[f"{prefix}.in_proj_qkvz.lora_A.weight"] = weights.pop(a_name)
        weights[f"{prefix}.in_proj_qkvz.lora_B.weight"] = torch.cat(
            [
                weights.pop(f"{prefix}.in_proj_qkv.lora_B.weight"),
                weights.pop(f"{prefix}.in_proj_z.lora_B.weight"),
            ]
        )
    save_file(weights, str(path / _ADAPTER_WEIGHTS), metadata={"format": "pt"})

    config = json.loads((path / _ADAPTER_CONFIG).read_text())
    modules = []
    for module in config["target_modules"]:
        prefix, _, leaf = module.rpartition(".")
        if leaf == "in_proj_qkv":
            modules.append(f"{prefix}.in_proj_qkvz" if prefix else "in_proj_qkvz")
        elif leaf not in ("in_proj_z", "in_proj_b", "in_proj_a"):
            modules.append(module)
    config["target_modules"] = modules
    (path / _ADAPTER_CONFIG).write_text(json.dumps(config))
