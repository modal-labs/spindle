import json
import types

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

from spindle.backends.miles_runtime.gdn_lora import (  # noqa: E402
    freeze_gdn_gate_adapters,
    merge_gdn_input_adapters,
)

LAYER = "base_model.model.model.language_model.layers.0"
RANK, HIDDEN, QKV, V_DIM, HEADS = 4, 8, 12, 4, 2


def write_adapter(path, gate_value=0.0):
    a = torch.randn(RANK, HIDDEN)
    tensors = {
        f"{LAYER}.self_attn.q_proj.lora_A.weight": torch.randn(RANK, HIDDEN),
        f"{LAYER}.self_attn.q_proj.lora_B.weight": torch.randn(6, RANK),
        f"{LAYER}.linear_attn.in_proj_qkv.lora_A.weight": a,
        f"{LAYER}.linear_attn.in_proj_qkv.lora_B.weight": torch.randn(QKV, RANK),
        f"{LAYER}.linear_attn.in_proj_z.lora_A.weight": a.clone(),
        f"{LAYER}.linear_attn.in_proj_z.lora_B.weight": torch.randn(V_DIM, RANK),
    }
    for gate in ("in_proj_b", "in_proj_a"):
        tensors[f"{LAYER}.linear_attn.{gate}.lora_A.weight"] = a.clone()
        tensors[f"{LAYER}.linear_attn.{gate}.lora_B.weight"] = torch.full((HEADS, RANK), gate_value)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    modules = [f"model.language_model.layers.0.{name}" for name in (
        "self_attn.q_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
    )]
    (path / "adapter_config.json").write_text(json.dumps({"r": RANK, "target_modules": modules}))
    return tensors


def test_export_serves_linear_attention_as_one_qkvz_adapter(tmp_path):
    original = write_adapter(tmp_path)

    merge_gdn_input_adapters(tmp_path)

    merged = load_file(str(tmp_path / "adapter_model.safetensors"))
    attn = f"{LAYER}.linear_attn"
    assert sorted(name.removeprefix(f"{LAYER}.") for name in merged) == [
        "linear_attn.in_proj_qkvz.lora_A.weight",
        "linear_attn.in_proj_qkvz.lora_B.weight",
        "self_attn.q_proj.lora_A.weight",
        "self_attn.q_proj.lora_B.weight",
    ]
    assert torch.equal(merged[f"{attn}.in_proj_qkvz.lora_A.weight"], original[f"{attn}.in_proj_qkv.lora_A.weight"])
    assert torch.equal(
        merged[f"{attn}.in_proj_qkvz.lora_B.weight"],
        torch.cat([original[f"{attn}.in_proj_qkv.lora_B.weight"], original[f"{attn}.in_proj_z.lora_B.weight"]]),
    )
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["target_modules"] == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkvz",
        "model.language_model.layers.0.linear_attn.out_proj",
    ]


def test_export_refuses_a_trained_gate_adapter(tmp_path):
    write_adapter(tmp_path, gate_value=0.5)

    with pytest.raises(ValueError, match="in_proj_b adapter is not zero"):
        merge_gdn_input_adapters(tmp_path)


class MultiLoRALinear(torch.nn.Module):
    def __init__(self, name, out_features, num_value_heads, slots=2):
        super().__init__()
        self.base_linear_name = name
        self.to_wrap = types.SimpleNamespace(config=types.SimpleNamespace(linear_num_value_heads=num_value_heads))
        self.adapters = torch.nn.ModuleList(
            torch.nn.Module() for _ in range(slots)
        )
        for adapter in self.adapters:
            adapter.linear_out = torch.nn.Linear(RANK, out_features, bias=False)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_trainer_keeps_linear_attention_gates_frozen(tensor_parallel_size):
    num_value_heads, local_rows = 4, 16
    gate_rows = 2 * num_value_heads // tensor_parallel_size
    gdn = MultiLoRALinear("decoder.layers.0.self_attention.in_proj", local_rows, num_value_heads)
    mamba = MultiLoRALinear("decoder.layers.1.mixer.in_proj", local_rows, num_value_heads)
    chunk = torch.nn.ModuleList([gdn, mamba])

    assert freeze_gdn_gate_adapters([chunk], tensor_parallel_size) == 1

    for module in (gdn, mamba):
        for adapter in module.adapters:
            adapter.linear_out(torch.ones(1, RANK)).sum().backward()
    for adapter in gdn.adapters:
        grad = adapter.linear_out.weight.grad
        assert not grad[-gate_rows:].any()
        assert grad[:-gate_rows].all()
    assert all(adapter.linear_out.weight.grad.all() for adapter in mamba.adapters)
