from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from spindle.backends.miles_config import MilesBackendConfig, parse_backend_config
from spindle.backends.miles_runtime.initialization import seeded_slot_initialization
from spindle.backends.miles_runtime.precision import _CastAfterWait
from spindle.deployments import DeploymentConfig
from spindle.configs.qwen3_8b_lora_32k_deterministic import Config as Qwen3
from spindle.configs.qwen35_9b_lora_16k_deterministic import Config as Qwen35


def deterministic_config(**kwargs):
    return MilesBackendConfig(
        hf_checkpoint="/model",
        model_type="qwen3-4B",
        actor_num_gpus_per_node=4,
        tensor_model_parallel_size=4,
        deterministic_training=True,
        tp_reduce_precision="float64",
        **kwargs,
    )


@pytest.mark.parametrize("config", [Qwen3, Qwen35])
def test_deterministic_recipes_roundtrip_with_safe_topology(config):
    deployment = DeploymentConfig.create(config())
    restored = DeploymentConfig.model_validate_json(deployment.model_dump_json())
    backend, _, _ = parse_backend_config(restored.trainer_settings)
    assert backend.deterministic_training
    assert backend.data_parallel_size == backend.context_parallel_size == 1
    assert backend.tp_reduce_precision == "float64"
    assert restored.recipe.trainer_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert restored.inference_settings["enable_deterministic_inference"]


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"tensor_model_parallel_size": 2}, "DP=1"),
        ({"context_parallel_size": 2, "tensor_model_parallel_size": 2}, "CP=1"),
        ({"tp_reduce_precision": "float32"}, "float64"),
        ({"tp_reduce_precision": None}, "float64"),
        ({"tp_reduce_precision": "float16"}, "float32, float64"),
        ({"dynamic_batch_size": False}, "dynamic batching"),
        ({"lora_dropout": 0.1}, "zero LoRA dropout"),
        ({"expert_model_parallel_size": 2}, "expert parallelism"),
        ({"cli_options": {"enable_mtp_training": True}}, "without MTP"),
        ({"cli_options": {"calculate_per_token_loss": True}}, "summed losses"),
        ({"deterministic_training": "true"}, "booleans"),
        (
            {"deterministic_training": False, "deterministic_gdn": True},
            "requires deterministic_training",
        ),
    ],
)
def test_rejects_unsupported_deterministic_settings(changes, error):
    with pytest.raises(ValueError, match=error):
        replace(deterministic_config(), **changes).validate()


def test_async_precision_cast_waits_then_copies_exactly_once():
    output, reduced = (
        torch.zeros(2, dtype=torch.bfloat16),
        torch.tensor([1.25, -2.5], dtype=torch.float64),
    )
    calls = []
    work = SimpleNamespace(wait=lambda: calls.append("wait"))
    deferred = _CastAfterWait(work, output, reduced)
    assert output.tolist() == [0, 0]
    deferred.wait()
    assert torch.equal(output, reduced.to(output.dtype))
    reduced.zero_()
    deferred.wait()
    assert output.tolist() == [1.25, -2.5]
    assert calls == ["wait", "wait"]


def test_seeded_initialization_preserves_override_and_restores_on_error(monkeypatch):
    modules = {}
    for name in (
        "megatron",
        "megatron.bridge",
        "megatron.bridge.peft",
        "megatron.bridge.peft.multi_lora_layers",
    ):
        module = modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    calls = []
    layers = modules["megatron.bridge.peft.multi_lora_layers"]
    original = layers.init_adapter_slot = lambda *args, **kwargs: calls.append(
        kwargs["seed"]
    )
    with pytest.raises(RuntimeError):
        with seeded_slot_initialization(1234):
            layers.init_adapter_slot(None, 0, 8, 8)
            layers.init_adapter_slot(None, 1, 8, 8, seed=17)
            raise RuntimeError("initialization failed")
    assert calls == [1234, 17]
    assert layers.init_adapter_slot is original
    with seeded_slot_initialization(None):
        assert layers.init_adapter_slot is original
