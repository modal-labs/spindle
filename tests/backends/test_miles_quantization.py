import pytest
import torch

from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_runtime.quantization import pad_expert_slot_ids
from spindle.backends.deployment import resolve_backend_settings
from spindle.configs.qwen35_35b_a3b_mxfp8_lora_8k import config as mxfp8_config


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4", "bf16", ""])
def test_invalid_base_quantization_fails_before_startup(quantization):
    config = MilesBackendConfig(
        hf_checkpoint="/model",
        model_type="qwen3.5-35B-A3B_lora",
        actor_num_gpus_per_node=4,
        base_quantization=quantization,
    )
    with pytest.raises(ValueError, match="base_quantization must be mxfp8 or None"):
        config.validate()


@pytest.mark.parametrize("counts", [[1, 0, 3, 2], [0, 0, 0, 0]])
def test_expert_padding_preserves_adapter_outputs_and_gradients(counts):
    torch.manual_seed(42)
    counts = torch.tensor(counts)
    slots = torch.arange(int(counts.sum()), dtype=torch.int32) % 2
    padded_slots, padded_counts = pad_expert_slot_ids(slots, counts, 32)
    torch.testing.assert_close(padded_counts, ((counts + 31) // 32) * 32)
    x = torch.randn(int(counts.sum()), 4, requires_grad=True)
    weights = torch.randn(2, len(counts), 4, 3, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    padded_x = torch.cat(
        [
            torch.nn.functional.pad(chunk, (0, 0, 0, int(padded - count)))
            for chunk, count, padded in zip(
                x.split(counts.tolist()), counts, padded_counts, strict=True
            )
        ]
    )
    expert_ids = torch.repeat_interleave(torch.arange(len(counts)), padded_counts)
    actual = (padded_x.unsqueeze(1) @ weights[padded_slots.long(), expert_ids]).squeeze(
        1
    )
    actual = torch.cat(
        [
            chunk[:count]
            for chunk, count in zip(
                actual.split(padded_counts.tolist()), counts, strict=True
            )
        ]
    )
    original_experts = torch.repeat_interleave(torch.arange(len(counts)), counts)
    expected = (
        reference_x.unsqueeze(1) @ reference_weights[slots.long(), original_experts]
    ).squeeze(1)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(weights.grad, reference_weights.grad)


def test_mxfp8_recipe_uses_lora_and_keeps_adapter_publications_separate():
    trainer, sampler = resolve_backend_settings(mxfp8_config, "/assets/model")
    assert mxfp8_config.parameterization == "lora"
    assert mxfp8_config.backend == "miles"
    assert "megatron" not in trainer
    assert trainer["miles"]["base_quantization"] == "mxfp8"
    assert trainer["miles"]["max_lora_rank"] == 16
    assert sampler["enable_lora"] is True
    assert sampler["quantization"] == "mxfp8"
    assert "weight_update_staging" not in sampler
