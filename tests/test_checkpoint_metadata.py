import pytest

from spindle.checkpoint_metadata import CheckpointMetadata

MEGATRON_LORA = {
    "schema_version": 1,
    "base_model": "Qwen/Qwen3-4B",
    "engine_definition_id": "qwen3_4b_lora",
    "parameterization": {"type": "lora"},
    "lora_config": {
        "rank": 32,
        "seed": None,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    },
}
MEGATRON_FULL = {
    "schema_version": 1,
    "base_model": "Qwen/Qwen3-4B",
    "engine_definition_id": "qwen3_4b_fft",
    "parameterization": {"type": "full"},
    "lora_config": None,
}
MILES_LORA = {
    **MEGATRON_LORA,
    "lora_config": {**MEGATRON_LORA["lora_config"], "alpha": 32.0, "seed": 7},
    "backend": "miles",
    "checkpoint_format": "miles_torch_dist",
    "miles_revision": "abc123",
    "optimizer_step": 3,
    "has_optimizer": True,
    "topology": {
        "world_size": 2,
        "data_parallel_size": 2,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    },
}


@pytest.mark.parametrize("saved", [MEGATRON_LORA, MEGATRON_FULL, MILES_LORA])
def test_saved_metadata_keeps_its_shape(saved) -> None:
    assert CheckpointMetadata.parse(saved).to_json() == saved


def test_unknown_fields_from_newer_writers_are_ignored() -> None:
    metadata = CheckpointMetadata.parse({**MEGATRON_FULL, "added_later": 1})

    assert metadata.base_model == "Qwen/Qwen3-4B"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 2}, "schema_version: Input should be 1"),
        ({"engine_definition_id": None}, "engine_definition_id"),
        ({"lora_config": None}, "lora checkpoint has no lora_config"),
        (
            {"parameterization": {"type": "full"}},
            "full checkpoint has lora_config",
        ),
    ],
)
def test_invalid_metadata_names_the_problem(changes, message) -> None:
    with pytest.raises(ValueError, match="invalid checkpoint metadata") as error:
        CheckpointMetadata.parse({**MEGATRON_LORA, **changes})

    assert message in str(error.value)
