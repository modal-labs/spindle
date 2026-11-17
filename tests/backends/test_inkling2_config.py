from __future__ import annotations

import pytest

from spindle.backends.deployment import backend_config, resolve_backend_settings
from spindle.backends.miles_config import MilesBackendConfig, lora_target_flags
from spindle.deployments import config_path, load


def test_inkling2_recipe_preserves_target_groups_and_serving_head_alias() -> None:
    recipe = load(config_path("inkling2-small-lora-8k-b300"))
    trainer, sampler = resolve_backend_settings(recipe, "/assets/inkling2")
    config = MilesBackendConfig(**trainer["miles"])
    assert lora_target_flags(config.target_modules) == (True, False, True)
    assert config.world_size == config.expert_model_parallel_size == 8
    assert config.max_lora_slots == recipe.trainer_max_clients_per_instance == 2
    assert sampler["lora_target_modules"] == [
        "model.llm.layers.*.attn.wq_du",
        "model.llm.layers.*.attn.wr_du",
        "model.llm.layers.*.attn.gate_attn",
        "model.llm.layers.*.attn.wk_dv",
        "model.llm.layers.*.attn.wv_dv",
        "model.llm.layers.*.attn.wo_ud",
        "lm_head",
    ]
    assert sampler["max_loras_per_batch"] == 1


@pytest.mark.parametrize(
    "parallel",
    [
        "tensor_model_parallel_size",
        "context_parallel_size",
        "expert_tensor_parallel_size",
    ],
)
def test_inkling2_rejects_unsupported_parallel_layout(parallel: str) -> None:
    recipe = load(config_path("inkling2-small-lora-8k-b300"))
    recipe.miles_cfg[parallel] = 2
    if parallel == "expert_tensor_parallel_size":
        recipe.miles_cfg["expert_model_parallel_size"] = 4
    with pytest.raises(ValueError, match="TP=CP=expert-TP=1"):
        backend_config(recipe)
