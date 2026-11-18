"""Opt-in and numerical failure gates require no GPU runtime."""

from pathlib import Path

import pytest
import torch

from spindle.backends.miles_config import MilesBackendConfig
from spindle.backends.miles_runtime import prefix_checks
from spindle.backends.miles_runtime.prefix_checks import check_forward_parity


def _config(**overrides: object) -> MilesBackendConfig:
    values = dict(megatron_to_hf_mode="raw", lora_type="canonical_lora")
    values.update(overrides)
    return MilesBackendConfig(
        hf_checkpoint="/models/test",
        model_type="test",
        actor_num_gpus_per_node=8,
        max_lora_slots=1,
        **values,
    )


def test_gate_defaults_off_and_roundtrips_when_explicit() -> None:
    assert not _config().shared_prefix_training
    assert "--deterministic-mode" not in _config().miles_arguments()
    enabled = _config(shared_prefix_training=True)
    assert "--deterministic-mode" in enabled.miles_arguments()
    assert "--prefix-tree-attention" in enabled.miles_arguments()
    assert MilesBackendConfig.model_validate_json(enabled.model_dump_json()) == enabled


@pytest.mark.parametrize(
    "options",
    [
        {"context_parallel_size": 2},
        {"megatron_to_hf_mode": "bridge"},
        {"lora_type": "lora"},
        {"cli_options": {"lora_kernel_policy": "native"}},
        {"cli_options": {"prefix_tree_attention": False}},
        {"expert_tensor_parallel_size": 2},
        {"lora_dropout": 0.1},
        {"extra_args": ("--hidden-dropout", "0.1")},
        {"cli_options": {"deterministic_mode": False}},
        {"cli_options": {"use_rollout_routing_replay": True}},
        {"cli_options": {"enable_mtp_training": True}},
        {"cli_options": {"pipeline_model_parallel_size": 2}},
        {"cli_options": {"moe_aux_loss_coeff": 0.1}},
        {"cli_options": {"qkv_format": "bshd"}},
    ],
)
def test_gate_rejects_unsupported_semantics(options: dict) -> None:
    with pytest.raises(ValueError, match="shared_prefix_training"):
        _config(shared_prefix_training=True, **options)


def test_gate_does_not_coerce_a_string_to_enabled() -> None:
    with pytest.raises(ValueError):
        _config(shared_prefix_training="true")


def test_zero_forward_error_is_reported_without_losing_float_precision() -> None:
    data = {
        "per_datum": [{"logprobs": torch.tensor([-2.0, -5.0], dtype=torch.float64)}]
    }
    assert check_forward_parity(data, data) == {
        "values": 2,
        "max_abs": 0.0,
        "relative_l2": 0.0,
    }


@pytest.mark.parametrize(
    "actual", [[], [float("nan")], [float("inf")], [None], [-1.1], [-1.0, -1.0]]
)
def test_bad_or_mismatched_forward_values_abort(actual: list[float]) -> None:
    with pytest.raises(ValueError, match="numerical preflight"):
        check_forward_parity({"per_datum": [-1.0]}, {"per_datum": actual})


def test_other_rank_failure_aborts_locally_passing_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prefix_checks.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(prefix_checks.dist, "get_backend", lambda: "gloo")

    def failed_peer(tensor: torch.Tensor, op: object) -> None:
        assert op == prefix_checks.dist.ReduceOp.MIN
        assert tensor.item() == 1
        tensor.zero_()

    monkeypatch.setattr(prefix_checks.dist, "all_reduce", failed_peer)
    with pytest.raises(ValueError, match="trainer rank"):
        check_forward_parity({"per_datum": [-1.0]}, {"per_datum": [-1.0]})


def test_conflicting_deterministic_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    with pytest.raises(ValueError, match="NVTE_ALLOW_NONDETERMINISTIC_ALGO"):
        prefix_checks.configure_prefix_environment()


def test_example_gate_survives_full_deployment_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spindle.deployments import DeploymentConfig, load

    monkeypatch.setenv("SPINDLE_MILES_COMMIT", "a" * 40)
    deployment = DeploymentConfig.create(
        load(Path(__file__).resolve().parents[2] / "examples/shared-prefix/config.py")
    )
    restored = DeploymentConfig.model_validate_json(deployment.model_dump_json())
    backend = MilesBackendConfig(**restored.trainer_settings["miles"])
    assert backend.shared_prefix_training is True
    assert backend.megatron_to_hf_mode == "raw"
    assert backend.lora_type == "canonical_lora"
    assert backend.cli_options["lora_kernel_policy"] == "rowwise"
    assert backend.extra_args == ("--seq-length", "32768")
    assert backend.max_lora_slots == 1
    assert restored.recipe.platform["modal"]["exclude_reserved_capacity"] is True
    assert restored.recipe.trainer_image.endswith(":image")
