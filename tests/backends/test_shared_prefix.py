"""Opt-in and numerical failure gates require no GPU runtime."""

import ast
import json
import sys
import time
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace
from typing import cast

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
    assert not _config().shared_prefix_check_forward
    assert "--deterministic-mode" not in _config().miles_arguments()
    enabled = _config(shared_prefix_training=True, shared_prefix_check_forward=True)
    assert "--deterministic-mode" in enabled.miles_arguments()
    assert "--prefix-tree-attention" in enabled.miles_arguments()
    assert "--lora-kernel-policy" not in enabled.miles_arguments()
    assert MilesBackendConfig.model_validate_json(enabled.model_dump_json()) == enabled


def test_forward_checks_require_the_feature_and_a_boolean() -> None:
    with pytest.raises(ValueError, match="requires shared_prefix_training"):
        _config(shared_prefix_check_forward=True)
    with pytest.raises(ValueError):
        _config(shared_prefix_training=True, shared_prefix_check_forward="true")


def _actor_dispatch_classes() -> tuple[type, type]:
    # Execute the production dispatch method against native-call fakes without
    # importing the actor's CUDA-only Miles/Megatron dependencies on CPU CI.
    source = (
        Path(__file__).resolve().parents[2]
        / "src/spindle/backends/miles_runtime/prefix_actor.py"
    )
    tree = ast.parse(source.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    shared = classes["SharedPrefixTrainRayActor"]
    validating = classes["ValidatingSharedPrefixTrainRayActor"]
    shared.body = [
        node
        for node in shared.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name == "forward_backward"
    ]

    class NativeCalls:
        def __init__(self, failure: str | None = None) -> None:
            self.prefix_enabled = True
            self.calls: list[tuple[str, bool]] = []
            self.prefix_rows: list[dict[str, int]] = []
            self.failure = failure

        def forward_only(
            self, *args: object, **kwargs: object
        ) -> dict[str, list[float]]:
            self.calls.append(("forward_only", self.prefix_enabled))
            if self.failure == "native" and not self.prefix_enabled:
                raise ValueError("native failed")
            value = -2.0 if self.failure == "parity" and self.prefix_enabled else -1.0
            return {"per_datum": [value]}

        def forward_backward(self, *args: object, **kwargs: object) -> dict[str, dict]:
            self.calls.append(("forward_backward", self.prefix_enabled))
            self.prefix_rows.append({"original_rows": 8, "compact_rows": 5})
            return {"metrics": {}}

    namespace = {
        "SpindleMilesTrainRayActor": NativeCalls,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None),
            distributed=SimpleNamespace(get_rank=lambda: 0),
        ),
        "time": time,
        "sys": sys,
        "json": json,
        "check_forward_parity": check_forward_parity,
    }
    code = ast.fix_missing_locations(
        ast.Module(body=[shared, validating], type_ignores=[])
    )
    exec(compile(code, str(source), "exec"), namespace)
    return (
        cast(type, namespace["SharedPrefixTrainRayActor"]),
        cast(type, namespace["ValidatingSharedPrefixTrainRayActor"]),
    )


@pytest.mark.parametrize("checked", [False, True])
def test_shipping_dispatch_skips_extra_forwards_and_debug_checks(checked: bool) -> None:
    classes = _actor_dispatch_classes()
    actor = classes[int(checked)]()
    result = actor.forward_backward("batch")
    expected = [("forward_only", False), ("forward_only", True)] if checked else []
    assert actor.calls == [*expected, ("forward_backward", True)]
    assert result["metrics"]["prefix_saved_fraction"] == 0.375


@pytest.mark.parametrize("failure", ["native", "parity"])
def test_debug_failure_restores_prefix_mode_and_stops_before_backward(
    failure: str,
) -> None:
    _, validating = _actor_dispatch_classes()
    actor = validating(failure)
    with pytest.raises(ValueError):
        actor.forward_backward("batch")
    assert actor.prefix_enabled is True
    assert all(name == "forward_only" for name, _ in actor.calls)


@pytest.mark.parametrize(
    "options",
    [
        {"context_parallel_size": 2},
        {"megatron_to_hf_mode": "bridge"},
        {"lora_type": "lora"},
        {"cli_options": {"prefix_tree_attention": False}},
        {"expert_tensor_parallel_size": 2},
        {"lora_dropout": 0.1},
        {"extra_args": ("--hidden-dropout", "0.1")},
        {"cli_options": {"deterministic_mode": False}},
        {"cli_options": {"use_rollout_routing_replay": True}},
        {"cli_options": {"enable_mtp_training": True}},
        {"cli_options": {"pipeline_model_parallel_size": 2}},
        {"cli_options": {"moe_aux_loss_coeff": 0.1}},
        {"cli_options": {"moe_aux_loss_coeff": [0.1]}},
        {"cli_options": {"moe_aux_loss_coeff": 0.0}},
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


def test_failed_forward_gate_retains_error_magnitudes() -> None:
    with pytest.raises(ValueError) as error:
        check_forward_parity({"per_datum": [-2.0]}, {"per_datum": [-2.25]})
    assert "local_passed=False" in str(error.value)
    assert "native_values=1, shared_values=1" in str(error.value)
    assert "max_abs=0.25, relative_l2=0.125" in str(error.value)


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
    with pytest.raises(
        ValueError, match=r"local_passed=True.*max_abs=0, relative_l2=0"
    ):
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
    assert backend.shared_prefix_check_forward is False
    assert backend.megatron_to_hf_mode == "raw"
    assert backend.lora_type == "canonical_lora"
    assert backend.cli_options["lora_kernel_policy"] == "native"
    assert backend.cli_options["moe_aux_loss_coeff"] == [0.0]
    assert backend.extra_args == ("--seq-length", "65552")
    assert backend.tensor_model_parallel_size == 4
    assert backend.max_lora_slots == 1
    assert any(
        fnmatchcase("model.language_model.layers.3.mlp.experts.gate_up_proj", target)
        for target in backend.target_modules
    )
    assert not any(
        fnmatchcase("model.visual.blocks.0.mlp.gate_proj", target)
        for target in backend.target_modules
    )
    assert restored.recipe.platform["modal"]["exclude_reserved_capacity"] is True
    assert restored.recipe.trainer_image.endswith(":image")


@pytest.mark.parametrize("policy", ["native", "rowwise"])
def test_shared_prefix_preserves_explicit_native_kernel_policy(policy: str) -> None:
    config = _config(
        shared_prefix_training=True, cli_options={"lora_kernel_policy": policy}
    )
    assert config.cli_options["lora_kernel_policy"] == policy
    assert "--lora-kernel-policy" not in config.miles_arguments()
