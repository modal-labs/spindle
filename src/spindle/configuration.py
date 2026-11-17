"""Flat Python recipes. Modal and the backend validate their own options."""

from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
from typing import Any


def _apply_overrides(values: dict[str, Any], overrides: dict[str, Any]) -> None:
    for path, value in overrides.items():
        parts = path.split(".")
        target = values
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = deepcopy(value)


def _default_platform() -> dict[str, Any]:
    return {
        "frontend": "spindle",
        "modal": {"environment": None, "region": None},
        "secrets": {
            "api": "spindle-api",
            "sampler_proxy": "spindle-proxy",
            "huggingface": "huggingface-secret",
        },
        "storage": {
            "assets": "spindle-model-assets",
            "checkpoints": "spindle-checkpoints",
            "bulletin": "spindle-snapshot-bulletin",
        },
    }


# Recipes subclass this and set plain class attributes and dotted ``overrides``,
# so ``__init__`` stays hand-written instead of dataclass-generated.
@dataclass(init=False)
class BaseConfig:
    platform: dict[str, Any] = field(default_factory=_default_platform)

    name: str = ""
    model: str = ""
    # Hugging Face repo for base weights; defaults to ``model``.
    model_weights: str = ""
    # Commit of the weights repo to download; empty follows its default branch.
    weights_revision: str = ""
    max_context_length: int = 16384
    parameterization: str = "lora"
    backend: str = "miles"

    trainer_gpu: str | list[str] = "H100"
    trainer_gpus_per_node: int = 1
    trainer_nodes: int = 1
    # Request an RDMA allocation even for a single trainer node.
    trainer_rdma: bool = False
    trainer_cpu: int = 8
    trainer_memory_mib: int = 32768
    trainer_disk_mib: int = 0
    trainer_max_instances: int = 1
    trainer_max_clients_per_instance: int = 1
    # Clients whose forward_backward requests may share one trainer call; None packs all.
    trainer_max_clients_per_batch: int | None = None
    trainer_timeout_s: int = 86400
    trainer_env: dict[str, str] = field(default_factory=dict)
    # A built-in trainer image name or module:attribute; empty uses the default.
    trainer_image: str = ""
    # A built-in sampler image name or module:attribute; empty uses the default.
    inference_image: str = ""
    sampler_persistence_concurrency: int = 8

    inference_gpu: str | list[str] = "H100"
    inference_gpus_per_node: int = 1
    inference_cpu: int = 8
    inference_memory_mib: int = 32768
    inference_disk_mib: int = 0
    inference_min_replicas: int = 0
    inference_max_replicas: int = 8
    # None scales out when a replica's max_running_requests slots are full.
    inference_target_concurrency: int | None = None
    # None measures the rate from the pool's own prefills.
    inference_routing_prefill_tokens_per_second: float | None = None
    # None allows a gap of 40% of max_running_requests, at least 4.
    inference_routing_balance_absolute: int | None = None
    inference_routing_balance_relative: float = 1.25
    inference_routing_cache_threshold: float = 0.5
    # None picks cache_wait for LoRA pools with one SGLang data-parallel rank,
    # and affinity otherwise.
    inference_lora_routing: str | None = None
    # None uses Modal's default, which adds replicas as soon as load passes the target.
    inference_scaleup_window_s: int | None = None
    inference_scaledown_window_s: int = 300
    inference_startup_timeout_s: int = 1200
    inference_env: dict[str, str] = field(default_factory=dict)
    # Optional Hugging Face checkpoint used by SGLang speculative decoding.
    # It is downloaded alongside the base model and passed as a local path.
    # NEXTN without a separate draft reuses the target checkpoint.
    inference_draft_model: str = ""
    inference_draft_model_revision: str = ""

    miles_cfg: dict[str, Any] = field(default_factory=dict)
    megatron_cfg: dict[str, Any] = field(default_factory=dict)
    sglang_cfg: dict[str, Any] = field(default_factory=dict)
    session_idle_timeout_s: int = 300
    pool_idle_timeout_s: int = 300
    sweep_interval_s: int = 300

    def __init__(self, **kwargs: Any) -> None:
        values: dict[str, Any] = {
            f.name: f.default_factory()
            for f in fields(BaseConfig)
            if f.default_factory is not MISSING
        }
        # Copy each ancestor's values before applying its overrides.
        for cls in reversed(type(self).__mro__):
            values.update(
                deepcopy(
                    {
                        key: value
                        for key, value in vars(cls).items()
                        if not key.startswith("_")
                        and key != "overrides"
                        and not callable(value)
                    }
                )
            )
            _apply_overrides(values, vars(cls).get("overrides", {}))
        overrides = kwargs.pop("overrides", {})
        values.update(deepcopy(kwargs))
        _apply_overrides(values, overrides)
        self.__dict__.update(values)
