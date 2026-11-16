from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str
    experiment: str | None = None
    suite: Literal[
        "inference", "eval", "trainer", "weight-sync", "async-rl", "control-plane"
    ]
    model: str
    parameterization: Literal["lora", "full"] = "lora"
    deployment: str | None = None
    clients: list[int] = Field(default_factory=lambda: [1, 2, 4], min_length=1)
    context_lengths: list[int] = Field(
        default_factory=lambda: [1024, 8192], min_length=1
    )
    steps: int = Field(default=10, ge=1)
    warmup_steps: int = Field(default=2, ge=0)
    output_tokens: int = Field(default=128, ge=1)
    batch_size: int = Field(default=1, ge=1)
    rank: int = Field(default=8, ge=1)
    seed: int = 42
    learning_rate: float = Field(default=1e-5, gt=0)
    timeout_s: float = Field(default=7200, gt=0)
    recipe: Literal["dapo", "swe-gym", "command"] = "dapo"
    requests_per_client: int = Field(default=100, ge=1, le=100000)
    command: list[str] = Field(default_factory=list)
    recipe_config: dict[str, Any] = Field(default_factory=dict)
    deployment_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_workload(self):
        if any(n < 1 for n in self.clients) or any(n < 2 for n in self.context_lengths):
            raise ValueError("clients must be positive; context lengths must be >= 2")
        if len(set(self.clients)) != len(self.clients) or len(
            set(self.context_lengths)
        ) != len(self.context_lengths):
            raise ValueError("duplicate sweep values")
        if self.suite == "async-rl" and self.recipe == "command" and not self.command:
            raise ValueError("the command recipe requires an argv list")
        if self.suite == "async-rl" and (
            len(self.clients) != 1 or len(self.context_lengths) != 1
        ):
            raise ValueError(
                "Each RL recipe config must select one client count and context length"
            )
        return self


def numeric_leaves(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from numeric_leaves(item, f"{prefix}.{index}")
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        yield prefix, value
