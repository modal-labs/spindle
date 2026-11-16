"""Schema of the metadata.json file saved next to every training checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Parameterization(_Frozen):
    type: Literal["lora", "full"]


class CheckpointLoraConfig(_Frozen):
    rank: int
    alpha: float | None = None
    seed: int | None = None
    train_attn: bool
    train_mlp: bool
    train_unembed: bool


class MilesTopology(_Frozen):
    world_size: int
    data_parallel_size: int = 1
    tensor_model_parallel_size: int
    context_parallel_size: int = 1
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int
    pipeline_model_parallel_size: int


class CheckpointMetadata(_Frozen):
    """Fields every backend writes, then fields only the Miles backend writes."""

    schema_version: Literal[1]
    base_model: NonEmptyString
    engine_definition_id: NonEmptyString
    parameterization: Parameterization
    lora_config: CheckpointLoraConfig | None = None

    backend: str | None = None
    checkpoint_format: str | None = None
    miles_revision: str | None = None
    optimizer_step: int | None = None
    has_optimizer: bool | None = None
    topology: MilesTopology | None = None

    @model_validator(mode="after")
    def _lora_config_matches_parameterization(self) -> Self:
        if self.parameterization.type == "lora" and self.lora_config is None:
            raise ValueError("lora checkpoint has no lora_config")
        if self.parameterization.type == "full" and self.lora_config is not None:
            raise ValueError("full checkpoint has lora_config")
        return self

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> CheckpointMetadata:
        try:
            return cls.model_validate(value)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, error['loc'])) or 'metadata'}: {error['msg']}"
                for error in exc.errors()
            )
            raise ValueError(f"invalid checkpoint metadata: {problems}") from exc

    def to_json(self) -> dict[str, Any]:
        """Dump only the fields the writer set, so each backend's file keeps its shape."""
        return self.model_dump(mode="json", exclude_unset=True)
