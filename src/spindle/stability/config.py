"""Stability info request types, recipe policy, and capture plan resolution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StabilityFieldPolicy = Literal["off", "allowed", "default"]
StabilityStoreKind = Literal["dict", "volume"]
StabilityField = Literal[
    "routed_experts",
    "sampling_mask",
    "sampling_probabilities",
]

STABILITY_FIELDS: tuple[StabilityField, ...] = (
    "routed_experts",
    "sampling_mask",
    "sampling_probabilities",
)
DEFAULT_CAPTURE_TTL_SECONDS = 86_400


class StabilityInfo(BaseModel):
    """Per-field client request for server-side stability info capture.

    None follows the deployment recipe default; True requires capture; False skips it.
    """

    model_config = ConfigDict(extra="forbid")

    routed_experts: bool | None = None
    sampling_mask: bool | None = None
    sampling_probabilities: bool | None = None


class RecipeStabilityConfig(BaseModel):
    """Deployment recipe settings controlling which stability fields may be captured."""

    model_config = ConfigDict(extra="forbid")

    routed_experts: StabilityFieldPolicy = "off"
    sampling_mask: StabilityFieldPolicy = "off"
    sampling_probabilities: StabilityFieldPolicy = "off"
    sampling_mask_top_k: int | None = Field(default=None, ge=1)
    store: StabilityStoreKind = "dict"
    ttl_seconds: int = Field(default=DEFAULT_CAPTURE_TTL_SECONDS, ge=1)

    @model_validator(mode="after")
    def _validate_policy_consistency(self) -> RecipeStabilityConfig:
        if self.sampling_probabilities == "default" and self.sampling_mask == "off":
            raise ValueError(
                "stability_info.sampling_probabilities='default' requires "
                "sampling_mask to be 'allowed' or 'default'"
            )
        return self


class CapturePlan(BaseModel):
    """Resolved server-side capture flags for one sampling request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    routed_experts: bool = False
    sampling_mask: bool = False
    sampling_probabilities: bool = False
    sampler_draws: bool = False
    sampling_mask_top_k: int | None = None

    @property
    def any_enabled(self) -> bool:
        return (
            self.routed_experts
            or self.sampling_mask
            or self.sampling_probabilities
            or self.sampler_draws
        )


StabilityRequest = bool | Mapping[str, Any] | StabilityInfo | None


def parse_stability_request(value: StabilityRequest) -> StabilityInfo | None:
    """Normalize duck-typed record_stability_info to StabilityInfo, or None if off."""
    if value is None or value is False:
        return None
    if value is True:
        return StabilityInfo()
    if isinstance(value, StabilityInfo):
        return value.model_copy()
    if isinstance(value, Mapping):
        return StabilityInfo.model_validate(dict(value))
    raise ValueError(
        "record_stability_info must be a boolean, mapping, or StabilityInfo"
    )


def serialize_stability_request(
    value: StabilityRequest,
) -> bool | dict[str, bool | None]:
    """Wire representation for /api/v1/asample."""
    parsed = parse_stability_request(value)
    if parsed is None:
        return False
    dumped = parsed.model_dump(exclude_none=True)
    return dumped or True


def resolve_capture_plan(
    request: StabilityRequest,
    recipe_config: Mapping[str, Any] | RecipeStabilityConfig | None = None,
) -> CapturePlan | None:
    """Resolve a client's record_stability_info against the deployment's policy."""
    parsed = parse_stability_request(request)
    if parsed is None:
        return None
    policy = (
        recipe_config
        if isinstance(recipe_config, RecipeStabilityConfig)
        else RecipeStabilityConfig.model_validate(dict(recipe_config or {}))
    )
    resolved: dict[str, bool] = {}
    for field in STABILITY_FIELDS:
        client_choice = getattr(parsed, field)
        recipe_choice = getattr(policy, field)
        if recipe_choice == "off":
            if client_choice is True:
                raise ValueError(
                    f"stability info capture for {field!r} is disabled by this deployment"
                )
            resolved[field] = False
        elif recipe_choice == "allowed":
            resolved[field] = client_choice is True
        else:
            resolved[field] = client_choice is not False

    if resolved["sampling_probabilities"]:
        if parsed.sampling_mask is False:
            raise ValueError("sampling_probabilities requires sampling_mask")
        if policy.sampling_mask == "off":
            raise ValueError(
                "stability info capture for 'sampling_mask' is disabled by this deployment"
            )
        resolved["sampling_mask"] = True

    return CapturePlan(
        routed_experts=resolved["routed_experts"],
        sampling_mask=resolved["sampling_mask"],
        sampling_probabilities=resolved["sampling_probabilities"],
        sampler_draws=True,
        sampling_mask_top_k=(
            policy.sampling_mask_top_k if resolved["sampling_mask"] else None
        ),
    )
