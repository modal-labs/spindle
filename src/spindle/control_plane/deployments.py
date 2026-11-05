"""The registry of recipes a frontend serves, and how requests pick one."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypedDict

from pydantic import ValidationError

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig
from spindle.providers.contracts import KeyValueStore

logger = logging.getLogger(__name__)

REGISTRY_TTL_S = 5.0
FRONTEND_FIELDS = ("session_idle_timeout_s", "pool_idle_timeout_s", "sweep_interval_s")


class Platform(TypedDict):
    frontend: str
    modal: dict[str, str | None]
    secrets: dict[str, str]
    storage: dict[str, str]


@dataclass(frozen=True)
class FrontendSettings:
    platform: Platform
    session_idle_timeout_s: int
    pool_idle_timeout_s: int
    sweep_interval_s: int

    @classmethod
    def from_recipe(cls, recipe: BaseConfig) -> FrontendSettings:
        return cls(
            platform=recipe.platform,
            **{field: getattr(recipe, field) for field in FRONTEND_FIELDS},
        )


class Routable(Protocol):
    definition_id: str
    name: str
    model: str
    weights_repo: str
    parameterization: str
    max_context_length: int


class DeploymentRoutes:
    def __init__(self, kv: Callable[[], KeyValueStore] | None):
        # A factory, since a Modal frontend learns its app ID only at runtime.
        self.kv = kv
        self._definitions: list[Routable] = []
        self._loaded_at: float | None = None

    @classmethod
    def from_definitions(cls, definitions: Iterable[Routable]) -> DeploymentRoutes:
        routes = cls(None)
        routes._definitions = list(definitions)
        return routes

    async def _load(self, *, refresh: bool = False) -> list[Routable]:
        if self.kv is not None and (
            refresh
            or self._loaded_at is None
            or time.monotonic() - self._loaded_at >= REGISTRY_TTL_S
        ):
            definitions = []
            for name, row in await self.kv().list_items(""):
                # One bad file must not take down routing for every recipe:
                # drop it from the registry so `spindle deploy` can write it again.
                try:
                    definitions.append(DeploymentConfig.model_validate(row))
                except (ValidationError, ValueError, TypeError) as error:
                    logger.error("Removing invalid recipe %r: %s", name, error)
                    await self.kv().delete(name)
            self._definitions = definitions
            self._loaded_at = time.monotonic()
        return self._definitions

    @staticmethod
    def _exact(definitions: Iterable[Routable], definition_id: str) -> Routable | None:
        return next((d for d in definitions if d.definition_id == definition_id), None)

    async def get(self, definition_id: str) -> Routable | None:
        found = self._exact(await self._load(), definition_id)
        if (
            found is None
            and self.kv is not None
            and await self.kv().get(definition_id) is not None
        ):
            found = self._exact(await self._load(refresh=True), definition_id)
        return found

    async def select(self, model: str, mode: str | None = None) -> Routable | None:
        """The exact recipe named `model`, else the longest-context one for it."""
        exact = await self.get(model)
        if exact is not None:
            return exact if mode is None or exact.parameterization == mode else None
        matches = [
            d
            for d in await self._load()
            if d.model == model and (mode is None or d.parameterization == mode)
        ]
        return max(matches, key=lambda d: d.max_context_length, default=None)

    async def definitions(self) -> list[Routable]:
        return list(await self._load())

    async def capabilities(self) -> list[dict[str, Any]]:
        definitions = await self.definitions()
        rows = []
        for model in dict.fromkeys(d.model for d in definitions):
            recipes = [d for d in definitions if d.model == model]
            context = min(
                max(
                    candidate.max_context_length
                    for candidate in recipes
                    if candidate.parameterization == d.parameterization
                )
                for d in recipes
            )
            rows.append({"model_name": model, "max_context_length": context})
            if len(recipes) > 1:
                rows += [
                    {
                        "model_name": d.definition_id,
                        "max_context_length": d.max_context_length,
                    }
                    for d in recipes
                ]
        return rows
