"""The registry of recipes a frontend serves, and how requests pick one."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypedDict

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
        self._skipped: set[str] = set()
        self._refresh: asyncio.Task | None = None
        self._refresh_started = 0.0

    @classmethod
    def from_definitions(cls, definitions: Iterable[Routable]) -> DeploymentRoutes:
        routes = cls(None)
        routes._definitions = list(definitions)
        return routes

    async def _load(self) -> list[Routable]:
        """The cached recipes; requests wait on the registry only for the first read."""
        if self.kv is None:
            return self._definitions
        if self._loaded_at is None:
            await self._reload()
        elif time.monotonic() - self._loaded_at >= REGISTRY_TTL_S:
            self._start_reload()
        return self._definitions

    def _start_reload(self) -> asyncio.Task:
        # One read at a time per container; tests run several event loops.
        loop = asyncio.get_running_loop()
        task = self._refresh
        if task is None or task.done() or task.get_loop() is not loop:
            self._refresh_started = time.monotonic()
            task = self._refresh = loop.create_task(self._read_registry())
        return task

    async def _reload(self, *, newer_than: float | None = None) -> None:
        await asyncio.shield(self._start_reload())
        if newer_than is not None and self._refresh_started < newer_than:
            # The read we joined began before the caller saw the new file.
            await asyncio.shield(self._start_reload())

    async def _read_registry(self) -> None:
        try:
            kv = self.kv()
            names = await kv.list_keys("")
            rows = await asyncio.gather(
                *(kv.get(name) for name in names), return_exceptions=True
            )
        except Exception:
            if self._loaded_at is None:
                raise
            logger.exception("Registry read failed; serving the last recipe list")
            self._loaded_at = time.monotonic()
            return
        definitions, skipped = [], set()
        for name, row in zip(names, rows):
            # One bad file must not take down routing for every recipe, and it
            # is never deleted: during a deploy the frontend may be older than
            # the CLI that wrote it.
            try:
                if isinstance(row, BaseException):
                    raise row
                definitions.append(DeploymentConfig.from_registry(row))
            except (ValueError, TypeError) as error:
                if name not in self._skipped:
                    logger.error("Skipping unreadable recipe %r: %s", name, error)
                skipped.add(name)
        self._definitions, self._skipped = definitions, skipped
        self._loaded_at = time.monotonic()

    @staticmethod
    def _exact(definitions: Iterable[Routable], definition_id: str) -> Routable | None:
        return next((d for d in definitions if d.definition_id == definition_id), None)

    async def get(self, definition_id: str) -> Routable | None:
        definitions = await self._load()
        found = self._exact(definitions, definition_id)
        if (
            found is not None
            or self.kv is None
            or "/" in definition_id
            or any(d.model == definition_id for d in definitions)
        ):
            # Recipe names never contain "/", and a cached base model is no recipe.
            return found
        # A recipe registered since the last read: check for its file alone.
        checked = time.monotonic()
        try:
            exists = await self.kv().get(definition_id) is not None
        except (ValueError, TypeError):
            exists = True  # unreadable; the reload will skip and log it
        except Exception:
            logger.exception("Registry lookup of %r failed", definition_id)
            return None
        if exists:
            await self._reload(newer_than=checked)
            found = self._exact(self._definitions, definition_id)
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
