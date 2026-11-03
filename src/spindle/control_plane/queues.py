"""Per-recipe training queue: models waiting for a trainer slot, and slots in use."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from spindle.providers.contracts import EnginePlatform, KeyValueStore

from .records import PlacementRecord


@dataclass(frozen=True)
class QueueLimits:
    definition_id: str
    slots_per_trainer: int
    max_trainers: int


@dataclass(frozen=True)
class Backlog:
    count: int
    oldest_age_s: float | None


@dataclass(frozen=True)
class Slots:
    used: int
    free: int
    per_trainer: int


@dataclass(frozen=True)
class Trainers:
    running: int
    starting: int
    max: int


@dataclass(frozen=True)
class QueueStats:
    backlog: Backlog
    slots: Slots
    trainers: Trainers


async def describe_queues(
    kv: KeyValueStore,
    engines: EnginePlatform,
    limits: list[QueueLimits],
    now: float,
) -> dict[str, QueueStats]:
    items, listed = await asyncio.gather(
        kv.list_items("placement:", "trainer_demand:"),
        engines.list_instances(),
    )
    placements = [
        PlacementRecord.model_validate(value)
        for key, value in items
        if key.startswith("placement:")
    ]
    placed_ids = {placement.model_id for placement in placements}
    waiting: dict[str, list[float]] = {}
    for key, value in items:
        if key.startswith("trainer_demand:") and value["model_id"] not in placed_ids:
            waiting.setdefault(str(value["definition_id"]), []).append(
                float(value["created_at"])
            )
    running_boots = {
        instance.instance_id: instance.boot_id
        for instance in listed
        if instance.state == "running"
    }
    # A restarted trainer keeps its id but loses the models placed before it.
    used = Counter(
        placement.engine_definition_id
        for placement in placements
        if placement.engine_instance_id in running_boots
        and placement.engine_boot_id
        in ("", running_boots[placement.engine_instance_id])
    )
    by_state = Counter((i.definition_id, i.state) for i in listed)

    result = {}
    for limit in limits:
        name = limit.definition_id
        created = waiting.get(name, [])
        running = by_state[name, "running"]
        result[name] = QueueStats(
            backlog=Backlog(
                count=len(created),
                oldest_age_s=max(0.0, now - min(created)) if created else None,
            ),
            slots=Slots(
                used=used[name],
                free=max(0, running * limit.slots_per_trainer - used[name]),
                per_trainer=limit.slots_per_trainer,
            ),
            trainers=Trainers(
                running=running,
                starting=by_state[name, "starting"],
                max=limit.max_trainers,
            ),
        )
    return result


class SharedSnapshot:
    """Shares one recent read between concurrent callers."""

    def __init__(
        self,
        read: Callable[[float], Awaitable[dict[str, QueueStats]]],
        clock: Callable[[], float],
        max_age_s: float = 3.0,
    ) -> None:
        self._read = read
        self._clock = clock
        self._max_age_s = max_age_s
        self._latest: tuple[float, dict[str, QueueStats]] | None = None
        self._pending: asyncio.Task | None = None

    async def get(self) -> tuple[float, dict[str, QueueStats]]:
        if self._latest and self._clock() - self._latest[0] < self._max_age_s:
            return self._latest
        if self._pending is None or self._pending.done():
            self._pending = asyncio.ensure_future(self._refresh())
        return await asyncio.shield(self._pending)

    async def _refresh(self) -> tuple[float, dict[str, QueueStats]]:
        observed_at = self._clock()
        self._latest = (observed_at, await self._read(observed_at))
        return self._latest
