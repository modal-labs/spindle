"""Snapshot of free training slots per deployment."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from spindle.providers.contracts import EnginePlatform, KeyValueStore

from .records import PlacementRecord

Availability = Literal["now", "after_trainer_start", "full"]


@dataclass(frozen=True)
class TrainerLimits:
    definition_id: str
    slots_per_trainer: int
    max_trainers: int


@dataclass(frozen=True)
class TrainerCapacity:
    availability: Availability
    free_slots: int
    waiting_models: int
    running_trainers: int
    max_trainers: int
    slots_per_trainer: int


async def trainer_capacity(
    kv: KeyValueStore,
    engines: EnginePlatform,
    limits: list[TrainerLimits],
) -> dict[str, TrainerCapacity]:
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
    waiting = Counter(
        str(value["definition_id"])
        for key, value in items
        if key.startswith("trainer_demand:") and value["model_id"] not in placed_ids
    )
    instances = [instance for instance in listed if not instance.terminal]
    running_boots = {
        instance.instance_id: instance.boot_id
        for instance in instances
        if instance.state == "running"
    }
    # A restarted trainer keeps its id but loses the models placed before it.
    placed_on_running = Counter(
        placement.engine_definition_id
        for placement in placements
        if placement.engine_instance_id in running_boots
        and placement.engine_boot_id
        in ("", running_boots[placement.engine_instance_id])
    )
    by_state = Counter((i.definition_id, i.state) for i in instances)
    alive = Counter(i.definition_id for i in instances)

    result = {}
    for limit in limits:
        name = limit.definition_id
        running = by_state[name, "running"]
        open_slots = running * limit.slots_per_trainer - placed_on_running[name]
        free_slots = max(0, open_slots - waiting[name])
        # Waiting models fill running trainers first, then ones that can start.
        unserved = max(0, waiting[name] - max(0, open_slots))
        startable = by_state[name, "starting"] + max(
            0, limit.max_trainers - alive[name]
        )
        if free_slots:
            availability = "now"
        elif startable * limit.slots_per_trainer > unserved:
            availability = "after_trainer_start"
        else:
            availability = "full"
        result[name] = TrainerCapacity(
            availability=availability,
            free_slots=free_slots,
            waiting_models=waiting[name],
            running_trainers=running,
            max_trainers=limit.max_trainers,
            slots_per_trainer=limit.slots_per_trainer,
        )
    return result


class CapacitySnapshots:
    """Shares one recent snapshot between concurrent callers."""

    def __init__(
        self,
        read: Callable[[], Awaitable[dict[str, TrainerCapacity]]],
        clock: Callable[[], float],
        max_age_s: float = 3.0,
    ) -> None:
        self._read = read
        self._clock = clock
        self._max_age_s = max_age_s
        self._latest: tuple[float, dict[str, TrainerCapacity]] | None = None
        self._pending: asyncio.Task | None = None

    async def get(self) -> tuple[float, dict[str, TrainerCapacity]]:
        if self._latest and self._clock() - self._latest[0] < self._max_age_s:
            return self._latest
        if self._pending is None or self._pending.done():
            self._pending = asyncio.ensure_future(self._refresh())
        return await asyncio.shield(self._pending)

    async def _refresh(self) -> tuple[float, dict[str, TrainerCapacity]]:
        observed_at = self._clock()
        self._latest = (observed_at, await self._read())
        return self._latest
