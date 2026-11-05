"""Per-recipe training queue: waiting models, slots in use, and recent changes."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from spindle.providers.contracts import EnginePlatform, KeyValueStore

from .keys import queue_event_key
from .records import PlacementRecord

RECENT_WINDOW_S = 3600.0
EVENT_WRITE_TIMEOUT_S = 5.0
QueueEvent = Literal["added", "placed", "released"]


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
class Wait:
    p50: float | None
    p95: float | None
    max: float | None


@dataclass(frozen=True)
class Recent:
    window_s: float
    added: int
    placed: int
    released: int
    wait_s: Wait


@dataclass(frozen=True)
class QueueStats:
    backlog: Backlog
    slots: Slots
    trainers: Trainers
    recent: Recent


async def record_queue_event(
    kv: KeyValueStore,
    kind: QueueEvent,
    model_id: str,
    definition_id: str,
    at: float,
    wait_s: float | None = None,
) -> None:
    # One key per kind and model, so retries and racing writers count once.
    event = {"kind": kind, "definition_id": definition_id, "at": at}
    if wait_s is not None:
        event["wait_s"] = wait_s
    try:
        async with asyncio.timeout(EVENT_WRITE_TIMEOUT_S):
            await kv.put(queue_event_key(kind, model_id), event)
    except Exception:
        logging.getLogger(__name__).exception("record %s event for %s", kind, model_id)


async def prune_queue_events(kv: KeyValueStore, before: float) -> int:
    stale = [
        key
        for key, event in await kv.list_items("queue_event:")
        if event["at"] < before
    ]
    await asyncio.gather(*(kv.delete(key) for key in stale))
    return len(stale)


def _recent(events: list[dict]) -> Recent:
    kinds = Counter(event["kind"] for event in events)
    waits = [event["wait_s"] for event in events if event["kind"] == "placed"]
    return Recent(
        window_s=RECENT_WINDOW_S,
        added=kinds["added"],
        placed=kinds["placed"],
        released=kinds["released"],
        wait_s=_wait(waits),
    )


def _wait(waits: list[float]) -> Wait:
    if not waits:
        return Wait(p50=None, p95=None, max=None)
    ordered = sorted(waits)

    def nearest_rank(q: float) -> float:
        return ordered[math.ceil(q * len(ordered)) - 1]

    return Wait(p50=nearest_rank(0.5), p95=nearest_rank(0.95), max=ordered[-1])


@dataclass(frozen=True)
class Waiter:
    model_id: str
    created_at: float
    polled_at: float | None


@dataclass(frozen=True)
class QueueView:
    waiting: dict[str, list[Waiter]]
    used: Counter[str]
    by_state: Counter[tuple[str, str]]

    def free_slots(self, definition_id: str, slots_per_trainer: int) -> int:
        running = self.by_state[definition_id, "running"]
        return max(0, running * slots_per_trainer - self.used[definition_id])


async def read_queues(kv: KeyValueStore, engines: EnginePlatform) -> QueueView:
    """Waiters per definition, oldest first, and slot use on running trainers."""
    items, listed = await asyncio.gather(
        kv.list_items("placement:", "trainer_demand:", "trainer_slot_wait:"),
        engines.list_instances(),
    )
    polled = {
        value["model_id"]: float(value["polled_at"])
        for key, value in items
        if key.startswith("trainer_slot_wait:")
    }
    placements = [
        PlacementRecord.model_validate(value)
        for key, value in items
        if key.startswith("placement:")
    ]
    placed_ids = {placement.model_id for placement in placements}
    waiting: dict[str, list[Waiter]] = {}
    for key, value in items:
        if key.startswith("trainer_demand:") and value["model_id"] not in placed_ids:
            model_id = value["model_id"]
            waiting.setdefault(str(value["definition_id"]), []).append(
                Waiter(model_id, float(value["created_at"]), polled.get(model_id))
            )
    for line in waiting.values():
        line.sort(key=lambda waiter: (waiter.created_at, waiter.model_id))
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
    return QueueView(waiting, used, by_state)


async def describe_queues(
    kv: KeyValueStore,
    engines: EnginePlatform,
    limits: list[QueueLimits],
    now: float,
) -> dict[str, QueueStats]:
    view, event_items = await asyncio.gather(
        read_queues(kv, engines), kv.list_items("queue_event:")
    )
    events: dict[str, list[dict]] = defaultdict(list)
    for _, event in event_items:
        if event["at"] >= now - RECENT_WINDOW_S:
            events[event["definition_id"]].append(event)
    result = {}
    for limit in limits:
        name = limit.definition_id
        created = [waiter.created_at for waiter in view.waiting.get(name, [])]
        result[name] = QueueStats(
            backlog=Backlog(
                count=len(created),
                oldest_age_s=max(0.0, now - min(created)) if created else None,
            ),
            slots=Slots(
                used=view.used[name],
                free=view.free_slots(name, limit.slots_per_trainer),
                per_trainer=limit.slots_per_trainer,
            ),
            trainers=Trainers(
                running=view.by_state[name, "running"],
                starting=view.by_state[name, "starting"],
                max=limit.max_trainers,
            ),
            recent=_recent(events[name]),
        )
    return result


class SharedSnapshot[T]:
    """Shares one recent read between concurrent callers."""

    def __init__(
        self,
        read: Callable[[float], Awaitable[T]],
        clock: Callable[[], float],
        max_age_s: float = 3.0,
    ) -> None:
        self._read = read
        self._clock = clock
        self._max_age_s = max_age_s
        self.latest: tuple[float, T] | None = None
        self._pending: asyncio.Task | None = None

    async def get(self) -> tuple[float, T]:
        if self.latest and self._clock() - self.latest[0] < self._max_age_s:
            return self.latest
        if self._pending is None or self._pending.done():
            self._pending = asyncio.ensure_future(self._refresh())
        return await asyncio.shield(self._pending)

    async def _refresh(self) -> tuple[float, T]:
        observed_at = self._clock()
        self.latest = (observed_at, await self._read(observed_at))
        return self.latest
