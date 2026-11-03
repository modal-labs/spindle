import asyncio
from types import SimpleNamespace

from spindle.control_plane import ControlPlane, FutureResolutionStatus
from spindle.control_plane.keys import placement_key, trainer_demand_key
from spindle.control_plane.queues import (
    QueueLimits,
    SharedSnapshot,
    describe_queues,
    prune_queue_events,
)
from spindle.providers.contracts import EngineInstance
from spindle.providers.local import InMemoryKeyValueStore, LocalEnginePlatform
from tests.support import EchoExecutor

LORA = "qwen_lora"


def describe(trainers, placed, waiting, boots=None):
    """placed maps model id to trainer id; waiting maps model id to created_at."""
    boots = boots or {}

    async def run():
        kv = InMemoryKeyValueStore()
        for model_id, created_at in {**dict.fromkeys(placed, 0.0), **waiting}.items():
            await kv.put(
                trainer_demand_key(model_id),
                {"model_id": model_id, "definition_id": LORA, "created_at": created_at},
            )
        for model_id, trainer in placed.items():
            await kv.put(
                placement_key(model_id),
                {
                    "model_id": model_id,
                    "engine_definition_id": LORA,
                    "engine_instance_id": trainer,
                    "placed_at": 0.0,
                    "engine_boot_id": "boot-1",
                },
            )
        instances = tuple(
            EngineInstance(LORA, trainer, state, boot_id=boots.get(trainer, "boot-1"))
            for trainer, state in trainers.items()
        )

        async def list_instances():
            return instances

        engines = SimpleNamespace(list_instances=list_instances)
        limits = [QueueLimits(LORA, slots_per_trainer=2, max_trainers=4)]
        return (await describe_queues(kv, engines, limits, now=100.0))[LORA]

    return asyncio.run(run())


def test_reports_backlog_slots_and_trainers():
    stats = describe(
        {"t1": "running", "t2": "starting", "t3": "stopped"},
        {"m1": "t1"},
        {"m2": 40.0, "m3": 70.0},
    )
    assert (stats.backlog.count, stats.backlog.oldest_age_s) == (2, 60.0)
    assert (stats.slots.used, stats.slots.free, stats.slots.per_trainer) == (1, 1, 2)
    assert (stats.trainers.running, stats.trainers.starting, stats.trainers.max) == (
        1,
        1,
        4,
    )


def test_empty_queue_has_no_backlog_age():
    stats = describe({}, {}, {})
    assert (stats.backlog.count, stats.backlog.oldest_age_s) == (0, None)
    assert stats.slots.free == 0


def test_placements_from_before_a_trainer_restart_do_not_hold_slots():
    stats = describe(
        {"t1": "running"}, {"m1": "t1", "m2": "t1"}, {}, boots={"t1": "boot-2"}
    )
    assert (stats.slots.used, stats.slots.free) == (0, 2)


def test_concurrent_callers_share_one_read():
    async def run():
        reads = 0

        async def read(now):
            nonlocal reads
            reads += 1
            await asyncio.sleep(0)
            return {}

        snapshot = SharedSnapshot(read, clock=lambda: 0.0)
        await asyncio.gather(*(snapshot.get() for _ in range(5)))
        await snapshot.get()
        return reads

    assert asyncio.run(run()) == 1


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def recent(plane, now):
    limits = [QueueLimits(LORA, slots_per_trainer=1, max_trainers=1)]
    return (await describe_queues(plane.kv, plane.engines, limits, now))[LORA].recent


async def placed_model(plane, clock, session_id, seq, created_at, placed_at):
    clock.now = created_at
    creation = await plane.create_model(
        session_id=session_id, model_seq_id=seq, definition_id=LORA, spec={}
    )
    clock.now = placed_at
    resolution = await plane.retrieve(creation.request_id)
    assert resolution.status == FutureResolutionStatus.COMPLETE
    return creation


def test_recent_counts_each_transition_once():
    async def run():
        clock = Clock()
        engines = LocalEnginePlatform(LORA, EchoExecutor)
        plane = ControlPlane(InMemoryKeyValueStore(), engines, clock=clock)
        session = await plane.create_session()
        first = await placed_model(plane, clock, session.session_id, 0, 10.0, 25.0)
        second = await placed_model(plane, clock, session.session_id, 1, 30.0, 40.0)
        third = await placed_model(plane, clock, session.session_id, 2, 41.0, 42.0)
        clock.now = 50.0
        await plane.unload_model(first.model.model_id)
        await plane.unload_model(first.model.model_id)

        (instance,) = await engines.list_instances()
        await engines.mark_dead(instance.instance_id)
        await asyncio.gather(
            *(plane.retrieve(c.request_id) for c in (second, second, third))
        )
        await plane.unload_model(second.model.model_id)

        stats = await recent(plane, now=60.0)
        assert (stats.added, stats.placed, stats.released) == (3, 3, 3)
        assert (stats.wait_s.p50, stats.wait_s.max) == (10.0, 15.0)

    asyncio.run(run())


def test_idle_reclaim_counts_as_a_release():
    async def run():
        clock = Clock()
        engines = LocalEnginePlatform(LORA, EchoExecutor, max_models=1)
        plane = ControlPlane(
            InMemoryKeyValueStore(), engines, session_idle_timeout=60.0, clock=clock
        )
        idle = await plane.create_session()
        await placed_model(plane, clock, idle.session_id, 0, 0.0, 0.0)
        live = await plane.create_session()
        await placed_model(plane, clock, live.session_id, 0, 100.0, 100.0)

        stats = await recent(plane, now=100.0)
        assert (stats.added, stats.placed, stats.released) == (2, 2, 1)

    asyncio.run(run())


def test_events_leave_the_window_and_are_pruned():
    async def run():
        clock = Clock()
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            LocalEnginePlatform(LORA, EchoExecutor),
            clock=clock,
        )
        session = await plane.create_session()
        await placed_model(plane, clock, session.session_id, 0, 0.0, 5.0)

        assert (await recent(plane, now=300.0)).added == 1
        stats = await recent(plane, now=304.0)
        assert (stats.added, stats.placed, stats.wait_s.p50) == (0, 1, 5.0)
        assert await prune_queue_events(plane.kv, before=4.0) == 1
        assert await prune_queue_events(plane.kv, before=6.0) == 1
        assert await plane.kv.list_items("queue_event:") == ()

    asyncio.run(run())


def test_event_write_failures_do_not_fail_requests():
    class FailingEvents(InMemoryKeyValueStore):
        async def put(self, key, value):
            if key.startswith("queue_event:"):
                raise ConnectionError("store unavailable")
            await super().put(key, value)

    async def run():
        clock = Clock()
        plane = ControlPlane(
            FailingEvents(), LocalEnginePlatform(LORA, EchoExecutor), clock=clock
        )
        session = await plane.create_session()
        creation = await placed_model(plane, clock, session.session_id, 0, 0.0, 1.0)
        await plane.unload_model(creation.model.model_id)
        assert (await recent(plane, now=2.0)).added == 0

    asyncio.run(run())
