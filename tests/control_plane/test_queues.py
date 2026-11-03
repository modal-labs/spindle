import asyncio
from types import SimpleNamespace

from spindle.control_plane.keys import placement_key, trainer_demand_key
from spindle.control_plane.queues import QueueLimits, SharedSnapshot, describe_queues
from spindle.providers.contracts import EngineInstance
from spindle.providers.local import InMemoryKeyValueStore

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
