import asyncio
from types import SimpleNamespace

import pytest

from spindle.control_plane.capacity import (
    CapacitySnapshots,
    TrainerLimits,
    trainer_capacity,
)
from spindle.control_plane.keys import placement_key, trainer_demand_key
from spindle.providers.contracts import EngineInstance
from spindle.providers.local import InMemoryKeyValueStore

LORA = "qwen_lora"


def snapshot(trainers, placed, waiting, max_trainers=2, boots=None):
    """placed maps model id to trainer id; waiting lists unplaced model ids."""
    boots = boots or {}

    async def run():
        kv = InMemoryKeyValueStore()
        for model_id in [*placed, *waiting]:
            await kv.put(
                trainer_demand_key(model_id),
                {"model_id": model_id, "definition_id": LORA, "created_at": 0.0},
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
        limits = [TrainerLimits(LORA, slots_per_trainer=2, max_trainers=max_trainers)]
        return (await trainer_capacity(kv, engines, limits))[LORA]

    return asyncio.run(run())


def test_free_slot_on_running_trainer_is_available_now():
    capacity = snapshot({"t1": "running"}, {"m1": "t1"}, [])
    assert capacity.availability == "now"
    assert capacity.free_slots == 1
    assert capacity.running_trainers == 1


@pytest.mark.parametrize(
    ("trainers", "placed", "waiting"),
    [
        ({}, {}, []),
        ({"t1": "running"}, {"m1": "t1"}, ["m2"]),
        ({"t1": "starting", "t2": "draining"}, {"m1": "t2"}, []),
        ({"t1": "running", "t2": "stopped"}, {"m1": "t1", "m2": "t1"}, []),
    ],
)
def test_no_free_slot_but_a_trainer_can_start(trainers, placed, waiting):
    capacity = snapshot(trainers, placed, waiting)
    assert capacity.free_slots == 0
    assert capacity.availability == "after_trainer_start"


def test_full_when_every_trainer_is_used_and_cap_is_reached():
    capacity = snapshot(
        {"t1": "running", "t2": "running"},
        {"m1": "t1", "m2": "t1", "m3": "t2", "m4": "t2"},
        ["m5"],
    )
    assert capacity.availability == "full"
    assert capacity.waiting_models == 1


def test_full_when_waiting_models_fill_the_starting_trainer():
    capacity = snapshot({"t1": "starting"}, {}, ["m1", "m2"], max_trainers=1)
    assert capacity.availability == "full"


def test_placements_from_before_a_trainer_restart_do_not_hold_slots():
    capacity = snapshot(
        {"t1": "running"}, {"m1": "t1", "m2": "t1"}, [], boots={"t1": "boot-2"}
    )
    assert capacity.free_slots == 2


def test_concurrent_callers_share_one_read():
    async def run():
        reads = 0

        async def read():
            nonlocal reads
            reads += 1
            await asyncio.sleep(0)
            return {}

        snapshots = CapacitySnapshots(read, clock=lambda: 0.0)
        await asyncio.gather(*(snapshots.get() for _ in range(5)))
        await snapshots.get()
        return reads

    assert asyncio.run(run()) == 1
