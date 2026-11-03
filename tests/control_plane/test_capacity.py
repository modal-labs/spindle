import asyncio
from types import SimpleNamespace

from spindle.control_plane.capacity import TrainerLimits, trainer_capacity
from spindle.control_plane.keys import placement_key, trainer_demand_key
from spindle.providers.contracts import EngineInstance
from spindle.providers.local import InMemoryKeyValueStore

LORA = "qwen_lora"
LIMITS = [TrainerLimits(LORA, slots_per_trainer=2, max_trainers=2)]


def snapshot(trainers, placed, waiting):
    """placed maps model id to trainer id; waiting lists unplaced model ids."""

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
                },
            )
        instances = tuple(
            EngineInstance(LORA, trainer, state) for trainer, state in trainers.items()
        )

        async def list_instances():
            return instances

        engines = SimpleNamespace(list_instances=list_instances)
        return (await trainer_capacity(kv, engines, LIMITS))[LORA]

    return asyncio.run(run())


def test_free_slot_on_running_trainer_is_available_now():
    capacity = snapshot({"t1": "running"}, {"m1": "t1"}, [])
    assert capacity.availability == "now"
    assert capacity.free_slots == 1
    assert capacity.running_trainers == 1


def test_waiting_models_take_free_slots_first():
    capacity = snapshot({"t1": "running"}, {"m1": "t1"}, ["m2"])
    assert capacity.free_slots == 0
    assert capacity.waiting_models == 1
    assert capacity.availability == "after_trainer_start"


def test_no_trainers_means_a_trainer_must_start():
    capacity = snapshot({}, {}, [])
    assert capacity.availability == "after_trainer_start"
    assert capacity.running_trainers == 0


def test_starting_and_draining_trainers_do_not_offer_free_slots():
    capacity = snapshot({"t1": "starting", "t2": "draining"}, {"m1": "t2"}, [])
    assert capacity.free_slots == 0
    assert capacity.running_trainers == 0
    assert capacity.availability == "after_trainer_start"


def test_full_when_every_trainer_is_used_and_cap_is_reached():
    capacity = snapshot(
        {"t1": "running", "t2": "running"},
        {"m1": "t1", "m2": "t1", "m3": "t2", "m4": "t2"},
        ["m5"],
    )
    assert capacity.availability == "full"
    assert capacity.waiting_models == 1


def test_stopped_trainers_free_room_under_the_cap():
    capacity = snapshot(
        {"t1": "running", "t2": "stopped"},
        {"m1": "t1", "m2": "t1"},
        [],
    )
    assert capacity.availability == "after_trainer_start"
