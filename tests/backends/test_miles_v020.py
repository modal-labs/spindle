from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    modules = {
        "miles.backends.training_utils.checkpoint.adapter.parametrization": dict(
            base_model_fingerprint=lambda config: "fingerprint"
        ),
        "miles.ray.placement_group": dict(create_trainer_handles=None),
        "miles.ray.specs.train": dict(
            ACTOR_ROLE="actor", TrainerSpec=None, compute_trainer_configs=None
        ),
        "miles.ray.train.init_request": dict(TrainerControllerInitRequest=None),
        "miles.ray.wiring": dict(compute_backend_capability=None),
        "miles.utils.hf_utils.config": dict(
            load_hf_config=lambda path: SimpleNamespace(to_dict=dict)
        ),
        "miles.utils.lora.hf_lora_targets": dict(get_hf_lora_targets=lambda config: ()),
    }
    for name, members in modules.items():
        module = ModuleType(name)
        vars(module).update(members)
        monkeypatch.setitem(sys.modules, name, module)
    name = "spindle.backends.miles_runtime._test_v020"
    path = Path(__file__).parents[2] / "src/spindle/backends/miles_runtime/v020.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("restore_optimizer", [False, True])
@pytest.mark.parametrize("lora_type", ["lora", "canonical_lora"])
def test_native_restore_preserves_slot_configuration_and_optimizer_intent(
    api: ModuleType, restore_optimizer: bool, lora_type: str
) -> None:
    class NativeBridge:
        async def load_slot(
            self,
            slot: int,
            rank: int,
            alpha: float,
            source: dict[str, object] | None = None,
            seed: int | None = None,
        ) -> dict[str, object] | None:
            assert seed == (None if source is not None else 0)
            if source is not None:
                assert source["load_optimizer"] is restore_optimizer
                assert source["path"] == "/checkpoint/miles"
                assert source["parametrization_fields"]["lora_rank"] == rank
                assert source["parametrization_fields"]["lora_alpha"] == alpha
                assert source["parametrization_fields"]["lora_type"] == lora_type
            return None

    args = SimpleNamespace(
        hf_checkpoint="/model", lora_type=lora_type, experts_shared_outer_loras=False
    )
    bridge = api.AdapterBridge(NativeBridge(), args)

    async def run() -> None:
        await bridge.load_slot(0, 4, 8, seed=0)
        await bridge.load_slot(1, 8, 16, seed=0)
        await bridge.load_slot(
            0,
            4,
            8,
            ckpt_path="/checkpoint/miles",
            load_optimizer=restore_optimizer,
            seed=99,
        )

    asyncio.run(run())
    assert bridge.slots == {0: (4, 8), 1: (8, 16)}


def test_rejected_native_load_keeps_slot_configuration(api: ModuleType) -> None:
    class NativeBridge:
        async def load_slot(
            self,
            slot: int,
            rank: int,
            alpha: float,
            source: dict[str, object] | None = None,
            seed: int | None = None,
        ) -> dict[str, object] | None:
            return {"error": "incompatible checkpoint", "error_category": "user"}

    bridge = api.AdapterBridge(
        NativeBridge(),
        SimpleNamespace(
            hf_checkpoint="/model", lora_type="lora", experts_shared_outer_loras=False
        ),
    )
    bridge.slots[0] = (4, 8)
    result = asyncio.run(bridge.load_slot(0, 8, 16, ckpt_path="/invalid"))
    assert result["error"] == "incompatible checkpoint"
    assert bridge.slots == {0: (4, 8)}


def test_native_actor_gate_keeps_other_workers_and_can_be_disabled(
    api: ModuleType,
) -> None:
    class Spec:
        def __init__(self, worker_class: str) -> None:
            self.worker_class = worker_class

        def model_copy(self, *, update: dict) -> Spec:
            return Spec(update["worker_class"])

        @classmethod
        def create(cls, config: str) -> Spec:
            return cls(config)

    api.TrainerSpec = Spec
    worker = "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"
    api.configure_actor(shared_prefix=True)
    once = Spec.create.__func__
    api.configure_actor(shared_prefix=True)
    assert Spec.create.__func__ is once
    assert Spec.create(worker).worker_class.endswith(
        "prefix_actor.SharedPrefixTrainRayActor"
    )
    assert Spec.create("full-trainer").worker_class == "full-trainer"
    api.configure_actor(shared_prefix=False)
    assert Spec.create(worker).worker_class == api.ACTOR_CLASS
