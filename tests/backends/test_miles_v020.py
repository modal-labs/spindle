import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def api(monkeypatch):
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
def test_native_restore_preserves_slot_configuration_and_optimizer_intent(
    api, restore_optimizer
):
    class NativeBridge:
        async def load_slot(self, slot, rank, alpha, source=None):
            if source is not None:
                assert source["load_optimizer"] is restore_optimizer
                assert source["path"] == "/checkpoint/miles"
                assert source["parametrization_fields"]["lora_rank"] == rank
                assert source["parametrization_fields"]["lora_alpha"] == alpha
            return None

    args = SimpleNamespace(
        hf_checkpoint="/model", lora_type="lora", experts_shared_outer_loras=False
    )
    bridge = api.AdapterBridge(NativeBridge(), args)

    async def run():
        await bridge.load_slot(0, 4, 8)
        await bridge.load_slot(1, 8, 16)
        await bridge.load_slot(
            0, 4, 8, ckpt_path="/checkpoint/miles", load_optimizer=restore_optimizer
        )

    asyncio.run(run())
    assert bridge.slots == {0: (4, 8), 1: (8, 16)}


def test_rejected_native_load_keeps_slot_configuration(api):
    class NativeBridge:
        async def load_slot(self, slot, rank, alpha, source=None):
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
