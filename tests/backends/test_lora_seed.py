import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from runtime_stubs import backend_runtime_imports


@pytest.mark.parametrize("fail", [False, True])
def test_seeded_reset_reproduces_slots_and_restores_rng(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    with backend_runtime_imports():
        spec = importlib.util.spec_from_file_location(
            "_test_lora_seed",
            Path(__file__).parents[2] / "src/spindle/backends/lora_seed.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    streams = {
        "dense": torch.Generator().manual_seed(100),
        "expert": torch.Generator().manual_seed(200),
    }
    initial_states = {name: stream.get_state() for name, stream in streams.items()}
    tracker = SimpleNamespace(
        get_states=lambda: dict(streams),
        set_states=lambda states: (streams.clear(), streams.update(states)),
    )

    @contextmanager
    def fork_rng() -> Iterator[None]:
        saved = dict(streams)
        try:
            yield
        finally:
            tracker.set_states(saved)

    monkeypatch.setattr(module, "_fork_rng", fork_rng)
    monkeypatch.setattr(module, "get_cuda_rng_tracker", lambda: tracker)
    monkeypatch.setattr(module, "_MODEL_PARALLEL_RNG_TRACKER_NAME", "dense")
    monkeypatch.setattr(
        module, "get_expert_parallel_rng_tracker_name", lambda: "expert"
    )
    topology = dict(pp=0, tp=0, ep=0, etp=0)
    monkeypatch.setattr(
        module,
        "parallel_state",
        SimpleNamespace(
            get_pipeline_model_parallel_rank=lambda: topology["pp"],
            get_tensor_model_parallel_rank=lambda: topology["tp"],
            get_expert_model_parallel_rank=lambda: topology["ep"],
            get_expert_tensor_parallel_rank=lambda: topology["etp"],
        ),
    )

    def reset(model: dict[int, torch.Tensor], slot: int) -> None:
        model[slot] = torch.stack(
            [torch.randn(8, generator=streams[name]) for name in ("dense", "expert")]
        )
        if fail:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(module, "clear_adapter_slot", reset)
    model = {}

    def initialize(slot: int, seed: int) -> torch.Tensor:
        if fail:
            with pytest.raises(RuntimeError, match="initialization failed"):
                module.reset_adapter_slot_with_seed(model, slot, seed)
        else:
            module.reset_adapter_slot_with_seed(model, slot, seed)
        assert all(
            torch.equal(stream.get_state(), initial_states[name])
            for name, stream in streams.items()
        )
        return model[slot].clone()

    first = initialize(0, 0)
    assert torch.equal(first, initialize(1, 0))
    assert not torch.equal(first, initialize(0, 42))
    assert torch.equal(first, initialize(0, 0))
    topology.update(tp=1)
    tp = initialize(0, 0)
    assert not torch.equal(first[0], tp[0])
    assert torch.equal(first[1], tp[1])
    topology.update(ep=1, etp=1)
    assert not torch.equal(tp[1], initialize(0, 0)[1])
    topology.update(pp=1)
    assert not torch.equal(tp, initialize(0, 0))
