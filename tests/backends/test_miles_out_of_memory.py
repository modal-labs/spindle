import asyncio
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spindle.backends.miles_runtime import out_of_memory
from spindle.backends.miles_runtime.runtime import MilesRuntime
from spindle.errors import TrainerOutOfMemory


@pytest.mark.parametrize("forward_only", [False, True])
def test_out_of_memory_fails_only_its_request(forward_only):
    runtime = MilesRuntime.__new__(MilesRuntime)
    runtime._closed = False
    runtime._failure = None
    runtime._call = asyncio.run
    runtime._unit_ids = iter(range(10))
    results = [{"error": "CUDA out of memory", out_of_memory.OUT_OF_MEMORY: True}, []]
    discarded = []

    async def run(*args):
        return results.pop(0)

    async def execute(method, **kwargs):
        discarded.append((method, kwargs))
        return [None]

    runtime._bridge = SimpleNamespace(forward_backward=run, forward_only=run)
    runtime._trainer = SimpleNamespace(_execute_slots=execute)
    rows = ((1, {}), (0, {}), (1, {}))

    with pytest.raises(TrainerOutOfMemory, match="out of GPU memory"):
        runtime.forward_backward(
            rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=forward_only
        )
    assert (
        runtime.forward_backward(
            rows, loss_fn="cross_entropy", loss_fn_config={}, forward_only=forward_only
        )
        == []
    )
    expected = [] if forward_only else [("discard_gradients", {"slots": [0, 1]})]
    assert discarded == expected
    assert runtime._failure is None


def _run_rank(rank, rendezvous, results):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        synced = []

        def all_reduce_gradients(*args, **kwargs):
            gradient = torch.ones(1)
            dist.all_reduce(gradient)
            synced.append(gradient.item())

        model = SimpleNamespace(
            finalize_model_grads_with_empty_cache=all_reduce_gradients
        )
        out_of_memory.install(model)
        out_of_memory.configure(recoverable=True, group=dist.group.WORLD)
        manager = SimpleNamespace(stage="fallthrough")

        def forward_backward(failing_rank):
            manager.stage = "replay_forward"
            if rank == failing_rank:
                raise torch.OutOfMemoryError("CUDA out of memory")
            model.finalize_model_grads_with_empty_cache()
            manager.stage = "fallthrough"
            return {"per_datum": []}

        failed = out_of_memory.run(
            lambda: forward_backward(0), backward=True, replay_managers=[manager]
        )
        stage = manager.stage
        succeeded = out_of_memory.run(
            lambda: forward_backward(None), backward=True, replay_managers=[manager]
        )
        results[rank] = (failed, stage, succeeded, synced)
    finally:
        dist.destroy_process_group()


def test_peers_skip_gradient_sync_when_one_rank_runs_out_of_memory(tmp_path):
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_run_rank, args=(f"file://{tmp_path}/rendezvous", results), nprocs=2)
        (failed, stage, succeeded, synced), (peer_failed, peer_stage, *peer_rest) = (
            results[0],
            results[1],
        )
    assert failed == {"error": "CUDA out of memory", "out_of_memory": True}
    assert peer_failed["out_of_memory"]
    assert "another trainer rank" in peer_failed["error"]
    assert stage == peer_stage == "fallthrough"
    # Only the second pass reached the gradient all-reduce, on both ranks.
    assert [succeeded, synced] == peer_rest == [{"per_datum": []}, [2.0]]
