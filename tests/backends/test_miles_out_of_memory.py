import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from spindle.backends.miles_runtime import gradient_accumulation, out_of_memory
from spindle.backends.miles_runtime.runtime import MilesRuntime
from spindle.errors import TrainerOutOfMemory


def test_out_of_memory_fails_only_its_request():
    runtime = MilesRuntime.__new__(MilesRuntime)
    runtime._closed = False
    runtime._failure = None
    runtime._call = asyncio.run
    runtime._unit_ids = iter(range(10))
    results = [{"error": "CUDA out of memory", out_of_memory.OUT_OF_MEMORY: True}, []]

    async def run(*args):
        return results.pop(0)

    runtime._bridge = SimpleNamespace(forward_backward=run)
    call = lambda: runtime.forward_backward(  # noqa: E731
        ((0, {}),), loss_fn="cross_entropy", loss_fn_config={}, forward_only=False
    )

    with pytest.raises(TrainerOutOfMemory, match="out of GPU memory"):
        call()
    assert call() == []
    assert runtime._failure is None


def test_failed_pass_keeps_earlier_gradients(monkeypatch):
    monkeypatch.setattr(out_of_memory, "_recoverable", True)
    buffer = SimpleNamespace(grad_data=torch.tensor([8.0]))
    chunk = SimpleNamespace(
        buffers=[buffer],
        expert_parallel_buffers=[],
        zero_grad_buffer=lambda: buffer.grad_data.zero_(),
    )

    def run_forward_backward(*args):
        buffer.grad_data.add_(4.0)
        raise torch.OutOfMemoryError("CUDA out of memory")

    lora_model = SimpleNamespace(run_forward_backward=run_forward_backward)
    gradient_accumulation.install(lora_model)
    args = Namespace(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    with pytest.raises(torch.OutOfMemoryError):
        lora_model.run_forward_backward(args, 0, [chunk], "batch")
    assert buffer.grad_data.item() == 8.0


def _run_rank(rank, rendezvous, forward_only, results):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        synced = []

        def all_reduce(*args, **kwargs):
            value = torch.ones(1)
            dist.all_reduce(value)
            synced.append(value.item())

        megatron_model = SimpleNamespace(
            finalize_model_grads_with_empty_cache=all_reduce
        )
        lora_model = SimpleNamespace(aggregate_train_losses=all_reduce)
        out_of_memory.install(megatron_model=megatron_model, lora_model=lora_model)
        out_of_memory.configure(recoverable=True, group=dist.group.WORLD)
        manager = SimpleNamespace(stage="fallthrough")

        def run_pass(failing_rank):
            manager.stage = "replay_forward"
            if rank == failing_rank:
                raise torch.OutOfMemoryError(
                    "CUDA out of memory. Tried to allocate 30 GiB. GPU 0 has 80 GiB"
                )
            if not forward_only:
                megatron_model.finalize_model_grads_with_empty_cache()
            lora_model.aggregate_train_losses()
            manager.stage = "fallthrough"
            return {"per_datum": []}

        failed = out_of_memory.run(lambda: run_pass(1), replay_managers=[manager])
        stage = manager.stage
        succeeded = out_of_memory.run(lambda: run_pass(None), replay_managers=[manager])
        results[rank] = (failed, stage, succeeded, synced)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("forward_only", [False, True])
def test_peers_skip_collectives_when_one_rank_runs_out_of_memory(
    tmp_path, forward_only
):
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _run_rank,
            args=(f"file://{tmp_path}/rendezvous", forward_only, results),
            nprocs=2,
        )
        ranks = [results[0], results[1]]
    passes = 1 if forward_only else 2
    for failed, stage, succeeded, synced in ranks:
        assert failed == {
            "error": "CUDA out of memory. Tried to allocate 30 GiB",
            "out_of_memory": True,
        }
        assert stage == "fallthrough"
        # Only the second pass reached the all-reduces.
        assert succeeded == {"per_datum": []}
        assert synced == [2.0] * passes
