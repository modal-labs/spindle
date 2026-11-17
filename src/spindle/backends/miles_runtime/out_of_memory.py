"""Fail only the training call that ran out of GPU memory, not the trainer.

Ranks that only meet in the gradient all-reduce agree before it, so a rank
that ran out of memory never leaves the others waiting in a collective.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from functools import wraps
from types import ModuleType

import torch
import torch.distributed as dist

OUT_OF_MEMORY = "out_of_memory"


class PeerOutOfMemory(RuntimeError):
    pass


class _State:
    recoverable = True
    # Gloo group the ranks agree on before the gradient all-reduce; None on one rank.
    group: object | None = None
    agreed = False


_state = _State()


def configure(*, recoverable: bool, group: object | None) -> None:
    _state.recoverable = recoverable
    _state.group = group if recoverable else None


def install(megatron_model: ModuleType) -> None:
    original = megatron_model.finalize_model_grads_with_empty_cache
    if getattr(original, "__spindle_out_of_memory__", False):
        return

    @wraps(original)
    def finalize(*args: object, **kwargs: object) -> object:
        _agree(failed=False)
        return original(*args, **kwargs)

    finalize.__spindle_out_of_memory__ = True  # type: ignore[attr-defined]
    megatron_model.finalize_model_grads_with_empty_cache = finalize


def run(
    call: Callable[[], object], *, backward: bool, replay_managers: Sequence[object]
) -> object:
    """Run one forward or forward-backward call; on out of memory, free it and return an error result."""
    _state.agreed = False
    stages = [manager.stage for manager in replay_managers]
    try:
        return call()
    except torch.OutOfMemoryError as error:
        # After the agreement, the other ranks may already be in the all-reduce.
        if not _state.recoverable or (backward and _state.agreed):
            raise
        # Keep "CUDA out of memory. Tried to allocate N GiB"; the rest is allocator advice for the server.
        detail = ". ".join(str(error).split(". ")[:2])
        if backward:
            _agree(failed=True)
    except PeerOutOfMemory as error:
        detail = str(error)
    for manager, stage in zip(replay_managers, stages, strict=True):
        manager.stage = stage
    gc.collect()
    torch.cuda.empty_cache()
    return {"error": detail, OUT_OF_MEMORY: True}


def _agree(*, failed: bool) -> None:
    if _state.group is None or _state.agreed:
        return
    _state.agreed = True
    any_failed = torch.tensor([int(failed)])
    dist.all_reduce(any_failed, op=dist.ReduceOp.MAX, group=_state.group)
    if any_failed.item() and not failed:
        raise PeerOutOfMemory("another trainer rank ran out of GPU memory")
