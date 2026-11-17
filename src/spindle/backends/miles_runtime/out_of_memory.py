"""Fail only the training call that ran out of GPU memory, not the trainer.

Ranks that meet only in the gradient and loss all-reduces agree before the
first of them, so a rank that ran out of memory never leaves the others
waiting in a collective.
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


_recoverable = False
# Gloo group the ranks agree on; None on one rank.
_group: dist.ProcessGroup | None = None
_agreed = False


def configure(*, recoverable: bool, group: dist.ProcessGroup | None) -> None:
    global _recoverable, _group
    _recoverable = recoverable
    _group = group if recoverable else None


def recoverable() -> bool:
    return _recoverable


def install(*, megatron_model: ModuleType, lora_model: ModuleType) -> None:
    _agree_before(megatron_model, "finalize_model_grads_with_empty_cache")
    _agree_before(lora_model, "aggregate_train_losses")


def run(call: Callable[[], object], *, replay_managers: Sequence[object]) -> object:
    """Run one forward or forward-backward call; on out of memory, free it and return an error result."""
    global _agreed
    _agreed = False
    stages = [manager.stage for manager in replay_managers]
    try:
        return call()
    except torch.OutOfMemoryError as error:
        # After the agreement, the other ranks may already be in an all-reduce.
        if not _recoverable or _agreed:
            raise
        # Keep "CUDA out of memory. Tried to allocate N GiB"; the rest is allocator advice for the server.
        detail = ". ".join(str(error).split(". ")[:2])
        _agree(detail)
    except PeerOutOfMemory as error:
        detail = str(error)
    for manager, stage in zip(replay_managers, stages, strict=True):
        manager.stage = stage
    gc.collect()
    torch.cuda.empty_cache()
    return {"error": detail, OUT_OF_MEMORY: True}


def _agree_before(module: ModuleType, name: str) -> None:
    original = getattr(module, name)
    if getattr(original, "__spindle_out_of_memory__", False):
        return

    @wraps(original)
    def agree_then_call(*args: object, **kwargs: object) -> object:
        _agree(None)
        return original(*args, **kwargs)

    agree_then_call.__spindle_out_of_memory__ = True  # type: ignore[attr-defined]
    setattr(module, name, agree_then_call)


def _agree(failure: str | None) -> None:
    global _agreed
    if _group is None or _agreed:
        return
    _agreed = True
    failures: list[str | None] = [None] * dist.get_world_size(_group)
    dist.all_gather_object(failures, failure, group=_group)
    peer_failure = next((f for f in failures if f is not None), None)
    if failure is None and peer_failure is not None:
        raise PeerOutOfMemory(peer_failure)
