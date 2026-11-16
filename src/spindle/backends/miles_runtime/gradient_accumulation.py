"""Backport of radixark/miles#4480 for the Miles v0.1.x images; v0.2.0 has it.

Miles v0.1.x keeps gradients across forward_backward calls and lets each pass's
reduce run over them again. That rescales kept gradients on two layouts: expert
gradients when experts are sharded differently from tensor parallelism, and
sequence-parallel adapters, which are summed over tensor-parallel ranks. Other
layouts skip the extra copy of the gradient buffers.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from functools import wraps
from types import ModuleType
from typing import Any


@contextmanager
def accumulate_gradients(model_chunks: Sequence[Any]) -> Generator[None]:
    # Synchronizing retained expert gradients would scale earlier requests again.
    buffers = [
        buffer
        for chunk in model_chunks
        for buffer in chunk.buffers + chunk.expert_parallel_buffers
    ]
    accumulated = [buffer.grad_data.clone() for buffer in buffers]
    for chunk in model_chunks:
        chunk.zero_grad_buffer()
    try:
        yield
    except BaseException:
        for buffer, previous in zip(buffers, accumulated, strict=True):
            buffer.grad_data.copy_(previous)
        raise
    else:
        for buffer, previous in zip(buffers, accumulated, strict=True):
            buffer.grad_data.add_(previous)


def install(lora_model: ModuleType) -> None:
    original: Callable[..., object] = lora_model.run_forward_backward
    if getattr(original, "__spindle_gradient_accumulation__", False):
        return

    @wraps(original)
    def run_forward_backward(
        args: Namespace,
        batch_id: int,
        model: Sequence[Any],
        *rest: object,
        **kwargs: object,
    ) -> object:
        if kwargs.get("forward_only", False) or not _rescales_kept_gradients(
            args, model
        ):
            return original(args, batch_id, model, *rest, **kwargs)
        with accumulate_gradients(model):
            return original(args, batch_id, model, *rest, **kwargs)

    run_forward_backward.__spindle_gradient_accumulation__ = True  # type: ignore[attr-defined]
    lora_model.run_forward_backward = run_forward_backward


def _rescales_kept_gradients(args: Namespace, model: Sequence[Any]) -> bool:
    expert_shards = args.expert_model_parallel_size * args.expert_tensor_parallel_size
    if expert_shards != args.tensor_model_parallel_size:
        return True
    return args.tensor_model_parallel_size > 1 and any(
        chunk.config.sequence_parallel and getattr(param, "sequence_parallel", False)
        for chunk in model
        for param in chunk.params_with_grad
    )
