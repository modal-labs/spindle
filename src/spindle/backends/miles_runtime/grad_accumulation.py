"""Megatron ends every backward pass by reducing gradients across ranks. Miles
keeps gradients until the optimizer step, so those from earlier forward_backward
calls are reduced again. Two reduces change them: expert buffers are divided by
the data-parallel size but summed over the expert data-parallel group, and
weights replicated across tensor-parallel ranks are summed over those ranks.
"""

from __future__ import annotations

import math

_TOLERANCE = 0.01


def keep_grads_through_reduce(model) -> None:
    """Scale kept gradients so the next backward pass's reduces return them unchanged."""
    for chunk in model:
        averaged = chunk.ddp_config.average_in_collective
        for buffer in (*chunk.buffers, *chunk.expert_parallel_buffers):
            ranks_summed = 1 if averaged else buffer.data_parallel_group.size()
            gain = buffer.gradient_scaling_factor * ranks_summed
            if not math.isclose(gain, 1.0):
                buffer.scale_gradients(1 / gain)
        tensor_parallel = chunk.tp_group.size()
        if tensor_parallel > 1:
            for param in chunk.params_with_grad:
                if _summed_over_tensor_parallel(param, chunk.config):
                    param.main_grad /= tensor_parallel


def check_reduce_keeps_grads(model, finalize_model_grads) -> None:
    """Reduce gradients of all ones once and fail if any comes back changed.

    keep_grads_through_reduce has to match every reduce that ends a backward pass.
    One it misses would rescale kept gradients silently, so find out at startup.
    """
    buffers = [
        buffer
        for chunk in model
        for buffer in (*chunk.buffers, *chunk.expert_parallel_buffers)
    ]
    try:
        for buffer in buffers:
            buffer.grad_data.fill_(1)
        keep_grads_through_reduce(model)
        finalize_model_grads(list(model), None)
        if all(_all_ones(buffer.grad_data) for buffer in buffers):
            return
        name, value = next(
            (
                (name, param.main_grad.flatten()[0].item())
                for chunk in model
                for name, param in chunk.module.named_parameters()
                if hasattr(param, "main_grad") and not _all_ones(param.main_grad)
            ),
            ("a gradient buffer", float("nan")),
        )
        raise RuntimeError(
            "The gradient reduce changes gradients kept between forward_backward "
            f"calls: a gradient of 1 for {name} came back as {value:g}."
        )
    finally:
        for chunk in model:
            chunk.zero_grad_buffer()


def _all_ones(grad) -> bool:
    return bool(grad.min() >= 1 - _TOLERANCE) and bool(grad.max() <= 1 + _TOLERANCE)


def _summed_over_tensor_parallel(param, config) -> bool:
    # Megatron also sums q and k layer norm weights, which adapters never are.
    if getattr(param, "sum_gradients_across_tp_domain", False):
        return True
    if getattr(param, "average_gradients_across_tp_domain", False):
        return False
    return config.sequence_parallel and getattr(param, "sequence_parallel", False)
