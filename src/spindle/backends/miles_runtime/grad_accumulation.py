"""Megatron ends every backward pass by reducing gradients across ranks. Miles
keeps gradients until the optimizer step, so those from earlier forward_backward
calls are reduced again. Two reduces change them: expert buffers are divided by
the data-parallel size but summed over the expert data-parallel group, and
weights replicated across tensor-parallel ranks are summed over those ranks.
"""

from __future__ import annotations

import math


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


def _summed_over_tensor_parallel(param, config) -> bool:
    # The test Megatron's finalize_model_grads applies to trainable weights.
    if getattr(param, "sum_gradients_across_tp_domain", False):
        return True
    if getattr(param, "average_gradients_across_tp_domain", False):
        return False
    return config.sequence_parallel and getattr(param, "sequence_parallel", False)
