from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from spindle.backends.miles_runtime import gradient_accumulation


def run_pass(expert_parallel: int, *, sequence_parallel_adapter: bool = False) -> float:
    """Keep 8 from an earlier call, then run a pass whose reduce scales by 1/4."""
    expert_buffer = SimpleNamespace(grad_data=torch.tensor([8.0]))
    adapter = SimpleNamespace(sequence_parallel=sequence_parallel_adapter)
    chunk = SimpleNamespace(
        buffers=[],
        expert_parallel_buffers=[expert_buffer],
        zero_grad_buffer=lambda: expert_buffer.grad_data.zero_(),
        config=SimpleNamespace(sequence_parallel=True),
        params_with_grad=[adapter],
    )

    def run_forward_backward(*args: object) -> None:
        expert_buffer.grad_data.add_(4.0).mul_(0.25)

    lora_model = SimpleNamespace(run_forward_backward=run_forward_backward)
    gradient_accumulation.install(lora_model)
    args = Namespace(
        tensor_model_parallel_size=2,
        expert_model_parallel_size=expert_parallel,
        expert_tensor_parallel_size=1,
    )
    lora_model.run_forward_backward(args, 0, [chunk], "batch")
    return expert_buffer.grad_data.item()


@pytest.mark.parametrize(
    ("expert_parallel", "sequence_parallel_adapter"), [(1, False), (2, True)]
)
def test_kept_gradients_skip_a_reduce_that_would_rescale_them(
    expert_parallel: int, sequence_parallel_adapter: bool
) -> None:
    assert (
        run_pass(expert_parallel, sequence_parallel_adapter=sequence_parallel_adapter)
        == 9.0
    )


def test_layouts_whose_reduce_keeps_gradients_skip_the_copy() -> None:
    assert run_pass(2) == (8.0 + 4.0) * 0.25
