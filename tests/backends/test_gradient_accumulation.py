from types import SimpleNamespace

import torch

from spindle.backends.miles_runtime import gradient_accumulation


def test_a_pass_reduces_only_its_own_gradients() -> None:
    expert_buffer = SimpleNamespace(grad_data=torch.tensor([8.0]))
    chunk = SimpleNamespace(
        buffers=[],
        expert_parallel_buffers=[expert_buffer],
        zero_grad_buffer=lambda: expert_buffer.grad_data.zero_(),
    )

    def run_forward_backward(*args: object) -> None:
        # Megatron's expert reduce scales the whole buffer, not just this pass.
        expert_buffer.grad_data.add_(4.0).mul_(0.25)

    lora_model = SimpleNamespace(run_forward_backward=run_forward_backward)
    gradient_accumulation.install(lora_model)
    lora_model.run_forward_backward(None, 0, [chunk], "batch")

    assert expert_buffer.grad_data.item() == 8.0 + 4.0 * 0.25
