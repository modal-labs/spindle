import copy
from types import SimpleNamespace

import pytest
import torch

from spindle.backends.megatron_runtime.common.fused_adam import (
    bound_fused_adam_tensor_lists,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA and Transformer Engine"
)
def test_large_fused_adam_groups_match_adamw_and_restore():
    # Transformer Engine is installed only in the GPU image.
    from transformer_engine.pytorch.optimizers import FusedAdam
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    def prepare(optimizer):
        chain = object.__new__(ChainedOptimizer)
        chain.chained_optimizers = [
            SimpleNamespace(optimizer=optimizer),
            SimpleNamespace(
                optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
            ),
        ]
        bound_fused_adam_tensor_lists(chain)

    torch.manual_seed(1234)
    values = torch.randn(6000, 4, device="cuda")
    parameters = [torch.nn.Parameter(row.clone()) for row in values]
    reference = torch.nn.Parameter(values.clone())
    options = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    optimizer = FusedAdam(parameters, **options)
    prepare(optimizer)
    expected = torch.optim.AdamW([reference], foreach=True, **options)
    restored = None

    for step in range(3):
        gradients = torch.randn_like(values)
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient.clone()
        reference.grad = gradients.clone()
        optimizer.step()
        expected.step()
        actual = torch.stack(parameters)
        torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
        if restored is not None:
            restored_parameters, restored_optimizer = restored
            for parameter, gradient in zip(restored_parameters, gradients, strict=True):
                parameter.grad = gradient.clone()
            restored_optimizer.step()
            torch.testing.assert_close(
                torch.stack(restored_parameters), actual, atol=0, rtol=0
            )
        if step == 1:
            restored_parameters = [
                torch.nn.Parameter(parameter.detach().clone())
                for parameter in parameters
            ]
            restored_optimizer = FusedAdam(restored_parameters, **options)
            restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            prepare(restored_optimizer)
            restored = restored_parameters, restored_optimizer
