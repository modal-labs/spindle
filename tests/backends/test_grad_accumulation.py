from types import SimpleNamespace

import pytest
import torch

from spindle.backends.miles_runtime import grad_accumulation
from spindle.backends.miles_runtime.grad_accumulation import (
    check_reduce_keeps_grads,
    keep_grads_through_reduce,
)


class Buffer:
    """The parts of Megatron's gradient buffer that its reduce uses."""

    def __init__(self, scaling, ranks):
        self.grad_data = torch.zeros(4)
        self.gradient_scaling_factor = scaling
        self.data_parallel_group = SimpleNamespace(size=lambda: ranks)

    def scale_gradients(self, factor):
        self.grad_data *= factor


def weight(buffer, tp_reduce=None, **tags):
    """A trainable weight whose gradient Megatron sums or averages over TP ranks."""
    return SimpleNamespace(main_grad=buffer.grad_data[:2], tp_reduce=tp_reduce, **tags)


def chunk(dense, expert, *, averaged=False, tensor_parallel=1, weights=None):
    weights = weights or {
        "dense.weight": weight(dense),
        "expert.weight": weight(expert),
    }
    buffers = [dense, expert]
    return SimpleNamespace(
        ddp_config=SimpleNamespace(average_in_collective=averaged),
        config=SimpleNamespace(sequence_parallel=tensor_parallel > 1),
        buffers=[dense],
        expert_parallel_buffers=[expert],
        params_with_grad=list(weights.values()),
        tp_group=SimpleNamespace(size=lambda: tensor_parallel),
        module=SimpleNamespace(named_parameters=weights.items),
        zero_grad_buffer=lambda: [buffer.grad_data.zero_() for buffer in buffers],
    )


def megatron_reduce(model, num_tokens=None):
    """Megatron's reduce, on one rank, for gradients every rank already agrees on."""
    for part in model:
        for buffer in (*part.buffers, *part.expert_parallel_buffers):
            buffer.grad_data *= buffer.gradient_scaling_factor
            if not part.ddp_config.average_in_collective:
                buffer.grad_data *= buffer.data_parallel_group.size()
        for param in part.params_with_grad:
            if param.tp_reduce == "sum":
                param.main_grad *= part.tp_group.size()


LAYOUTS = {
    "TP2/EP1": dict(dense=(1.0, 1), expert=(1.0, 2)),
    "TP2/EP4/DP2": dict(dense=(1 / 2, 2), expert=(1 / 2, 1)),
    "TP4/EP24/DP6": dict(dense=(1 / 6, 6), expert=(1 / 6, 1)),
    "averaged": dict(dense=(1.0, 4), expert=(0.5, 2), averaged=True),
}


def layout(name, **kwargs):
    spec = dict(LAYOUTS[name])
    return chunk(
        Buffer(*spec.pop("dense")), Buffer(*spec.pop("expert")), **spec, **kwargs
    )


@pytest.mark.parametrize("name", LAYOUTS)
def test_kept_gradients_come_back_from_the_reduce_unchanged(name):
    model = [layout(name)]
    kept = [torch.randn(4), torch.randn(4)]
    for buffer, grad in zip(
        (*model[0].buffers, *model[0].expert_parallel_buffers), kept
    ):
        buffer.grad_data.copy_(grad)

    keep_grads_through_reduce(model)
    megatron_reduce(model)

    torch.testing.assert_close(model[0].buffers[0].grad_data, kept[0])
    torch.testing.assert_close(model[0].expert_parallel_buffers[0].grad_data, kept[1])


def test_gradients_summed_over_tensor_parallel_ranks_come_back_unchanged():
    buffer = Buffer(1.0, 1)
    weights = {
        "replicated": weight(buffer, "sum", sequence_parallel=True),
        "marked": weight(Buffer(1.0, 1), "sum", sum_gradients_across_tp_domain=True),
        "averaged": weight(
            Buffer(1.0, 1),
            sequence_parallel=True,
            average_gradients_across_tp_domain=True,
        ),
        "sharded": weight(Buffer(1.0, 1)),
    }
    model = [chunk(buffer, Buffer(1.0, 1), tensor_parallel=4, weights=weights)]
    kept = {name: torch.randn(2) for name in weights}
    for name, param in weights.items():
        param.main_grad.copy_(kept[name])

    keep_grads_through_reduce(model)
    megatron_reduce(model)

    for name, param in weights.items():
        torch.testing.assert_close(param.main_grad, kept[name])


def test_startup_check_passes_and_leaves_no_gradients():
    model = [layout("TP2/EP1")]
    check_reduce_keeps_grads(model, megatron_reduce)
    assert not model[0].buffers[0].grad_data.any()
    assert not model[0].expert_parallel_buffers[0].grad_data.any()


@pytest.mark.parametrize(
    ("name", "came_back"), [("TP2/EP1", "2"), ("TP2/EP4/DP2", "0.5")]
)
def test_startup_check_names_a_weight_the_reduce_changes(monkeypatch, name, came_back):
    monkeypatch.setattr(
        grad_accumulation, "keep_grads_through_reduce", lambda model: None
    )
    model = [layout(name)]
    with pytest.raises(RuntimeError, match=f"expert.weight came back as {came_back}"):
        check_reduce_keeps_grads(model, megatron_reduce)
    assert not model[0].expert_parallel_buffers[0].grad_data.any()
