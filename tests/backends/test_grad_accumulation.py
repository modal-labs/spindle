from types import SimpleNamespace

import pytest
import torch

from spindle.backends.miles_runtime.grad_accumulation import keep_grads_through_reduce


def group(ranks):
    return SimpleNamespace(size=lambda: ranks)


class Buffer:
    """The parts of Megatron's gradient buffer that its reduce uses."""

    def __init__(self, scaling, ranks):
        self.grad_data = torch.zeros(3)
        self.gradient_scaling_factor = scaling
        self.data_parallel_group = group(ranks)

    def scale_gradients(self, factor):
        self.grad_data *= factor


def chunk(
    *, buffers=(), expert_buffers=(), params=(), tensor_parallel=1, averaged=False
):
    return SimpleNamespace(
        ddp_config=SimpleNamespace(average_in_collective=averaged),
        config=SimpleNamespace(sequence_parallel=tensor_parallel > 1),
        buffers=list(buffers),
        expert_parallel_buffers=list(expert_buffers),
        params_with_grad=list(params),
        tp_group=group(tensor_parallel),
    )


def megatron_reduce(buffers, *, averaged=False):
    """Megatron's grad sync: scale each rank's buffer, then reduce over the group."""
    for buffer in buffers:
        buffer.grad_data *= buffer.gradient_scaling_factor
    total = torch.stack([buffer.grad_data for buffer in buffers]).sum(dim=0)
    for buffer in buffers:
        buffer.grad_data.copy_(total / len(buffers) if averaged else total)


def accumulate(calls, *, scaling, ranks, expert, keep, loss_scale, averaged=False):
    """Run forward_backward calls on every rank of one group; return rank 0's gradient."""
    buffers = [Buffer(scaling, ranks) for _ in range(ranks)]
    chunks = [
        chunk(
            buffers=[] if expert else [buffer],
            expert_buffers=[buffer] if expert else [],
            averaged=averaged,
        )
        for buffer in buffers
    ]
    for call in calls:
        for rank, buffer in enumerate(buffers):
            if keep:
                keep_grads_through_reduce([chunks[rank]])
            buffer.grad_data += loss_scale * call[rank]
        megatron_reduce(buffers, averaged=averaged)
    return buffers[0].grad_data


@pytest.mark.parametrize(
    ("data_parallel", "expert_ranks", "averaged"),
    [
        (1, 2, False),
        (2, 1, False),
        (4, 1, False),
        (6, 1, False),
        (4, 8, False),
        (2, 2, False),
        (1, 2, True),
        (4, 2, True),
    ],
)
def test_expert_gradients_add_up_across_forward_backward_calls(
    data_parallel, expert_ranks, averaged
):
    torch.manual_seed(0)
    calls = [torch.randn(expert_ranks, 3) for _ in range(4)]
    expected = torch.stack(calls).sum(dim=(0, 1))
    # Megatron scales expert buffers so that the reduce divides by the data-parallel
    # size; Miles multiplies the loss by that size.
    scaling = expert_ranks / data_parallel if averaged else 1 / data_parallel
    layout = dict(
        scaling=scaling,
        ranks=expert_ranks,
        expert=True,
        loss_scale=data_parallel,
        averaged=averaged,
    )

    kept = accumulate(calls, **layout, keep=True)
    torch.testing.assert_close(kept, expected, rtol=1e-5, atol=1e-6)

    if data_parallel != expert_ranks:
        unkept = accumulate(calls, **layout, keep=False)
        assert not torch.allclose(unkept, expected, rtol=1e-2)


@pytest.mark.parametrize("averaged", [False, True])
def test_dense_gradients_are_left_alone(averaged):
    calls = [torch.ones(4, 3), 2 * torch.ones(4, 3)]
    layout = dict(
        scaling=1.0 if averaged else 1 / 4,
        ranks=4,
        expert=False,
        loss_scale=4,
        averaged=averaged,
    )

    kept = accumulate(calls, **layout, keep=True)
    assert torch.equal(kept, accumulate(calls, **layout, keep=False))
    assert torch.equal(kept, torch.full((3,), 12.0))


def tagged(**tags):
    return SimpleNamespace(main_grad=torch.zeros(3), **tags)


@pytest.mark.parametrize("tensor_parallel", [2, 4, 8])
def test_gradients_summed_over_tensor_parallel_ranks_add_up(tensor_parallel):
    torch.manual_seed(0)
    calls = [torch.randn(tensor_parallel, 3) for _ in range(3)]
    expected = torch.stack(calls).sum(dim=(0, 1))

    def run(keep):
        ranks = [
            {
                "replicated": tagged(sequence_parallel=True),
                "marked": tagged(sum_gradients_across_tp_domain=True),
                "averaged": tagged(
                    sequence_parallel=True, average_gradients_across_tp_domain=True
                ),
                "sharded": tagged(),
            }
            for _ in range(tensor_parallel)
        ]
        chunks = [
            chunk(params=params.values(), tensor_parallel=tensor_parallel)
            for params in ranks
        ]
        for call in calls:
            for rank, params in enumerate(ranks):
                if keep:
                    keep_grads_through_reduce([chunks[rank]])
                for param in params.values():
                    param.main_grad += call[rank]
            # Megatron's finalize_model_grads: sum or average the tagged gradients.
            for name, reduce in (
                ("replicated", torch.sum),
                ("marked", torch.sum),
                ("averaged", torch.mean),
            ):
                total = reduce(
                    torch.stack([params[name].main_grad for params in ranks]), dim=0
                )
                for params in ranks:
                    params[name].main_grad.copy_(total)
        return ranks[0]

    kept = run(keep=True)
    torch.testing.assert_close(
        kept["replicated"].main_grad, expected, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(kept["marked"].main_grad, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(kept["sharded"].main_grad, torch.stack(calls).sum(dim=0)[0])
    unkept = run(keep=False)
    assert torch.equal(kept["averaged"].main_grad, unkept["averaged"].main_grad)
    assert not torch.allclose(unkept["replicated"].main_grad, expected, rtol=1e-2)
