import pytest

torch = pytest.importorskip("torch")

from spindle.backends.megatron_runtime.common.chunked_logprobs import (  # noqa: E402
    chunked_logprobs,
    defer_output_layer,
)


@pytest.mark.parametrize(
    ("fp32", "temperature"), [(True, 1.0), (True, 0.7), (False, 1.0)]
)
def test_chunked_logprobs_match_full_log_softmax(
    fp32: bool, temperature: float
) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(37, 16, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(50, 16, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 50, (37,))
    labels[[3, 10]] = -100
    grad = torch.randn(37, dtype=torch.float64)

    logprobs = chunked_logprobs(
        hidden, weight, labels, chunk=8, fp32=fp32, temperature=temperature
    )
    (logprobs.double() * grad).sum().backward()
    chunked = (logprobs.detach().double(), hidden.grad.clone(), weight.grad.clone())
    hidden.grad = weight.grad = None

    full = (
        torch.log_softmax(hidden @ weight.t() / temperature, dim=-1)
        .gather(1, labels.clamp(min=0)[:, None])
        .squeeze(1)
        .masked_fill(labels == -100, 0)
    )
    (full * grad).sum().backward()

    for got, want in zip(chunked, (full.detach(), hidden.grad, weight.grad)):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("hidden_dtype", [torch.float32, torch.bfloat16])
def test_chunked_logprobs_accept_mixed_hidden_and_bf16_weight(
    fp32: bool, hidden_dtype: torch.dtype
) -> None:
    hidden = torch.randn(9, 8, dtype=hidden_dtype, requires_grad=True)
    weight = torch.randn(12, 8, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 12, (9,))

    chunked_logprobs(hidden, weight, labels, chunk=4, fp32=fp32).sum().backward()

    assert hidden.grad.dtype == hidden_dtype
    assert weight.grad.dtype == torch.bfloat16


class _OutputLayer(torch.nn.Module):
    def _forward_impl(self, **kwargs: object) -> torch.Tensor:
        raise AssertionError("output layer should be deferred")


def test_defer_output_layer_returns_input_and_records_weight() -> None:
    model = torch.nn.Module()
    model.output_layer = _OutputLayer()
    deferred = defer_output_layer([model])
    hidden = torch.randn(3, 1, 4)
    weight = torch.randn(5, 4)

    # Megatron's ColumnParallelLinear passes these keyword arguments.
    out = model.output_layer._forward_impl(
        input=hidden,
        weight=weight,
        bias=None,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=False,
        sequence_parallel=False,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
    )

    assert out is hidden
    assert deferred["weight"] is weight
    with pytest.raises(ValueError, match="bias"):
        model.output_layer._forward_impl(input=hidden, weight=weight, bias=weight[:, 0])
    with pytest.raises(ValueError, match="no output layer"):
        defer_output_layer([torch.nn.Module()])
