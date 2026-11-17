import pytest
import torch

from spindle.backends.megatron_runtime.common.chunked_logprobs import chunked_logprobs


@pytest.mark.parametrize(("fp32", "temperature"), [(True, 1.0), (True, 0.7), (False, 1.0)])
def test_chunked_logprobs_match_full_log_softmax(fp32: bool, temperature: float) -> None:
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
