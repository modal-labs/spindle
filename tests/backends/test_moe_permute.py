import pytest

torch = pytest.importorskip("torch")

from spindle.backends.miles_runtime.moe_permute import PermuteRows  # noqa: E402


@pytest.mark.parametrize("deterministic", [False, True])
def test_backward_matches_index_select(deterministic: bool) -> None:
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        tokens = torch.randn(10, 8, requires_grad=True)
        index = torch.randint(0, 10, (30,))
        grad = torch.randn(30, 8)

        (ours,) = torch.autograd.grad(PermuteRows.apply(tokens, index), tokens, grad)
        (reference,) = torch.autograd.grad(tokens.index_select(0, index), tokens, grad)
    finally:
        torch.use_deterministic_algorithms(before)

    torch.testing.assert_close(ours, reference)
