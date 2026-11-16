import pytest

torch = pytest.importorskip("torch")

from spindle.backends.miles_runtime.response_logits import split_layout  # noqa: E402


def miles_windows(total_lengths, response_lengths):
    end, windows = 0, []
    for total, response in zip(total_lengths, response_lengths):
        end += total
        windows.append((end - response - 1, end - 1))
    return windows


def test_split_pieces_and_gradient_match_per_row_slices() -> None:
    totals, responses = [5, 7, 3, 9], [4, 6, 0, 8]
    rows = sum(totals) + 4
    sizes, pieces_of_rows = split_layout(totals, responses, rows)
    logits = torch.randn(rows, 6, requires_grad=True)
    weights = [torch.randn(response, 6) for response in responses]

    pieces = logits.split(sizes)
    split_loss = sum((pieces[p] * w).sum() for p, w in zip(pieces_of_rows, weights))
    (split_grad,) = torch.autograd.grad(split_loss, logits)

    slices = [logits[a:b] if b > a else logits[0:0] for a, b in miles_windows(totals, responses)]
    for piece, sliced in zip(pieces_of_rows, slices):
        assert torch.equal(pieces[piece], sliced)
    slice_loss = sum((s * w).sum() for s, w in zip(slices, weights))
    (slice_grad,) = torch.autograd.grad(slice_loss, logits)
    assert torch.equal(split_grad, slice_grad)


def test_windows_outside_the_logits_fall_back() -> None:
    assert split_layout([4], [4], 4) is None
    assert split_layout([4, 4], [3, 3], 6) is None
