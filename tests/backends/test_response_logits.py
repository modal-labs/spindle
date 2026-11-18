import itertools

import pytest

torch = pytest.importorskip("torch")

from spindle.backends.miles_runtime.response_logits import split_layout  # noqa: E402

CASES = [
    ([5, 7, 3, 9], [4, 6, 0, 8], 4),
    ([3, 4], [2, 4], 0),
    ([0, 4], [0, 2], 1),
    ([4], [4], 0),
    ([4, 4], [3, 3], -2),
]


@pytest.mark.parametrize(("totals", "responses", "padding"), CASES)
def test_split_matches_miles_slices_or_falls_back(
    totals: list[int], responses: list[int], padding: int
) -> None:
    rows = sum(totals) + padding
    logits = torch.randn(rows, 6, requires_grad=True)
    ends = list(itertools.accumulate(totals))
    slices = [logits[e - r - 1 : e - 1] for e, r in zip(ends, responses)]

    layout = split_layout(totals, responses, rows)

    if layout is None:
        assert any(s.size(0) != r for s, r in zip(slices, responses))
        return
    sizes, pieces_of_rows = layout
    pieces = logits.split(sizes)
    for piece, sliced in zip(pieces_of_rows, slices):
        assert torch.equal(pieces[piece], sliced)
    weights = [torch.randn(r, 6) for r in responses]
    split_loss = sum((pieces[p] * w).sum() for p, w in zip(pieces_of_rows, weights))
    slice_loss = sum((s * w).sum() for s, w in zip(slices, weights))
    assert torch.equal(
        torch.autograd.grad(split_loss, logits)[0],
        torch.autograd.grad(slice_loss, logits)[0],
    )
