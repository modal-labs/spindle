import pytest
import torch

from spindle.inference.mxfp8_lora import add_packed_lora


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("masked_layout", [False, True])
def test_lora_keeps_expert_permutation_and_applies_router_weights_once(
    packed, masked_layout
):
    indices = torch.tensor([[3, 0], [5, -1]])
    topk_ids = torch.tensor([[1, 0], [1, -1]])
    weights = torch.tensor([[0.25, 0.75], [0.8, 0.0]])
    output = torch.arange(24, dtype=torch.float32).view(6, 4)
    if masked_layout:
        output = output.view(2, 3, 4)
    before = output.clone().view(6, 4)
    inputs = torch.arange(12 if packed else 4, dtype=torch.float32).view(-1, 2)
    state = {"src2dst": indices, "topk_ids": topk_ids, "topk_weights": weights}

    def add_delta(x, values, router_weights, expert_ids):
        if packed:
            delta = x.sum(-1).view(2, 2, 1)
        else:
            delta = x.sum(-1).view(2, 1, 1)
        values.add_(delta * router_weights.unsqueeze(-1))

    add_packed_lora(add_delta, inputs, output, state, packed_input=packed)
    expected = before.clone()
    for token in range(2):
        for choice in range(2):
            row = int(indices[token, choice])
            if row >= 0:
                input_row = row if packed else token
                expected[row] += inputs[input_row].sum()
    torch.testing.assert_close(output.view(6, 4), expected)
    actual_reduced = sum(
        output.view(6, 4)[int(indices[0, k])] * weights[0, k] for k in range(2)
    )
    expected_reduced = sum(
        expected[int(indices[0, k])] * weights[0, k] for k in range(2)
    )
    torch.testing.assert_close(actual_reduced, expected_reduced)
