import torch


def add_packed_lora(hook, inputs, output, state, *, packed_input=False):
    if output.numel() == 0:
        return
    output = output.view(-1, output.shape[-1])
    topk_ids = state["topk_ids"]
    indices = state["src2dst"].reshape(-1).long()
    valid = (topk_ids.reshape(-1) >= 0) & (indices >= 0) & (indices < output.shape[0])
    indices = indices.clamp(0, output.shape[0] - 1)
    unpacked = output.index_select(0, indices).view(*topk_ids.shape, output.shape[-1])
    original = unpacked.clone()
    if packed_input:
        inputs = inputs.view(-1, inputs.shape[-1]).index_select(0, indices)
    hook(inputs, unpacked, torch.ones_like(state["topk_weights"]), topk_ids)
    delta = (unpacked - original).view(-1, output.shape[-1])
    delta.masked_fill_(~valid.unsqueeze(-1), 0)
    output.index_add_(0, indices, delta)
