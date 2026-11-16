"""MoE token permute whose backward is a scatter-add.

With expert adapters, Miles keeps Megatron's unfused permute, which copies each
token to its experts' rows with ``index_select``. PyTorch's backward for that
is ``index_add_``, whose BF16 kernel is several times slower than the
``scatter_add_`` Megatron already uses to combine expert outputs. This gives
the permute the same backward. The forward is Megatron's code for the
unfused, no-padding case, so the patch checks the pinned source first.
"""

from __future__ import annotations

import hashlib
import inspect

import torch

PERMUTE_SOURCE_SHA256 = "d37d15db0b41d189ec517f345ca84ff9c62f979e7d879f7fd963cf86de0f220d"


class PermuteRows(torch.autograd.Function):
    """``tokens.index_select(0, index)`` whose backward scatter-adds rows."""

    @staticmethod
    def forward(ctx, tokens, index):
        ctx.save_for_backward(index)
        ctx.rows = tokens.size(0)
        return tokens.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        out = grad.new_zeros((ctx.rows, grad.size(1)))
        if torch.are_deterministic_algorithms_enabled():
            return out.index_add_(0, index, grad), None
        out.scatter_add_(0, index.unsqueeze(1).expand(-1, grad.size(1)), grad.contiguous())
        return out, None


def install(moe_utils=None, token_dispatcher=None) -> bool:
    if moe_utils is None:
        from megatron.core.transformer.moe import moe_utils, token_dispatcher
    original = moe_utils.permute
    if getattr(original, "__spindle_permute__", False):
        return True
    digest = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if digest != PERMUTE_SOURCE_SHA256:
        _log("skipped", reason=f"source sha256 {digest[:12]} differs")
        return False

    def permute(
        tokens,
        routing_map,
        probs=None,
        num_out_tokens=None,
        fused=False,
        drop_and_pad=False,
        tokens_per_expert=None,
        align_size=0,
    ):
        if fused or drop_and_pad or num_out_tokens is None:
            return original(
                tokens, routing_map, probs, num_out_tokens, fused,
                drop_and_pad, tokens_per_expert, align_size,
            )
        num_tokens = tokens.shape[0]
        flat_sorted = (
            routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)
        )[:num_out_tokens]
        sorted_indices = flat_sorted % num_tokens
        permuted_probs = None
        if probs is not None:
            permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]
        permuted_input = PermuteRows.apply(tokens, sorted_indices)
        return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert

    permute.__spindle_permute__ = True
    moe_utils.permute = permute
    if token_dispatcher is not None and getattr(token_dispatcher, "permute", None) is original:
        token_dispatcher.permute = permute
    _log("installed")
    return True


def _log(action: str, **fields) -> None:
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"spindle_moe_permute action={action} {rendered}".rstrip(), flush=True)
