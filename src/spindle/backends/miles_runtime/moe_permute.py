"""Megatron's unfused MoE permute with a scatter-add backward.

The forward is copied from the pinned Megatron source, so ``install`` checks
its hash first.
"""

from __future__ import annotations

import hashlib
import inspect

import torch

PERMUTE_SOURCE_SHA256 = (
    "d37d15db0b41d189ec517f345ca84ff9c62f979e7d879f7fd963cf86de0f220d"
)


class PermuteRows(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tokens: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.rows = tokens.size(0)
        return tokens.index_select(0, index)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad,) = grad_outputs
        (index,) = ctx.saved_tensors
        out = grad.new_zeros((ctx.rows, grad.size(1)))
        if torch.are_deterministic_algorithms_enabled():
            return out.index_add_(0, index, grad), None
        out.scatter_add_(
            0, index.unsqueeze(1).expand(-1, grad.size(1)), grad.contiguous()
        )
        return out, None


def install() -> bool:
    try:
        from megatron.core.transformer.moe import moe_utils, token_dispatcher
    except ImportError as error:
        _log("skipped", reason=type(error).__name__)
        return False
    original = moe_utils.permute
    if getattr(original, "__spindle_permute__", False):
        return True
    if hasattr(original, "__wrapped__"):
        _log("skipped", reason="permute is already wrapped")
        return False
    digest = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if digest != PERMUTE_SOURCE_SHA256:
        _log("skipped", reason=f"source sha256 {digest[:12]} differs")
        return False

    def permute(
        tokens: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
        fused: bool = False,
        drop_and_pad: bool = False,
        tokens_per_expert: torch.Tensor | None = None,
        align_size: int = 0,
    ) -> tuple:
        if fused or drop_and_pad or num_out_tokens is None:
            return original(
                tokens,
                routing_map,
                probs,
                num_out_tokens,
                fused,
                drop_and_pad,
                tokens_per_expert,
                align_size,
            )
        num_tokens = tokens.shape[0]
        flat_sorted = (
            routing_map.bool()
            .T.contiguous()
            .reshape(-1)
            .argsort(descending=True, stable=True)
        )[:num_out_tokens]
        sorted_indices = flat_sorted % num_tokens
        permuted_probs = None
        if probs is not None:
            permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]
        permuted_input = PermuteRows.apply(tokens, sorted_indices)
        return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert

    permute.__spindle_permute__ = True
    moe_utils.permute = permute
    if getattr(token_dispatcher, "permute", None) is original:
        token_dispatcher.permute = permute
    _log("installed")
    return True


def _log(action: str, **fields: object) -> None:
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"spindle_moe_permute action={action} {rendered}".rstrip(), flush=True)
