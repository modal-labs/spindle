"""Target log-probs from the LM head a chunk of tokens at a time, recomputed in backward."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def defer_output_layer(model_chunks: Iterable[torch.nn.Module]) -> dict[str, Any]:
    """Make the output layer return its input; the loss applies the weight."""
    deferred: dict[str, Any] = {}
    patched = 0
    for chunk in model_chunks:
        if getattr(getattr(chunk, "config", None), "use_mup", False):
            raise ValueError("chunked log-probs do not support MuP logit scaling")
        for name, layer in chunk.named_modules():
            if not name.endswith("output_layer"):
                continue

            def forward_impl(
                *,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor | None,
                **_: object,
            ) -> torch.Tensor:
                if bias is not None:
                    raise ValueError("chunked log-probs do not support an output bias")
                deferred["weight"] = weight
                return input

            layer._forward_impl = forward_impl
            patched += 1
    if not patched:
        raise ValueError("chunked log-probs found no output layer to defer")
    return deferred


def _chunk_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    fp32: bool,
    temperature: float = 1.0,
) -> torch.Tensor:
    if fp32:
        logits = hidden.float() @ weight.float().t()
    else:
        logits = (hidden.to(weight.dtype) @ weight.t()).float()
    return logits.div_(temperature) if temperature != 1.0 else logits


class _ChunkedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        chunk: int,
        fp32: bool,
        temperature: float,
    ) -> torch.Tensor:
        valid = labels != -100
        targets = labels.masked_fill(~valid, 0)
        logprobs = torch.empty(hidden.shape[0], device=hidden.device)
        lse = torch.empty_like(logprobs)
        logit_weight = weight.float() if fp32 else weight
        for start in range(0, hidden.shape[0], chunk):
            stop = start + chunk
            logits = _chunk_logits(hidden[start:stop], logit_weight, fp32, temperature)
            lse[start:stop] = torch.logsumexp(logits, dim=-1)
            target = logits.gather(1, targets[start:stop, None]).squeeze(1)
            logprobs[start:stop] = target - lse[start:stop]
            del logits
        ctx.save_for_backward(hidden, weight, targets, valid, lse)
        ctx.chunk, ctx.fp32, ctx.temperature = chunk, fp32, temperature
        return logprobs.masked_fill(~valid, 0)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        (grad,) = grad_outputs
        hidden, weight, targets, valid, lse = ctx.saved_tensors
        chunk, fp32, temperature = ctx.chunk, ctx.fp32, ctx.temperature
        weight_fp32 = weight.float()
        grad = (grad * valid).float()
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight_fp32)
        for start in range(0, hidden.shape[0], chunk):
            stop = start + chunk
            rows = hidden[start:stop]
            logits = _chunk_logits(
                rows, weight_fp32 if fp32 else weight, fp32, temperature
            )
            # d logprob / d logits = onehot(target) - softmax, scaled by grad.
            probs = logits.sub_(lse[start:stop, None]).exp_().neg_()
            probs.scatter_add_(
                1,
                targets[start:stop, None],
                torch.ones_like(probs[:, :1]),
            )
            probs.mul_(grad[start:stop, None] / temperature)
            grad_hidden[start:stop] = (probs @ weight_fp32).to(hidden.dtype)
            grad_weight.addmm_(probs.t(), rows.float())
            del logits, probs
        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None


def chunked_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk: int,
    fp32: bool,
    temperature: float = 1.0,
) -> torch.Tensor:
    hidden = hidden.reshape(-1, hidden.shape[-1])
    return _ChunkedLogprobs.apply(
        hidden, weight, labels.reshape(-1), chunk, fp32, temperature
    )


def materialize_logits(
    hidden: torch.Tensor, weight: torch.Tensor, *, fp32: bool
) -> torch.Tensor:
    return _chunk_logits(hidden.reshape(-1, hidden.shape[-1]), weight, fp32)
