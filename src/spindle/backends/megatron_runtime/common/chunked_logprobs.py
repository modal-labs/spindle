"""Target log-probs from the LM head without materializing all logits at once.

Qwen3.5 has a 248K-token vocabulary, so full FP32 logits for a 64K-token
micro-batch take 60 GiB, and cross entropy keeps a second copy for backward.
This computes the LM head and log-softmax a few thousand tokens at a time and
recomputes each chunk's logits in backward, so memory stays flat in sequence
length. It needs the whole vocabulary on each rank (tensor parallel size 1).
"""

from __future__ import annotations

import torch


def defer_output_layer(model_chunks) -> dict:
    """Make the output layer return its input; the loss applies the weight."""
    deferred: dict = {}
    for chunk in model_chunks:
        for name, layer in chunk.named_modules():
            if not name.endswith("output_layer"):
                continue

            def forward_impl(*, input, weight, bias, **_):
                if bias is not None:
                    raise ValueError("chunked log-probs do not support an output bias")
                deferred["weight"] = weight
                return input

            layer._forward_impl = forward_impl
    return deferred


def _chunk_logits(hidden, weight, fp32, temperature):
    logits = hidden.float() @ weight.t() if fp32 else (hidden @ weight.t()).float()
    return logits / temperature if temperature != 1.0 else logits


class _ChunkedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk, fp32, temperature):
        valid = labels != -100
        targets = labels.masked_fill(~valid, 0)
        matrix = weight.float() if fp32 else weight
        logprobs = torch.empty(hidden.shape[0], device=hidden.device)
        lse = torch.empty_like(logprobs)
        for start in range(0, hidden.shape[0], chunk):
            stop = start + chunk
            logits = _chunk_logits(hidden[start:stop], matrix, fp32, temperature)
            lse[start:stop] = torch.logsumexp(logits, dim=-1)
            target = logits.gather(1, targets[start:stop, None]).squeeze(1)
            logprobs[start:stop] = target - lse[start:stop]
            del logits
        ctx.save_for_backward(hidden, weight, targets, valid, lse)
        ctx.chunk, ctx.fp32, ctx.temperature = chunk, fp32, temperature
        return logprobs.masked_fill(~valid, 0)

    @staticmethod
    def backward(ctx, grad):
        hidden, weight, targets, valid, lse = ctx.saved_tensors
        chunk, fp32, temperature = ctx.chunk, ctx.fp32, ctx.temperature
        matrix = weight.float() if fp32 else weight
        matrix_fp32 = matrix.float()
        grad = (grad * valid).float()
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        for start in range(0, hidden.shape[0], chunk):
            stop = start + chunk
            rows = hidden[start:stop]
            logits = _chunk_logits(rows, matrix, fp32, temperature)
            # d logprob / d logits = onehot(target) - softmax, scaled by grad.
            probs = logits.sub_(lse[start:stop, None]).exp_().neg_()
            probs.scatter_add_(
                1,
                targets[start:stop, None],
                torch.ones_like(probs[:, :1]),
            )
            probs.mul_(grad[start:stop, None] / temperature)
            grad_hidden[start:stop] = (probs @ matrix_fp32).to(hidden.dtype)
            grad_weight.addmm_(probs.t(), rows.float())
            del logits, probs
        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None


def chunked_logprobs(hidden, weight, labels, *, chunk, fp32, temperature=1.0):
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1).to(hidden.device)
    return _ChunkedLogprobs.apply(hidden, weight, labels, chunk, fp32, temperature)


def materialize_logits(hidden, weight, *, fp32):
    hidden = hidden.reshape(-1, hidden.shape[-1])
    return hidden.float() @ weight.float().t() if fp32 else (hidden @ weight.t()).float()
