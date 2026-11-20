"""Score candidate labels with one vocabulary-parallel softmax."""

import torch
import torch.distributed as dist


class CandidateLogProbs(torch.autograd.Function):
    """Log probabilities of K label columns from vocabulary-parallel logits [T, V/TP].

    Consumes ``logits`` in place, like Megatron's vocab-parallel cross entropy.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        world = (
            dist.get_world_size(group)
            if group is not None and dist.is_initialized()
            else 1
        )
        start = (dist.get_rank(group) if world > 1 else 0) * logits.size(-1)
        logits_max = logits.max(dim=-1, keepdim=True).values
        if world > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits.sub_(logits_max)
        outside = (targets < start) | (targets >= start + logits.size(-1))
        local = (targets - start).masked_fill(outside, 0)
        picked = logits.gather(-1, local).masked_fill(outside, 0.0)
        softmax = logits.exp_()
        sum_exp = softmax.sum(dim=-1, keepdim=True)
        if world > 1:
            dist.all_reduce(picked, group=group)
            dist.all_reduce(sum_exp, group=group)
        softmax.div_(sum_exp)
        ctx.save_for_backward(softmax, local, outside)
        return picked - sum_exp.log()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (grad,) = grad_outputs
        softmax, local, outside = ctx.saved_tensors
        grad_logits = softmax.mul_(-grad.sum(dim=-1, keepdim=True))
        grad_logits.scatter_add_(-1, local, grad.masked_fill(outside, 0.0))
        return grad_logits, None, None
