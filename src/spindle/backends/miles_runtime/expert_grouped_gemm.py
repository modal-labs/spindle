"""Run frozen MoE expert weights through one grouped GEMM per projection.

Enabled by ``EXPERT_GROUPED_GEMM=1`` in a recipe's ``trainer_env``.

Transformer Engine's ``GroupedLinear`` issues one cuBLAS GEMM per expert on
Hopper, so a 128-expert layer launches hundreds of kernels per projection and
the GPU waits on the host between them. With LoRA the expert weights never
train, so each projection can instead be a single ``torch._grouped_mm`` over
the experts' weights stacked once into one tensor.
"""

from __future__ import annotations

import os

import torch

ENV_FLAG = "EXPERT_GROUPED_GEMM"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def install(grouped_linear_cls=None) -> bool:
    if not enabled():
        return False
    if grouped_linear_cls is None:
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        grouped_linear_cls = TEGroupedLinear
    original = grouped_linear_cls.forward
    if getattr(original, "__spindle_grouped_gemm__", False):
        return True

    def forward(self, x, m_splits):
        if not _eligible(self, x):
            return original(self, x, m_splits)
        counts = torch.as_tensor(m_splits, dtype=torch.int32)
        if counts.device != x.device:
            counts = counts.pin_memory().to(x.device, non_blocking=True)
        offsets = counts.cumsum(0, dtype=torch.int32)
        return torch._grouped_mm(x, stacked_weight(self).transpose(1, 2), offsets), None

    forward.__spindle_grouped_gemm__ = True
    grouped_linear_cls.forward = forward
    print(f"spindle_expert_grouped_gemm action=installed class={grouped_linear_cls.__name__}", flush=True)
    return True


def _eligible(module, x: torch.Tensor) -> bool:
    config = module.config
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.size(0) > 0
        and not module.use_bias
        and not config.fp8
        and not getattr(config, "fp4", None)
        and not any(weight.requires_grad for weight in _weights(module))
    )


def _weights(module) -> list[torch.Tensor]:
    return [getattr(module, f"weight{i}") for i in range(module.num_gemms)]


def stacked_weight(module) -> torch.Tensor:
    """``[experts, out, in]`` weights; the per-expert parameters become views into it."""
    stacked = module.__dict__.get("_spindle_stacked_weight")
    if stacked is None:
        weights = _weights(module)
        stacked = torch.stack([weight.detach() for weight in weights])
        for index, weight in enumerate(weights):
            weight.data = stacked[index]
        module.__dict__["_spindle_stacked_weight"] = stacked
    return stacked
