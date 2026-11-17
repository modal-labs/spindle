"""Run frozen BF16 MoE expert weights through one ``torch._grouped_mm`` per projection.

Enabled by ``EXPERT_GROUPED_GEMM=1`` in a recipe's ``trainer_env``.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import torch

ENV_FLAG = "EXPERT_GROUPED_GEMM"


class _GroupedLinear(Protocol):
    num_gemms: int
    use_bias: bool
    config: Any
    weight0: torch.Tensor
    __dict__: dict[str, Any]


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def install(grouped_linear_cls: type | None = None) -> bool:
    if not enabled():
        return False
    if grouped_linear_cls is None:
        # Megatron is absent where the tests import this module.
        from megatron.core.extensions.transformer_engine import TEGroupedLinear

        grouped_linear_cls = TEGroupedLinear
    original = grouped_linear_cls.forward
    if getattr(original, "__spindle_grouped_gemm__", False):
        return True

    def forward(self: _GroupedLinear, x: torch.Tensor, m_splits: object) -> tuple:
        if not _eligible(self, x) or (
            isinstance(m_splits, list) and sum(m_splits) != x.size(0)
        ):
            return original(self, x, m_splits)
        counts = torch.as_tensor(m_splits, dtype=torch.int32)
        if counts.device != x.device:
            counts = counts.pin_memory().to(x.device, non_blocking=True)
        offsets = counts.cumsum(0, dtype=torch.int32)
        weight = stacked_weight(self).transpose(1, 2)
        return torch._grouped_mm(x, weight, offsets), None

    forward.__spindle_grouped_gemm__ = True
    grouped_linear_cls.forward = forward
    _log("installed", cls=grouped_linear_cls.__name__)
    return True


def _eligible(module: _GroupedLinear, x: torch.Tensor) -> bool:
    config = module.config
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.size(0) > 0
        and not module.use_bias
        and not config.fp8
        and not getattr(config, "fp4", None)
        and getattr(module, "te_quant_params", None) is None
        and not getattr(module, "single_grouped_weight", False)
        and not module.weight0.requires_grad
    )


def stacked_weight(module: _GroupedLinear) -> torch.Tensor:
    """``[experts, out, in]`` weights; the per-expert parameters become views into it."""
    stacked = module.__dict__.get("_spindle_stacked_weight")
    if stacked is None or module.weight0.data_ptr() != stacked.data_ptr():
        weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
        stacked = torch.stack([weight.detach() for weight in weights])
        for index, weight in enumerate(weights):
            weight.data = stacked[index]
        module.__dict__["_spindle_stacked_weight"] = stacked
    return stacked


def _log(action: str, **fields: object) -> None:
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"spindle_expert_grouped_gemm action={action} {rendered}".rstrip(), flush=True
    )
