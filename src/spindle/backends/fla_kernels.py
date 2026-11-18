"""Pin FLA's unused NB launch argument so new packed lengths reuse one compiled kernel."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol

FLA_KERNELS = {
    "fla.modules.conv.triton.ops": (
        "causal_conv1d_fwd_kernel",
        "causal_conv1d_bwd_kernel",
    ),
    "fla.modules.l2norm": ("l2norm_fwd_kernel", "l2norm_bwd_kernel"),
    "fla.modules.fused_norm_gate": (
        "layer_norm_gated_fwd_kernel",
        "layer_norm_gated_bwd_kernel",
    ),
}


class _Kernel(Protocol):
    def __getitem__(self, grid: object) -> Callable[..., object]: ...


class _PinnedBlockCount:
    def __init__(self, kernel: _Kernel) -> None:
        self.kernel = kernel

    def __getitem__(self, grid: object) -> Callable[..., object]:
        launch = self.kernel[grid]
        return lambda *args, **kwargs: launch(*args, **{**kwargs, "NB": 1})


def pin_fla_block_counts() -> None:
    for module_name, kernel_names in FLA_KERNELS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in kernel_names:
            kernel = getattr(module, name, None)
            # fla-core 0.6 keeps these module paths but removed NB from the conv kernels.
            if kernel is not None and "NB" in getattr(kernel, "arg_names", ()):
                setattr(module, name, _PinnedBlockCount(kernel))
