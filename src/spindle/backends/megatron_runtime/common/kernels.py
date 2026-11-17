"""Stop FLA kernels from recompiling for every packed batch length."""

from __future__ import annotations

import importlib

# These FLA kernels take NB, a bucket of the token count, as a compile-time
# constant and autotune key that the kernel body never reads. Packed
# micro-batches have a new length on almost every step, so each one
# recompiled and re-autotuned the kernels, which cost seconds per step.
# Pinning NB keeps one compiled, tuned kernel per process.
PINNED_NB = 1
FLA_KERNELS = {
    "fla.modules.conv.triton.ops": (
        "causal_conv1d_fwd_kernel",
        "causal_conv1d_bwd_kernel",
    ),
    "fla.modules.l2norm": ("l2norm_fwd_kernel", "l2norm_bwd_kernel"),
}


class _PinnedBlockCount:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            kwargs["NB"] = PINNED_NB
            return launch(*args, **kwargs)

        return run


def pin_fla_block_counts() -> None:
    for module_name, kernel_names in FLA_KERNELS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in kernel_names:
            kernel = getattr(module, name, None)
            if kernel is not None and not isinstance(kernel, _PinnedBlockCount):
                setattr(module, name, _PinnedBlockCount(kernel))
