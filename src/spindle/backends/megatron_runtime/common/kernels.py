"""Stop FLA's convolution kernels from recompiling for every packed batch length."""

from __future__ import annotations

# FLA passes NB, the token count in blocks of 1K, as a compile-time constant
# and autotune key that the kernel body never reads. Packed micro-batches have
# a new length on almost every step, so each one recompiled and re-autotuned
# both kernels, which cost seconds per step. Pinning NB keeps one compiled,
# tuned kernel per process.
PINNED_NB = 1


class _PinnedBlockCount:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            kwargs["NB"] = PINNED_NB
            return launch(*args, **kwargs)

        return run


def pin_fla_convolution_block_count() -> None:
    try:
        from fla.modules.conv.triton import ops
    except ImportError:
        return
    for name in ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"):
        kernel = getattr(ops, name, None)
        if kernel is not None and not isinstance(kernel, _PinnedBlockCount):
            setattr(ops, name, _PinnedBlockCount(kernel))
