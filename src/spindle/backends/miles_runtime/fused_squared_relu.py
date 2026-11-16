"""Fuse the MoE squared-ReLU with the router-probability multiply.

Enabled by ``FUSED_WEIGHTED_SQUARED_RELU=1`` in a recipe's ``trainer_env``.

Without it, each expert activation runs as separate kernels: squared ReLU,
a multiply by the router probability that promotes the activation to fp32,
and a cast back to bf16, plus the matching kernels in backward. Megatron has
a fused kernel for this (``use_fused_weighted_squared_relu``), but Miles builds
the model config from a fixed list of arguments that does not include it, so
it is set on the model provider here.
"""

from __future__ import annotations

import os

ENV_FLAG = "FUSED_WEIGHTED_SQUARED_RELU"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def install(auto_bridge=None) -> bool:
    if not enabled():
        return False
    if auto_bridge is None:
        from megatron.bridge import AutoBridge

        auto_bridge = AutoBridge
    original = auto_bridge.to_megatron_provider
    if getattr(original, "__spindle_fused_squared_relu__", False):
        return True

    def to_megatron_provider(bridge, *args, **kwargs):
        provider = original(bridge, *args, **kwargs)
        provider.use_fused_weighted_squared_relu = True
        return provider

    to_megatron_provider.__spindle_fused_squared_relu__ = True
    auto_bridge.to_megatron_provider = to_megatron_provider
    return True
