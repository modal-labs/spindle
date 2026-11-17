"""Opt-in via ``FUSED_WEIGHTED_SQUARED_RELU=1``; Miles does not pass Megatron's
``use_fused_weighted_squared_relu`` through to the model provider."""

from __future__ import annotations

import os

ENV_FLAG = "FUSED_WEIGHTED_SQUARED_RELU"


def install(auto_bridge: type) -> None:
    if os.environ.get(ENV_FLAG) != "1":
        return
    original = auto_bridge.to_megatron_provider

    def to_megatron_provider(bridge: object, *args: object, **kwargs: object) -> object:
        provider = original(bridge, *args, **kwargs)
        provider.use_fused_weighted_squared_relu = True
        return provider

    auto_bridge.to_megatron_provider = to_megatron_provider
    print("spindle_fused_squared_relu action=installed", flush=True)
