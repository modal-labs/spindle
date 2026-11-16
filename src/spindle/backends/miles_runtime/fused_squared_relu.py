"""Set Megatron's ``use_fused_weighted_squared_relu`` on the model provider.

Enabled by ``FUSED_WEIGHTED_SQUARED_RELU=1`` in a recipe's ``trainer_env``;
Miles does not pass this setting through.
"""

from __future__ import annotations

import os

ENV_FLAG = "FUSED_WEIGHTED_SQUARED_RELU"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def install(auto_bridge: type) -> bool:
    if not enabled():
        return False
    original = auto_bridge.to_megatron_provider
    if getattr(original, "__spindle_fused_squared_relu__", False):
        return True

    def to_megatron_provider(bridge: object, *args: object, **kwargs: object) -> object:
        provider = original(bridge, *args, **kwargs)
        provider.use_fused_weighted_squared_relu = True
        return provider

    to_megatron_provider.__spindle_fused_squared_relu__ = True
    auto_bridge.to_megatron_provider = to_megatron_provider
    print("spindle_fused_squared_relu action=installed", flush=True)
    return True
