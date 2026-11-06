"""Use the pinned Bridge's topology-aware seeded slot initialization."""

from contextlib import contextmanager
from functools import wraps


@contextmanager
def seeded_slot_initialization(seed):
    if seed is None:
        yield
        return
    from megatron.bridge.peft import multi_lora_layers

    original = multi_lora_layers.init_adapter_slot

    @wraps(original)
    def initialize(model, idx, rank, alpha, **kwargs):
        kwargs.setdefault("seed", seed)
        return original(model, idx, rank, alpha, **kwargs)

    multi_lora_layers.init_adapter_slot = initialize
    try:
        yield
    finally:
        multi_lora_layers.init_adapter_slot = original
