from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot
from megatron.core.tensor_parallel.random import _fork_rng


def clear_adapter_slot_preserving_rng(model, slot: int) -> None:
    with _fork_rng():
        clear_adapter_slot(model, slot)


def iter_named_multi_lora_modules(model) -> Iterator[tuple[str, Any]]:
    for index, chunk in enumerate(model):
        for name, module in chunk.named_modules():
            if module.__class__.__name__ == "MultiLoRALinear":
                yield f"chunk{index}.{name}", module


def iter_adapter_named_params_for_slot(
    model,
    slot: int,
) -> Iterator[tuple[str, Any]]:
    for prefix, module in iter_named_multi_lora_modules(model):
        for name, parameter in module.adapters[slot].named_parameters():
            yield f"{prefix}.adapter.{name}", parameter


def iter_adapter_params_for_slots(
    model,
    slots: Sequence[int],
) -> Iterator[Any]:
    for _, module in iter_named_multi_lora_modules(model):
        for slot in slots:
            yield from module.adapters[slot].parameters()


def zero_adapter_grads_for_slots(model, slots: Sequence[int]) -> None:
    for parameter in iter_adapter_params_for_slots(model, slots):
        main = getattr(parameter, "main_grad", None)
        if main is not None:
            main.zero_()
        parameter.grad = None


def snapshot_adapter_grads_for_slots(model, slots: Sequence[int]) -> list[tuple]:
    snapshots = []
    for parameter in iter_adapter_params_for_slots(model, slots):
        main = getattr(parameter, "main_grad", None)
        snapshots.append(
            (
                parameter,
                main.clone() if main is not None else None,
                parameter.grad.clone() if parameter.grad is not None else None,
            )
        )
    return snapshots


def restore_adapter_grads(snapshots: list[tuple]) -> None:
    for parameter, main, grad in snapshots:
        if main is not None:
            parameter.main_grad.copy_(main)
        parameter.grad = grad
