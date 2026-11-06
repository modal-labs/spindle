"""Stage slot updates while another adapter uses shared model buffers."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class DetachedSlotState:
    weights: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    gradients: list[torch.Tensor] = field(default_factory=list)

    def commit(self):
        for parameter, updated in self.weights:
            parameter.data.copy_(updated)


@contextmanager
def detached_slot_state(parameters):
    saved = []
    detached = DetachedSlotState()
    try:
        for parameter in parameters:
            main_grad = getattr(parameter, "main_grad", None)
            grad = parameter.grad
            weight = parameter.data
            saved.append((parameter, main_grad, grad, weight))
            updated = weight.clone()
            detached.weights.append((parameter, updated))
            parameter.data = updated
            copied_main = None if main_grad is None else main_grad.clone()
            copied_grad = None if grad is None else grad.clone()
            detached.gradients.extend(
                value for value in (copied_main, copied_grad) if value is not None
            )
            if main_grad is not None:
                main_grad.zero_()
                parameter.main_grad = copied_main
            if grad is not None:
                grad.zero_()
            parameter.grad = copied_grad
        yield detached
    finally:
        for parameter, main_grad, grad, weight in saved:
            parameter.data = weight
            if main_grad is not None:
                parameter.main_grad = main_grad
            parameter.grad = grad


def optim_step_with_forward_backward(
    actor, *, batch_id, rollout_data_ref, adam_params_by_slot, forward_only
):
    # GPU-only Miles imports are unavailable to CPU tests importing this helper.
    from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

    parameters = [
        parameter
        for slot in sorted(adam_params_by_slot)
        for parameter in adapter_slot_parameters(actor.model, slot)
    ]
    stream = getattr(actor, "_spindle_optimizer_stream", None)
    if stream is None:
        actor._spindle_optimizer_stream = stream = torch.cuda.Stream()
    compute = torch.cuda.current_stream()
    detached = DetachedSlotState()
    try:
        with detached_slot_state(parameters) as detached:
            stream.wait_stream(compute)
            with torch.cuda.stream(stream):
                slots = sorted(adam_params_by_slot)
                for slot in slots:
                    optimizer = actor.slot_optimizers[slot]
                    optimizer.apply_adam_params(adam_params_by_slot[slot])
                    optimizer.prepare_grads()
                outcomes = {
                    slot: actor.slot_optimizers[slot].clip_and_step(
                        adam_params_by_slot[slot]["grad_clip_norm"]
                    )
                    for slot in slots
                }
                for slot in slots:
                    actor.slot_optimizers[slot].zero_grads()
        method = actor.forward_only if forward_only else actor.forward_backward
        result = method(batch_id=batch_id, rollout_data_ref=rollout_data_ref)
    finally:
        # Keep detached tensors alive and drain GPU work before lifecycle operations.
        stream.synchronize()
    detached.commit()
    for slot in slots:
        if "grad_norm" in outcomes[slot]:
            actor.slot_optimizers[slot].allgather_params()
    result["optimizer_outcomes"] = outcomes
    return result
