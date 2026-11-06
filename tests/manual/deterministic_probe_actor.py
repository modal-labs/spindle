"""Validation-only actor: fingerprint every adapter tensor and gradient per rank."""

import hashlib
import json

import torch
from miles.backends.megatron_utils.lora.optimizer import adapter_slot_parameters

from spindle.backends.miles_runtime.actor import SpindleMilesTrainRayActor


def fingerprint(value):
    digest = hashlib.sha256()
    count = 0

    def visit(item):
        nonlocal count
        if isinstance(item, torch.Tensor):
            item = item.detach().contiguous().cpu()
            if item.is_floating_point() and not torch.isfinite(item).all():
                raise ValueError("Nonfinite validation tensor")
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += 1
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())

    visit(value)
    return {"sha256": digest.hexdigest(), "tensors": count}


class DeterminismProbeActor(SpindleMilesTrainRayActor):
    def init(self, *args, **kwargs):
        result = super().init(*args, **kwargs)
        from spindle.backends.miles_runtime import batching

        original = batching.pack_slot_microbatches
        self._last_schedule = {"total": 0, "mixed": 0}

        def record(slot_batches, lengths, budget, **kwargs):
            schedule = original(slot_batches, lengths, budget, **kwargs)
            slots = {
                index: slot
                for slot, batches in slot_batches.items()
                for batch in batches
                for index in batch
            }
            self._last_schedule = {
                "total": len(schedule),
                "mixed": sum(
                    len({slots[index] for index in batch}) > 1 for batch in schedule
                ),
            }
            return schedule

        batching.pack_slot_microbatches = record
        return result

    def determinism_schedule(self):
        return {"rank": torch.distributed.get_rank(), **self._last_schedule}

    def determinism_snapshot(self, slot):
        parameters = adapter_slot_parameters(self.model, slot)
        gradients = [
            getattr(parameter, "main_grad", None)
            if getattr(parameter, "main_grad", None) is not None
            else parameter.grad
            for parameter in parameters
        ]
        inner = self.slot_optimizers[slot]._inner
        return {
            "rank": torch.distributed.get_rank(),
            "weights": fingerprint(parameters),
            "gradients": fingerprint(gradients),
            "optimizer": fingerprint(inner.state_dict()),
        }
