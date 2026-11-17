"""State fingerprints for the live LoRA seed check."""

import hashlib
import os
from typing import TYPE_CHECKING

import torch
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

if TYPE_CHECKING or os.environ.get("SPINDLE_MILES_API") != "0.2":
    from spindle.backends.miles_runtime.actor import SpindleMilesTrainRayActor
else:
    from spindle.backends.miles_runtime.actor_v020 import SpindleMilesTrainRayActor


class SeedProbeActor(SpindleMilesTrainRayActor):
    def seed_probe(self, slot: int) -> dict[str, str]:
        weights = hashlib.sha256()
        replicated = 0
        modules = 0
        for chunk in self.model:
            for module in chunk.modules():
                if isinstance(module, MultiLoRALinear):
                    modules += 1
                    replicated += int(getattr(module, "replicate_adapter", False))
                    for parameter in module.adapters[slot].parameters():
                        weights.update(
                            parameter.detach()
                            .contiguous()
                            .view(torch.uint8)
                            .cpu()
                            .numpy()
                            .tobytes()
                        )
        rng = hashlib.sha256()
        rng.update(torch.get_rng_state().numpy().tobytes())
        rng.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
        for name, state in sorted(get_cuda_rng_tracker().get_states().items()):
            rng.update(name.encode())
            if not torch.is_tensor(state):
                state = state.get_state()
            rng.update(state.cpu().numpy().tobytes())
        return {
            "weights": weights.hexdigest(),
            "rng": rng.hexdigest(),
            "replicated_modules": str(replicated),
            "adapter_modules": str(modules),
        }
