from __future__ import annotations

import torch
from megatron.bridge.peft.multi_lora_layers import clear_adapter_slot
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import (
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
    _fork_rng,
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)


def _seeded_tracker_state(reference: object, seed: int) -> object:
    if torch.is_tensor(reference):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.manual_seed(seed)
            return torch.cuda.get_rng_state()
    clone_state = getattr(reference, "clone_state", None)
    if callable(clone_state):
        state = clone_state()
        state.manual_seed(seed)
        return state
    raise TypeError(f"unsupported Megatron RNG tracker state: {type(reference)!r}")


def reset_adapter_slot_with_seed(model: object, slot: int, job_seed: int) -> None:
    base_seed = int(job_seed) + (
        100 * parallel_state.get_pipeline_model_parallel_rank()
    )
    tracker = get_cuda_rng_tracker()
    with _fork_rng():
        if getattr(tracker, "is_inference_rng_tracker", False):
            torch.cuda.manual_seed(base_seed)
        else:
            states = dict(tracker.get_states())
            seeds = {
                _MODEL_PARALLEL_RNG_TRACKER_NAME: (
                    base_seed + 2718 + parallel_state.get_tensor_model_parallel_rank()
                ),
                get_expert_parallel_rng_tracker_name(): (
                    base_seed
                    + 1024
                    + 100 * parallel_state.get_expert_model_parallel_rank()
                    + parallel_state.get_expert_tensor_parallel_rank()
                ),
            }
            missing = set(seeds).difference(states)
            if missing:
                raise RuntimeError(
                    f"Megatron RNG tracker is missing states: {sorted(missing)}"
                )
            for name, seed in seeds.items():
                states[name] = _seeded_tracker_state(states[name], seed)
            tracker.set_states(states)
        clear_adapter_slot(model, slot)
