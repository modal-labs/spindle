"""Pieces both Miles trainer actors use. Importing this module patches nothing."""

from __future__ import annotations

from datetime import timedelta
from types import ModuleType

import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout
from megatron.core import parallel_state
from miles.ray import train_actor
from miles.utils import distributed_utils
from miles.utils.replay_base import all_replay_managers

from spindle.patching import same_signature
from spindle.startup_timing import timeline

from . import out_of_memory
from .profiling import RankProfiler, TorchProfileConfig
from .qwen3_vl_cp import install_qwen3_vl_cp_position_ids
from .replay import install_replay_hooks
from .replay_data import install_distribution_sharding
from .score_centering import install_losses


def install_replay_patches(
    *,
    megatron_model: ModuleType,
    lora_model: ModuleType,
    logit_processors: ModuleType,
    tinker_losses: ModuleType,
    math_utils: ModuleType,
    fill_replay_data: object,
    routing_replay_manager: object,
    train_data_conversion: ModuleType,
    qwen3_vl_model: ModuleType,
    miles_qwen3_vl: ModuleType,
) -> None:
    """Install the Miles patches every image needs, given that image's modules."""
    install_replay_hooks(
        megatron_model=megatron_model,
        lora_model=lora_model,
        logit_processors=logit_processors,
        tinker_losses=tinker_losses,
        math_utils=math_utils,
        fill_replay_data=fill_replay_data,
        manager=routing_replay_manager,
    )
    install_losses(tinker_losses, logit_processors)
    install_distribution_sharding(train_data_conversion)
    install_qwen3_vl_cp_position_ids(
        bridge_model=qwen3_vl_model, miles_qwen3_vl=miles_qwen3_vl
    )
    out_of_memory.install(megatron_model=megatron_model, lora_model=lora_model)


def use_distributed_timeout_for_gloo(minutes: int) -> None:
    timeout = max(timedelta(minutes=minutes), default_pg_timeout)

    @same_signature(train_actor.init_gloo_group)
    def init_gloo_group() -> object:
        if distributed_utils.GLOO_GROUP is None:
            distributed_utils.GLOO_GROUP = dist.new_group(
                backend="gloo", timeout=timeout
            )
        return distributed_utils.GLOO_GROUP

    train_actor.init_gloo_group = init_gloo_group


def mark_actor_initialized() -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        timeline("trainer-rank0").mark("actor_model_loaded")


def log_peak_memory(operation: str) -> None:
    gib = 1024**3
    print(
        f"spindle_memory op={operation} "
        f"allocated_gb={torch.cuda.memory_allocated() / gib:.2f} "
        f"max_allocated_gb={torch.cuda.max_memory_allocated() / gib:.2f} "
        f"reserved_gb={torch.cuda.memory_reserved() / gib:.2f} "
        f"max_reserved_gb={torch.cuda.max_memory_reserved() / gib:.2f}",
        flush=True,
    )


class ProfiledActor:
    """Mixin placed before Miles' actor class: profiling and peak-memory logs per training call."""

    _spindle_profiler: RankProfiler | None = None
    _out_of_memory_configured = False

    def torch_profile_start(self) -> None:
        if not TorchProfileConfig.from_env().profiles_rank(dist.get_rank()):
            self._spindle_profiler = None
            return
        self._spindle_profiler = RankProfiler()
        self._spindle_profiler.start()

    def torch_profile_stop(self, output_dir: str) -> dict | None:
        profiler = self._spindle_profiler
        if profiler is None:
            return None
        self._spindle_profiler = None
        return profiler.stop(output_dir, f"rank{dist.get_rank()}")

    def forward_backward(self, *args: object, **kwargs: object) -> object:
        return self._fail_request_on_out_of_memory("forward_backward", args, kwargs)

    def forward_only(self, *args: object, **kwargs: object) -> object:
        return self._fail_request_on_out_of_memory("forward_only", args, kwargs)

    def optim_step(self, *args: object, **kwargs: object) -> object:
        return self._profiled("optim_step", *args, **kwargs)

    def _fail_request_on_out_of_memory(
        self, operation: str, args: tuple, kwargs: dict
    ) -> object:
        if not self._out_of_memory_configured:
            self._configure_out_of_memory()
        result = out_of_memory.run(
            lambda: self._profiled(operation, *args, **kwargs),
            replay_managers=all_replay_managers,
        )
        if isinstance(result, dict) and result.get(out_of_memory.OUT_OF_MEMORY):
            log_peak_memory(f"{operation}_out_of_memory")
        return result

    def _configure_out_of_memory(self) -> None:
        world_size = dist.get_world_size()
        # Other layouts and MoE routers run collectives mid-pass, where a failed rank would strand its peers.
        dense_data_parallel = (
            parallel_state.get_data_parallel_world_size() == world_size
            and not any(
                chunk.config.num_moe_experts or chunk.ddp_config.overlap_grad_reduce
                for chunk in self.model
            )
        )
        out_of_memory.configure(
            recoverable=world_size == 1 or dense_data_parallel,
            group=distributed_utils.get_gloo_group() if world_size > 1 else None,
        )
        self._out_of_memory_configured = True

    def _profiled(self, operation: str, *args: object, **kwargs: object) -> object:
        with torch.profiler.record_function(f"spindle/{operation}"):
            result = getattr(super(), operation)(*args, **kwargs)
        log_peak_memory(operation)
        return result
