"""Spindle actor using Miles v0.2.0's native adapter state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Callable

if TYPE_CHECKING:
    from miles.utils.args.runtime import TrainerConfig


from argparse import Namespace
from dataclasses import replace
from functools import wraps
from pathlib import Path

from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor
from miles.backends.megatron_utils.lora.checkpoint import LayerWiseSlotMembers
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    save_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    agree_on_failure_context,
)
from miles.backends.training_utils.loss.hub import tinker_losses

from spindle.backends.score_centering import REPLAY_LOSSES

from .actor import (
    SpindleMilesTrainRayActor as LegacyActor,
    _mark_actor_initialized,
    _use_distributed_timeout_for_gloo,
)
from .native_checkpoint import validate_adapter
from .volumes import commit_if_on_checkpoint_volume, sync_checkpoint_volume


class WeightsOnlyMembers(LayerWiseSlotMembers):
    def export_adapter_state(self, slot: int) -> object:
        state = super().export_adapter_state(slot)
        return replace(state, exp_avgs=[], exp_avg_sqs=[], has_optimizer_state=False)


class SpindleMilesTrainRayActor(MultiLoRATrainRayActor):
    torch_profile_start = LegacyActor.torch_profile_start
    torch_profile_stop = LegacyActor.torch_profile_stop

    def forward_backward(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self._profiled("forward_backward", *args, **kwargs)

    def forward_only(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self._profiled("forward_only", *args, **kwargs)

    def optim_step(
        self, *args: object, **kwargs: object
    ) -> dict[int, dict[str, float]]:
        return self._profiled("optim_step", *args, **kwargs)

    def _profiled(self, operation: str, *args: object, **kwargs: object) -> object:
        result = getattr(super(), operation)(*args, **kwargs)
        LegacyActor._log_peak_memory(operation)
        return result

    def save_slot(
        self,
        slot: int,
        root: str,
        name: str,
        tenant_digest: str,
        parametrization_fields: dict,
    ) -> object:
        result = super().save_slot(
            slot, root, name, tenant_digest, parametrization_fields
        )
        commit_if_on_checkpoint_volume(root)
        return result

    def export_slot(
        self,
        slot: int,
        root: str,
        name: str,
        tenant_digest: str,
        parametrization_fields: dict,
    ) -> object:
        result = super().export_slot(
            slot, root, name, tenant_digest, parametrization_fields
        )
        commit_if_on_checkpoint_volume(root)
        return result

    def init(self, args: TrainerConfig, role: str, **kwargs: object) -> int | None:
        _use_distributed_timeout_for_gloo(args.backend.distributed_timeout_minutes)
        result = MultiLoRATrainRayActor.init(self, args, role, **kwargs)
        _mark_actor_initialized()
        return result

    def load_slot(
        self,
        slot: int,
        rank: int,
        alpha: float,
        source: dict[str, Any] | None = None,
    ) -> dict[str, object] | None:
        if source is not None:
            sync_checkpoint_volume("reload")
            with agree_on_failure_context():
                validate_adapter(
                    Path(source["path"]),
                    checkpoint=True,
                    expected_shapes=self.adapter_slots.global_shapes(rank),
                )
        return MultiLoRATrainRayActor.load_slot(self, slot, rank, alpha, source=source)

    def save_slot_weights(
        self, slot: int, root: str, name: str, parametrization_fields: dict[str, object]
    ) -> object:
        result = save_adapter_dir(
            Path(root),
            name,
            WeightsOnlyMembers(self.adapter_slots, self.slot_optimizers),
            slot,
            self.adapter_slots.parametrization(**parametrization_fields),
            tenant_digest="spindle",
        )
        commit_if_on_checkpoint_volume(root)
        return result


def _with_vocabulary(function: Callable[..., object]) -> Callable[..., object]:
    @wraps(function)
    def call(args: TrainerConfig, *positional: object, **keywords: object) -> object:
        values = dict(args)
        values["vocab_size"] = args.backend.vocab_size
        return function(Namespace(**values), *positional, **keywords)

    return call


tinker_losses._target_logprobs = _with_vocabulary(tinker_losses._target_logprobs)
for _loss in REPLAY_LOSSES:
    tinker_losses.TINKER_LOSS_FUNCTIONS[_loss] = _with_vocabulary(
        tinker_losses.TINKER_LOSS_FUNCTIONS[_loss]
    )
