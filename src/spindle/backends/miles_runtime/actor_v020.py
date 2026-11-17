"""Spindle actor using Miles v0.2.0's native adapter state."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as qwen3_vl_model
from miles.backends.megatron_utils import model as megatron_model
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor
from miles.backends.megatron_utils.lora.checkpoint import LayerWiseSlotMembers
from miles.backends.training_utils.checkpoint.adapter.adapter_dir import (
    save_adapter_dir,
)
from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
    agree_on_failure_context,
)
from miles.backends.training_utils.loss.hub import (
    logit_processors,
    math_utils,
    tinker_losses,
)
from miles.backends.training_utils.replay.data import fill_replay_data
from miles.ray.rollout import train_data_conversion
from miles.utils.replay_base import routing_replay_manager
from miles_plugins.models import qwen3_vl as miles_qwen3_vl

from spindle.backends.score_centering import REPLAY_LOSSES

from .actor_shared import (
    ProfiledActor,
    install_replay_patches,
    mark_actor_initialized,
    use_distributed_timeout_for_gloo,
)
from .native_checkpoint import validate_adapter
from .volumes import sync_checkpoint_volume

if TYPE_CHECKING:
    from miles.utils.args.runtime import TrainerConfig


class WeightsOnlyMembers(LayerWiseSlotMembers):
    def export_adapter_state(self, slot: int) -> object:
        state = super().export_adapter_state(slot)
        return replace(state, exp_avgs=[], exp_avg_sqs=[], has_optimizer_state=False)


class SpindleMilesTrainRayActor(ProfiledActor, MultiLoRATrainRayActor):
    def init(self, args: TrainerConfig, role: str, **kwargs: object) -> int | None:
        use_distributed_timeout_for_gloo(args.backend.distributed_timeout_minutes)
        result = MultiLoRATrainRayActor.init(self, args, role, **kwargs)
        mark_actor_initialized()
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
            # Miles checks keys and shapes, not that each shard on the volume is complete.
            with agree_on_failure_context():
                validate_adapter(Path(source["path"]), checkpoint=True)
        return MultiLoRATrainRayActor.load_slot(self, slot, rank, alpha, source=source)

    def save_slot_weights(
        self, slot: int, root: str, name: str, parametrization_fields: dict[str, object]
    ) -> object:
        return save_adapter_dir(
            Path(root),
            name,
            WeightsOnlyMembers(self.adapter_slots, self.slot_optimizers),
            slot,
            self.adapter_slots.parametrization(**parametrization_fields),
            tenant_digest="spindle",
        )


def _with_vocabulary(function: Callable[..., object]) -> Callable[..., object]:
    @wraps(function)
    def call(args: TrainerConfig, *positional: object, **keywords: object) -> object:
        values = dict(args)
        values["vocab_size"] = args.backend.vocab_size
        return function(Namespace(**values), *positional, **keywords)

    return call


install_replay_patches(
    megatron_model=megatron_model,
    lora_model=lora_model,
    logit_processors=logit_processors,
    tinker_losses=tinker_losses,
    math_utils=math_utils,
    fill_replay_data=fill_replay_data,
    routing_replay_manager=routing_replay_manager,
    train_data_conversion=train_data_conversion,
    qwen3_vl_model=qwen3_vl_model,
    miles_qwen3_vl=miles_qwen3_vl,
)
tinker_losses._target_logprobs = _with_vocabulary(tinker_losses._target_logprobs)
for _loss in REPLAY_LOSSES:
    tinker_losses.TINKER_LOSS_FUNCTIONS[_loss] = _with_vocabulary(
        tinker_losses.TINKER_LOSS_FUNCTIONS[_loss]
    )
