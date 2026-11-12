"""Spindle actor using Miles v0.2.0's native adapter state."""

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
from .volumes import sync_checkpoint_volume


class WeightsOnlyMembers(LayerWiseSlotMembers):
    def export_adapter_state(self, slot):
        state = super().export_adapter_state(slot)
        return replace(state, exp_avgs=[], exp_avg_sqs=[], has_optimizer_state=False)


class SpindleMilesTrainRayActor(MultiLoRATrainRayActor):
    torch_profile_start = LegacyActor.torch_profile_start
    torch_profile_stop = LegacyActor.torch_profile_stop

    def forward_backward(self, *args, **kwargs):
        return self._profiled("forward_backward", *args, **kwargs)

    def forward_only(self, *args, **kwargs):
        return self._profiled("forward_only", *args, **kwargs)

    def optim_step(self, *args, **kwargs):
        return self._profiled("optim_step", *args, **kwargs)

    def _profiled(self, operation, *args, **kwargs):
        result = getattr(super(), operation)(*args, **kwargs)
        LegacyActor._log_peak_memory(operation)
        return result

    def save_slot(self, *args, **kwargs):
        result = super().save_slot(*args, **kwargs)
        sync_checkpoint_volume("commit")
        return result

    def export_slot(self, *args, **kwargs):
        result = super().export_slot(*args, **kwargs)
        sync_checkpoint_volume("commit")
        return result

    def init(self, args, role, **kwargs):
        _use_distributed_timeout_for_gloo(args.backend.distributed_timeout_minutes)
        result = MultiLoRATrainRayActor.init(self, args, role, **kwargs)
        _mark_actor_initialized()
        return result

    def load_slot(self, slot, rank, alpha, source=None):
        if source is not None:
            sync_checkpoint_volume("reload")
            with agree_on_failure_context():
                validate_adapter(
                    Path(source["path"]),
                    checkpoint=True,
                    expected_shapes=self.adapter_slots.global_shapes(rank),
                )
        return MultiLoRATrainRayActor.load_slot(self, slot, rank, alpha, source=source)

    def save_slot_weights(self, slot, root, name, parametrization_fields):
        result = save_adapter_dir(
            Path(root),
            name,
            WeightsOnlyMembers(self.adapter_slots, self.slot_optimizers),
            slot,
            self.adapter_slots.parametrization(**parametrization_fields),
            tenant_digest="spindle",
        )
        sync_checkpoint_volume("commit")
        return result


def _with_vocabulary(function):
    @wraps(function)
    def call(args, *positional, **keywords):
        values = dict(args)
        values["vocab_size"] = args.backend.vocab_size
        return function(Namespace(**values), *positional, **keywords)

    return call


tinker_losses._target_logprobs = _with_vocabulary(tinker_losses._target_logprobs)
for _loss in REPLAY_LOSSES:
    tinker_losses.TINKER_LOSS_FUNCTIONS[_loss] = _with_vocabulary(
        tinker_losses.TINKER_LOSS_FUNCTIONS[_loss]
    )
