from __future__ import annotations

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as qwen3_vl_model
from megatron.core import dist_checkpointing
from miles.backends.megatron_utils import model as megatron_model
from miles.backends.megatron_utils.lora import checkpoint
from miles.backends.megatron_utils.lora import model as lora_model
from miles.backends.megatron_utils.lora.actor import MultiLoRATrainRayActor
from miles.backends.training_utils.weight_update import snapshot_publisher
from miles.utils.replay_base import routing_replay_manager
from miles_plugins.models import qwen3_vl as miles_qwen3_vl

try:
    from miles.backends.training_utils.checkpoint import io as checkpoint_io
    from miles.backends.training_utils.loss.hub import (
        logit_processors,
        math_utils,
        tinker_losses,
    )
    from miles.backends.training_utils.replay.data import fill_replay_data

    tinker_cp = None
except ImportError:
    # Miles v0.1.1: older module paths, and its Tinker losses need tinker_cp to
    # gather log probabilities across context-parallel ranks (v0.1.2 does it itself).
    from miles.backends.training_utils import checkpoint_io
    from miles.backends.training_utils.loss_hub import (
        logit_processors,
        math_utils,
        tinker_losses,
    )
    from miles.backends.training_utils.replay_data import fill_replay_data

    from . import tinker_cp

from . import allgather_cp_lora, expert_lora_compat, host_syncs
from .actor_shared import (
    ProfiledActor,
    install_replay_patches,
    mark_actor_initialized as _mark_actor_initialized,
    use_distributed_timeout_for_gloo as _use_distributed_timeout_for_gloo,
)
from .volumes import install_cross_node_checkpoint_publish, sync_checkpoint_volume

try:
    from miles.ray.rollout import train_data_conversion
except ImportError as error:
    raise RuntimeError(
        "Miles exact-support replay requires miles.ray.rollout.train_data_conversion"
    ) from error


if tinker_cp is not None:
    tinker_cp.install()

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

host_syncs.install()
allgather_cp_lora.install(megatron_model)


def _publish_checkpoints_across_nodes() -> None:
    install_cross_node_checkpoint_publish(checkpoint_io, checkpoint, snapshot_publisher)


_publish_checkpoints_across_nodes()


def _sync_checkpoint_volume(action: str) -> None:
    sync_checkpoint_volume(action)


class SpindleMilesTrainRayActor(ProfiledActor, MultiLoRATrainRayActor):
    """Upstream multi-LoRA actor with Qwen MTP and weights-only save support."""

    def init(self, args, role, **kwargs):
        _use_distributed_timeout_for_gloo(args.distributed_timeout_minutes)
        # Miles's LoRA builder inherits checkpoint MTP heads without honoring
        # enable_mtp_training. Qwen3.5 then injects an auxiliary backward loss
        # even for a client datum whose weights are all zero.
        if args.enable_mtp_training:
            result = super().init(args, role, **kwargs)
            _mark_actor_initialized()
            return result
        original = AutoBridge.to_megatron_provider

        def provider_without_mtp(bridge, *provider_args, **provider_kwargs):
            provider = original(bridge, *provider_args, **provider_kwargs)
            provider.mtp_num_layers = None
            provider.mtp_hybrid_override_pattern = None
            return provider

        AutoBridge.to_megatron_provider = provider_without_mtp
        try:
            result = super().init(args, role, **kwargs)
        finally:
            AutoBridge.to_megatron_provider = original
        _mark_actor_initialized()
        return result

    def load_slot(
        self,
        slot: int,
        rank: int,
        alpha: float,
        ckpt_path: str | None = None,
        load_optimizer: bool = True,
    ):
        if ckpt_path:
            _sync_checkpoint_volume("reload")
        result = super().load_slot(
            slot, rank, alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer
        )
        if ckpt_path and expert_lora_compat.enabled():
            expert_lora_compat.restore_factory_weights(
                self.model,
                self.slot_optimizers[slot],
                ckpt_path,
                load_optimizer=load_optimizer,
            )
        return result

    def save_slot(self, *args, **kwargs):
        result = super().save_slot(*args, **kwargs)
        _sync_checkpoint_volume("commit")
        return result

    def export_slot(self, *args, **kwargs):
        result = super().export_slot(*args, **kwargs)
        _sync_checkpoint_volume("commit")
        return result

    def export_slot_peft(
        self,
        *,
        slot: int,
        path: str,
        rank: int,
        alpha: float,
        base_model: str,
        target_modules: tuple[str, ...],
        lora_dropout: float,
    ) -> None:
        with torch.profiler.record_function("spindle/export_slot_peft"):
            return super().export_slot_peft(
                slot=slot,
                path=path,
                rank=rank,
                alpha=alpha,
                base_model=base_model,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
            )

    def save_slot_weights(self, slot: int, path: str) -> None:
        self._save_slot_weights(slot, path)
        _sync_checkpoint_volume("commit")

    def _save_slot_weights(self, slot: int, path: str) -> None:
        weights = checkpoint._slot_weights_sharded_state_dict(self.model, slot)
        sharded = {checkpoint._WEIGHTS_KEY: weights}
        checkpoint._canonicalize_slot_keys(sharded, slot)
        checkpoint_io.write_checkpoint_dir(
            path, lambda temporary: dist_checkpointing.save(sharded, str(temporary))
        )
