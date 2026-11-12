"""Miles v0.2.0 configuration and adapter-directory API."""

from argparse import Namespace
from pathlib import Path

from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    base_model_fingerprint,
)
from miles.ray.placement_group import create_trainer_handles
from miles.ray.specs.train import ACTOR_ROLE, TrainerSpec, compute_trainer_configs
from miles.ray.train.init_request import TrainerControllerInitRequest
from miles.ray.wiring import compute_backend_capability
from miles.utils.hf_utils.config import load_hf_config
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

from .native_checkpoint import materialize_adapter

CHECKPOINT_FORMAT = "miles_adapter_state"
ACTOR_CLASS = "spindle.backends.miles_runtime.actor_v020.SpindleMilesTrainRayActor"


def configure_args(args):
    args.lora_A_init_method = "kaiming"
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True


def configure_actor():
    original = TrainerSpec.create.__func__
    if getattr(original, "_spindle_actor_spec", False):
        return

    def create(cls, config):
        spec = original(cls, config)
        if (
            spec.worker_class
            == "miles.backends.megatron_utils.lora.actor.MultiLoRATrainRayActor"
        ):
            return spec.model_copy(update={"worker_class": ACTOR_CLASS})
        return spec

    create._spindle_actor_spec = True
    TrainerSpec.create = classmethod(create)


async def initialize_trainer(args):
    configs = compute_trainer_configs(args)
    (actor,) = [config for config in configs if config.role == ACTOR_ROLE]
    handles = create_trainer_handles(
        args, trainer_configs=configs, capability=compute_backend_capability(args)
    )
    trainer = handles[actor.trainer_id]
    return trainer, TrainerControllerInitRequest.from_args(args, trainer=actor)


def replay_args(args):
    return Namespace(**(dict(args) | args.raw_megatron.base_args))


class AdapterBridge:
    def __init__(self, bridge, args):
        self.bridge = bridge
        self.args = args
        self.slots = {}
        config = load_hf_config(args.hf_checkpoint).to_dict()
        self.fingerprint = base_model_fingerprint(config)
        bridge.lora_layout = get_hf_lora_targets(config)

    def __getattr__(self, name):
        return getattr(self.bridge, name)

    def fields(self, rank, alpha):
        return dict(
            base_model=self.args.hf_checkpoint,
            lora_rank=rank,
            lora_alpha=alpha,
            lora_type=self.args.lora_type,
            experts_shared_outer_loras=self.args.experts_shared_outer_loras,
            base_model_fingerprint=self.fingerprint,
        )

    async def load_slot(
        self, slot, rank, alpha, *, ckpt_path=None, load_optimizer=True
    ):
        source = None
        if ckpt_path is not None:
            source = dict(
                path=ckpt_path,
                load_optimizer=load_optimizer,
                parametrization_fields=self.fields(rank, alpha),
            )
        result = await self.bridge.load_slot(slot, rank, alpha, source=source)
        if result is None:
            self.slots[slot] = (rank, alpha)
        return result

    async def unload_slot(self, slot):
        result = await self.bridge.unload_slot(slot)
        if result is None:
            self.slots.pop(slot)
        return result

    async def save_slot(self, slot, path, *, include_optimizer=True):
        target = Path(path)
        rank, alpha = self.slots[slot]
        fields = self.fields(rank, alpha)
        if include_optimizer:
            written = await self.bridge.save_slot(
                slot, str(target.parent), target.name, "spindle", fields
            )
        else:
            results = await self.bridge.trainer._execute_slots(
                "save_slot_weights",
                slot=slot,
                root=str(target.parent),
                name=target.name,
                parametrization_fields=fields,
            )
            written = results[0]
        materialize_adapter(written.path, target, checkpoint=True)

    async def export_slot(self, slot, rank, alpha, path):
        target = Path(path)
        written = await self.bridge.export_slot(
            slot, str(target.parent), target.name, "spindle", self.fields(rank, alpha)
        )
        materialize_adapter(written.path, target, checkpoint=False)
