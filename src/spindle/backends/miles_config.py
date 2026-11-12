from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PEFT_TARGETS = {
    "linear_qkv": ("q_proj", "k_proj", "v_proj"),
    "linear_q": ("q_proj",),
    "linear_k": ("k_proj",),
    "linear_v": ("v_proj",),
    "linear_proj": ("o_proj",),
    "linear_fc1": ("gate_proj", "up_proj"),
    "linear_fc1_gate": ("gate_proj",),
    "linear_fc1_up": ("up_proj",),
    "linear_fc2": ("down_proj",),
    "output_layer": ("lm_head",),
}
_ATTN_LEAVES = frozenset(
    {
        "linear_qkv",
        "linear_q",
        "linear_k",
        "linear_v",
        "linear_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "b_proj",
        "f_a_proj",
        "f_b_proj",
        "g_a_proj",
        "g_b_proj",
        "wq_a",
        "wkv",
        "wq_b",
        "wo_b",
    }
)
_MLP_LEAVES = frozenset(
    {
        "linear_fc1",
        "linear_fc1_gate",
        "linear_fc1_up",
        "linear_fc2",
        "gate_proj",
        "up_proj",
        "down_proj",
        "w1",
        "w2",
        "w3",
    }
)
_UNEMBED_LEAVES = frozenset({"output_layer", "lm_head"})


def _peft_names(target: str) -> tuple[str, ...]:
    if "*" in target:
        return (target.rsplit("*.", 1)[-1],)
    leaf = target.rsplit(".", 1)[-1]
    return _PEFT_TARGETS.get(leaf, (leaf,))


def lora_target_flags(target_modules: tuple[str, ...]) -> tuple[bool, bool, bool]:
    """Return ``(train_attn, train_mlp, train_unembed)`` implied by target modules."""
    leaves = {module.rsplit(".", 1)[-1] for module in target_modules}
    return (
        bool(leaves & _ATTN_LEAVES),
        bool(leaves & _MLP_LEAVES),
        bool(leaves & _UNEMBED_LEAVES),
    )


PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
NonEmptyString = Annotated[str, Field(min_length=1)]


class MilesBackendConfig(BaseModel):
    """Stable Spindle configuration translated to Miles CLI arguments at startup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hf_checkpoint: NonEmptyString
    model_type: str
    actor_num_gpus_per_node: PositiveInteger
    actor_num_nodes: PositiveInteger = 1
    megatron_to_hf_mode: Literal["bridge", "raw"] = "bridge"
    tensor_model_parallel_size: PositiveInteger = 1
    context_parallel_size: PositiveInteger = 1
    expert_model_parallel_size: PositiveInteger = 1
    expert_tensor_parallel_size: PositiveInteger = 1
    max_lora_slots: PositiveInteger = 8
    max_lora_rank: PositiveInteger = 32
    default_lora_alpha: float = Field(default=32.0, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)] = (
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "output_layer",
    )
    max_tokens_per_gpu: PositiveInteger = 8192
    dynamic_batch_size: bool = True
    align_sequences_to_parallel_layout: bool = False
    extra_args: tuple[str, ...] = ()
    cli_options: dict[str, Any] = Field(default_factory=dict)
    dangerously_allow_checkpoint_revision_mismatch: bool = Field(
        default=False, strict=True
    )

    @property
    def world_size(self) -> int:
        return self.actor_num_nodes * self.actor_num_gpus_per_node

    @property
    def data_parallel_size(self) -> int:
        return self.world_size // (
            self.tensor_model_parallel_size * self.context_parallel_size
        )

    @property
    def sequence_alignment(self) -> int:
        if not self.align_sequences_to_parallel_layout:
            return 1
        return 2 * self.context_parallel_size * self.tensor_model_parallel_size

    @property
    def peft_target_modules(self) -> tuple[str, ...]:
        targets: list[str] = []
        for target in self.target_modules:
            for name in _peft_names(target):
                if name not in targets:
                    targets.append(name)
        return tuple(targets)

    @field_validator("default_lora_alpha")
    @classmethod
    def validate_lora_alpha(cls, value: float) -> float:
        if not value.is_integer():
            raise ValueError("default_lora_alpha must be a positive integer")
        return value

    @model_validator(mode="after")
    def validate_architecture_and_topology(self) -> Self:
        if not self.model_type and not self.cli_options:
            raise ValueError(
                "model_type or explicit native architecture options are required"
            )
        model_parallel_size = (
            self.tensor_model_parallel_size * self.context_parallel_size
        )
        if self.world_size % model_parallel_size != 0:
            raise ValueError(
                "actor_num_nodes * actor_num_gpus_per_node must be a multiple of "
                "tensor_model_parallel_size * context_parallel_size"
            )
        if (
            self.actor_num_nodes > 1
            and self.actor_num_gpus_per_node % self.tensor_model_parallel_size != 0
        ):
            raise ValueError(
                "tensor_model_parallel_size must evenly divide "
                "actor_num_gpus_per_node on each node"
            )

        return self

    def miles_arguments(self) -> list[str]:
        """Arguments owned by the integration, excluding model architecture flags."""

        arguments = [
            "--train-backend",
            "megatron",
            "--hf-checkpoint",
            self.hf_checkpoint,
            "--load",
            self.hf_checkpoint,
            "--pretrained-checkpoint",
            self.hf_checkpoint,
            "--ref-load",
            self.hf_checkpoint,
            "--megatron-to-hf-mode",
            self.megatron_to_hf_mode,
            "--debug-train-only",
            "--rollout-num-gpus",
            "0",
            "--actor-num-nodes",
            str(self.actor_num_nodes),
            "--actor-num-gpus-per-node",
            str(self.actor_num_gpus_per_node),
            "--multi-lora-n-adapters",
            str(self.max_lora_slots),
            "--lora-rank",
            str(self.max_lora_rank),
            "--lora-alpha",
            str(int(self.default_lora_alpha)),
            "--lora-dropout",
            str(self.lora_dropout),
            "--target-modules",
            ",".join(self.target_modules),
            "--no-gradient-accumulation-fusion",
            "--optimizer",
            "adam",
            "--lr",
            "1e-4",
            "--tensor-model-parallel-size",
            str(self.tensor_model_parallel_size),
            "--pipeline-model-parallel-size",
            "1",
            "--context-parallel-size",
            str(self.context_parallel_size),
            "--expert-model-parallel-size",
            str(self.expert_model_parallel_size),
            "--expert-tensor-parallel-size",
            str(self.expert_tensor_parallel_size),
            *self._batching_arguments(),
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend",
            "flash",
        ]
        if self.tensor_model_parallel_size > 1:
            arguments.append("--sequence-parallel")
        return [*arguments, *self.extra_args]

    def _batching_arguments(self) -> list[str]:
        if self.dynamic_batch_size:
            return [
                "--use-dynamic-batch-size",
                "--max-tokens-per-gpu",
                str(self.max_tokens_per_gpu),
            ]
        # Attention layouts such as bshd take one sequence per micro-batch.
        return ["--micro-batch-size", "1"]


def parse_backend_config(
    value: dict[str, Any],
) -> tuple[MilesBackendConfig, Path, Path]:
    miles = dict(value.get("miles") or value)
    config = MilesBackendConfig(**miles)
    checkpoint_dir = Path(value.get("checkpoint_dir") or "/tmp/spindle")
    capture_dir = Path(value.get("capture_dir") or "/tmp/spindle-miles-captures")
    return config, checkpoint_dir, capture_dir
