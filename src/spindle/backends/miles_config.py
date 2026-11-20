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
    # SGLang serves the Qwen3.5 linear-attention input projections as in_proj_qkvz and in_proj_ba.
    "in_proj_qkv": ("in_proj_qkvz",),
    "in_proj_z": ("in_proj_qkvz",),
    "in_proj_b": ("in_proj_ba",),
    "in_proj_a": ("in_proj_ba",),
}
GDN_INPUT_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
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
        *GDN_INPUT_PROJECTIONS,
        "out_proj",
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


def check_lora_request(
    target_modules: tuple[str, ...],
    max_lora_rank: int,
    *,
    rank: int,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> None:
    """Reject a LoRA the deployment cannot train. A requested unembedding the
    deployment does not target is allowed; the trainer keeps it frozen."""
    if not 0 < rank <= max_lora_rank:
        raise ValueError(f"LoRA rank must be between 1 and {max_lora_rank}")
    attn, mlp, unembed = lora_target_flags(target_modules)
    if (train_attn, train_mlp) != (attn, mlp) or (unembed and not train_unembed):
        raise ValueError(
            f"this deployment trains LoRA with train_attn={attn}, "
            f"train_mlp={mlp}, train_unembed={unembed}; "
            "create the model with matching settings"
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
    lora_type: Literal["lora", "canonical_lora"] = "lora"
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
    shared_prefix_training: bool = Field(default=False, strict=True)
    shared_prefix_check_forward: bool = Field(default=False, strict=True)
    align_sequences_to_parallel_layout: bool = False
    extra_args: tuple[str, ...] = ()
    cli_options: dict[str, Any] = Field(default_factory=dict)

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
        gdn_targets = [
            name
            for name in self.target_modules
            if name.rsplit(".", 1)[-1] in GDN_INPUT_PROJECTIONS
        ]
        if (
            self.lora_type == "lora"
            and gdn_targets
            and sorted(gdn_targets) != sorted(GDN_INPUT_PROJECTIONS)
        ):
            raise ValueError(
                "linear-attention LoRA trains one adapter over the fused input "
                f"projection; list all of {', '.join(GDN_INPUT_PROJECTIONS)} by name"
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

        if self.shared_prefix_check_forward and not self.shared_prefix_training:
            raise ValueError(
                "shared_prefix_check_forward requires shared_prefix_training"
            )
        if self.shared_prefix_training:
            self._validate_shared_prefix()
        return self

    def _validate_shared_prefix(self) -> None:
        if self.megatron_to_hf_mode != "raw" or self.lora_type != "canonical_lora":
            raise ValueError(
                "shared_prefix_training requires native raw canonical LoRA"
            )
        required = {
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "qkv_format": "thd",
            "deterministic_mode": True,
            "use_rollout_routing_replay": False,
            "enable_mtp_training": False,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "moe_aux_loss_coeff": [0.0],
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": False,
            "prefix_tree_attention": True,
        }
        if self.context_parallel_size != 1 or self.expert_tensor_parallel_size != 1:
            raise ValueError("shared_prefix_training requires CP=ETP=1")
        if self.max_lora_slots != 1 or self.actor_num_nodes != 1:
            raise ValueError(
                "shared_prefix_training currently requires one adapter slot on one node"
            )
        if self.lora_dropout:
            raise ValueError("shared_prefix_training requires zero LoRA dropout")
        managed_sequence_length = (
            len(self.extra_args) == 2
            and self.extra_args[0] == "--seq-length"
            and self.extra_args[1].isdecimal()
            and int(self.extra_args[1]) > 0
        )
        if self.extra_args and not managed_sequence_length:
            raise ValueError(
                "shared_prefix_training requires cli_options instead of extra_args"
            )
        for name, expected in required.items():
            if self.cli_options.get(name, expected) != expected:
                raise ValueError(f"shared_prefix_training requires {name}={expected!r}")

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
            "--lora-type",
            self.lora_type,
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
        if self.shared_prefix_training:
            arguments.extend(
                [
                    "--prefix-tree-attention",
                    "--deterministic-mode",
                ]
            )
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
