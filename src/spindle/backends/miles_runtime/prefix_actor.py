"""Opt-in packed-prefix execution around native Miles losses and optimizer."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from argparse import Namespace

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import unwrap_model
from miles.backends.megatron_utils import model as execution
from miles.backends.training_utils.prefix_tree.megatron import (
    validate_megatron_tree_config,
)
from miles.backends.training_utils.prefix_tree.packed import prepare_packed_tree
from miles.backends.training_utils.prefix_tree.reduction import OrderedRowGather
from spindle.backends.miles_runtime.actor_v020 import SpindleMilesTrainRayActor
from spindle.patching import same_signature

from miles.utils.args.runtime import TrainerConfig

from .prefix_checks import check_forward_parity


class SharedPrefixTrainRayActor(SpindleMilesTrainRayActor):
    def init(self, args: TrainerConfig, role: str, **kwargs: object) -> int | None:
        backend = args.backend
        if args.megatron_to_hf_mode != "raw" or args.lora_type != "canonical_lora":
            raise ValueError("Shared prefixes require native canonical LoRA")
        if (
            backend.context_parallel_size != 1
            or backend.pipeline_model_parallel_size != 1
        ):
            raise ValueError("Shared prefixes require CP=PP=1")
        if backend.expert_tensor_parallel_size != 1 or args.multi_lora_n_adapters != 1:
            raise ValueError("Shared prefixes require ETP=1 and one adapter slot")
        if (
            args.use_rollout_routing_replay
            or args.enable_mtp_training
            or args.lora_dropout
        ):
            raise ValueError(
                "Shared prefixes require zero dropout and no replay or MTP"
            )
        if not backend.deterministic_mode:
            raise ValueError("Shared prefixes require deterministic mode")
        if not args.prefix_tree_attention:
            raise ValueError("Shared prefixes require native tree attention")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        torch.utils.deterministic.fill_uninitialized_memory = False
        result = super().init(args, role, **kwargs)
        if not torch.are_deterministic_algorithms_enabled():
            raise ValueError("Shared prefixes require native deterministic algorithms")
        self.prefix_enabled = True
        self.prefix_rows = []
        roots = unwrap_model(self.model)
        self.prefix_layout_type = roots[0].config.prefix_tree_layout_type
        for root in roots:
            validate_megatron_tree_config(root.config)
            for module in root.modules():
                config = getattr(module, "config", None)
                if config is not None and hasattr(config, "moe_router_fusion"):
                    config.moe_router_fusion = False
            root.register_forward_pre_hook(self._prepare_prefix, with_kwargs=True)
            root.register_forward_hook(self._restore_prefix, with_kwargs=True)
        native_packed = execution.get_packed_seq_params

        @same_signature(native_packed)
        def packed(batch: dict[str, torch.Tensor], args: Namespace) -> PackedSeqParams:
            value = native_packed(batch, args)
            if value is None:
                raise ValueError("Shared-prefix RL requires THD packed sequences")
            value._prefix_source_batch = batch
            return value

        execution.get_packed_seq_params = packed
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.distributed.get_rank() == 0:
            print(
                "PREFIX_RUNTIME_POLICY "
                + json.dumps(
                    dict(
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        cpu_threads=torch.get_num_threads(),
                        fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
                    )
                ),
                flush=True,
                file=sys.stderr,
            )
        return result

    def _prepare_prefix(
        self, module: torch.nn.Module, args: tuple, kwargs: dict
    ) -> tuple:
        native = kwargs.get("packed_seq_params")
        if native is None:
            raise ValueError("Prefix training requires packed sequence metadata")
        batch = getattr(native, "_prefix_source_batch", None)
        if hasattr(native, "_prefix_source_batch"):
            del native._prefix_source_batch
        if batch is None:
            raise ValueError("Missing original RL batch at the shared-prefix boundary")
        if not self.prefix_enabled:
            return args, kwargs
        start = time.perf_counter()
        sequences = [value.tolist() for value in batch["unconcat_tokens"]]
        slots = batch.get("adapter_slots")
        slots = [0] * len(sequences) if slots is None else [int(slot) for slot in slots]
        n_adapters = len(batch["adapter_token_counts"])
        transport = prepare_packed_tree(
            sequences,
            slots,
            padded_rows=kwargs["input_ids"].numel(),
            alignment=self.args.backend.tensor_model_parallel_size,
            num_adapters=n_adapters,
            share_prefix=True,
        )
        plan = transport.plan
        device = kwargs["input_ids"].device
        packed = copy.copy(native)
        packed.cu_seqlens_q = torch.tensor(
            plan.segment_offsets, device=device, dtype=torch.int32
        )
        packed.cu_seqlens_kv = packed.cu_seqlens_q
        for field in ("cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
            if hasattr(packed, field):
                setattr(packed, field, None)
        packed.cu_seqlens_host = plan.segment_offsets
        maximum = max(
            b - a
            for a, b in zip(
                plan.segment_offsets[:-1], plan.segment_offsets[1:], strict=True
            )
        )
        packed.max_seqlen_q = packed.max_seqlen_kv = maximum
        layout = self.prefix_layout_type.from_plan(plan, device=device)
        layout.prepare_packed(module, packed)
        packed._prefix_restore = OrderedRowGather.from_indices(
            transport.restore_rows, len(plan.tokens), device=device
        )
        packed._prefix_compact_rows = len(plan.tokens)
        kwargs = dict(kwargs)
        kwargs.update(
            input_ids=torch.tensor([plan.tokens], device=device, dtype=torch.long),
            position_ids=torch.tensor(
                [plan.positions], device=device, dtype=torch.long
            ),
            packed_seq_params=packed,
            loss_mask=None,
        )
        self.adapter_slots.route_tokens(transport.adapter_counts)
        row = dict(
            original_rows=len(transport.restore_rows),
            compact_rows=len(plan.tokens),
            samples=len(sequences),
            segments=len(plan.segment_parents),
            planning_s=time.perf_counter() - start,
        )
        self.prefix_rows.append(row)
        if torch.distributed.get_rank() == 0:
            print("SHARED_PREFIX_BATCH " + json.dumps(row), flush=True, file=sys.stderr)
        return args, kwargs

    def _restore_prefix(
        self, module: torch.nn.Module, args: tuple, kwargs: dict, output: torch.Tensor
    ) -> torch.Tensor:
        packed = kwargs.get("packed_seq_params")
        restore = getattr(packed, "_prefix_restore", None)
        if restore is None:
            return output
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 3
            or output.shape[0] != 1
        ):
            raise ValueError(
                "Unexpected vocabulary-logit layout at the packed RL loss boundary"
            )
        if output.shape[1] != packed._prefix_compact_rows:
            raise ValueError("Compact logits have the wrong row inventory")
        return restore.gather(output[0]).unsqueeze(0)

    def forward_backward(self, *args: object, **kwargs: object) -> dict:
        # Keep this gate mandatory while gradient/optimizer equivalence remains
        # experimental. Both arms use the same installed numerical policy.
        self.prefix_enabled = False
        try:
            native = super().forward_only(*args, **kwargs)
        finally:
            self.prefix_enabled = True
        shared = super().forward_only(*args, **kwargs)
        report = check_forward_parity(native, shared)
        if torch.distributed.get_rank() == 0:
            print(
                "SHARED_PREFIX_PREFLIGHT " + json.dumps(report),
                flush=True,
                file=sys.stderr,
            )
        self.prefix_rows = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = super().forward_backward(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        original = sum(row["original_rows"] for row in self.prefix_rows)
        compact = sum(row["compact_rows"] for row in self.prefix_rows)
        result.setdefault("metrics", {}).update(
            prefix_elapsed_s=elapsed,
            prefix_expanded_rows_per_s=original / elapsed,
            prefix_original_rows=original,
            prefix_compact_rows=compact,
            prefix_saved_fraction=1 - compact / original if original else 0.0,
            prefix_microbatches=len(self.prefix_rows),
        )
        if not original:
            raise ValueError(
                "Training completed without invoking shared-prefix execution"
            )
        if torch.distributed.get_rank() == 0:
            print(
                "SHARED_PREFIX_TRAIN "
                + json.dumps(
                    {
                        k: v
                        for k, v in result["metrics"].items()
                        if k.startswith("prefix_")
                    }
                ),
                flush=True,
                file=sys.stderr,
            )
        return result
