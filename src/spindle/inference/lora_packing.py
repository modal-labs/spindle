"""Prepare immutable CPU upload buffers before a LoRA enters the scheduler registry."""

# ruff: noqa: ANN401

from __future__ import annotations

import copy
import functools
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from spindle.patching import same_signature

BufferKey = tuple[str, str, int | None]
LAYER_GROUPS = ("A_buffer", "B_buffer")
EMBEDDING_GROUPS = (
    "embedding_A_buffer",
    "embedding_B_buffer",
    "lm_head_A_buffer",
    "lm_head_B_buffer",
    "new_embeddings_buffer",
)


def buffers(pool: Any) -> Iterator[tuple[BufferKey, torch.Tensor]]:
    for group in LAYER_GROUPS:
        for key, layers in getattr(pool, group).items():
            for layer, tensor in enumerate(layers):
                yield (group, key, layer), tensor
    for group in EMBEDDING_GROUPS:
        for key, tensor in getattr(pool, group).items():
            yield (group, key, None), tensor


@dataclass(frozen=True)
class UploadPlan:
    pool: weakref.ReferenceType
    sources: tuple[tuple[BufferKey, torch.Tensor], ...]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for _, t in self.sources)

    def resolve(self, pool: Any) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if self.pool() is not pool:
            return None
        current = dict(buffers(pool))
        if current.keys() != {key for key, _ in self.sources}:
            return None
        result = []
        for key, source in self.sources:
            destination = current[key]
            if (
                destination.shape[1:] != source.shape
                or destination.dtype != source.dtype
            ):
                return None
            result.append((destination, source))
        return result

    def copy_to(self, pool: Any, slot: int) -> bool:
        transfers = self.resolve(pool)
        if transfers is None:
            return False
        for destination, source in transfers:
            destination[slot].copy_(source, non_blocking=True)
        return True


def build_upload_plan(
    manager: Any, adapter: Any, load: Callable[..., None]
) -> UploadPlan:
    pool = manager.memory_pool
    shadow = copy.copy(pool)
    shadow.pin_memory_available = False

    def allocate(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (1, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )

    with torch.cuda.device(manager.device):
        for group in LAYER_GROUPS:
            setattr(
                shadow,
                group,
                {
                    key: [allocate(tensor) for tensor in layers]
                    for key, layers in getattr(pool, group).items()
                },
            )
        for group in EMBEDDING_GROUPS:
            setattr(
                shadow,
                group,
                {key: allocate(tensor) for key, tensor in getattr(pool, group).items()},
            )
        load(
            shadow,
            adapter.uid,
            0,
            adapter,
            manager.lora_modules,
            manager.embed_tokens_module,
            manager.lm_head_module,
        )
    return UploadPlan(
        weakref.ref(pool), tuple((key, tensor[0]) for key, tensor in buffers(shadow))
    )


def drain_completed_uploads(loader: Any, cpu_group: Any) -> None:
    pending = loader.lora_to_overlap_load_event
    if not pending:
        return
    events = sorted(
        pending.items(), key=lambda item: (item[0] is not None, item[0] or "")
    )
    ready = torch.tensor(
        [event.query() for _, event in events], dtype=torch.int32, device="cpu"
    )
    if dist.is_initialized():
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=cpu_group)
    reservations = getattr(loader, "_spindle_lora_reservations", set())
    loader._spindle_lora_reservations = reservations
    wanted = getattr(loader, "_spindle_wanted_loras", None)
    for (uid, event), complete in zip(events, ready.tolist(), strict=True):
        if complete:
            loader.device_module.current_stream().wait_event(event)
            del pending[uid]
            if wanted is None or uid in wanted:
                reservations.add(uid)


def prepare_admission(loader: Any, wanted: set[str | None]) -> None:
    loader._spindle_wanted_loras = wanted
    reservations = getattr(loader, "_spindle_lora_reservations", set())
    reservations.intersection_update(wanted)
    reservations.intersection_update(loader.lora_manager.memory_pool.uid_to_buffer_id)


def install_readiness() -> None:
    # SGLang is installed only in sampler images.
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader

    original = LoRAOverlapLoader._drain_completed_overlap_loads
    if getattr(original, "_spindle_tp_consensus", False):
        return

    @same_signature(original)
    @functools.wraps(original)
    def drain(self: Any) -> None:
        drain_completed_uploads(self, get_tp_group().cpu_group)

    drain._spindle_tp_consensus = True
    LoRAOverlapLoader._drain_completed_overlap_loads = drain
    original_start = LoRAOverlapLoader._try_start_overlap_load
    original_try = LoRAOverlapLoader.try_overlap_load_lora

    @same_signature(original_start)
    @functools.wraps(original_start)
    def start(self: Any, lora_id: Any, running_loras: set) -> bool:
        reserved = getattr(self, "_spindle_lora_reservations", set())
        return original_start(self, lora_id, running_loras | reserved)

    @same_signature(original_try)
    @functools.wraps(original_try)
    def try_load(self: Any, lora_id: Any, running_loras: set) -> bool:
        loaded = original_try(self, lora_id, running_loras)
        if loaded:
            getattr(self, "_spindle_lora_reservations", set()).discard(lora_id)
        return loaded

    LoRAOverlapLoader._try_start_overlap_load = start
    LoRAOverlapLoader.try_overlap_load_lora = try_load


def install_admission() -> None:
    # SGLang is installed only in sampler images.
    from sglang.srt.managers.scheduler import Scheduler

    original = Scheduler.get_next_batch_to_run
    if getattr(original, "_spindle_lora_admission", False):
        return

    @same_signature(original)
    @functools.wraps(original)
    def next_batch(self: Any, running_batch: Any, last_batch: Any) -> Any:
        loader = getattr(self, "lora_overlap_loader", None)
        if loader is not None:
            requests = list(self.waiting_queue)
            for batch in (running_batch, last_batch):
                if batch is not None:
                    requests.extend(batch.reqs)
            if self.chunked_req is not None:
                requests.append(self.chunked_req)
            prepare_admission(
                loader, {req.lora_id for req in requests if not req.finished()}
            )
        return original(self, running_batch, last_batch)

    next_batch._spindle_lora_admission = True
    Scheduler.get_next_batch_to_run = next_batch
    original_can_schedule = Scheduler.can_schedule_lora_req

    @same_signature(original_can_schedule)
    @functools.wraps(original_can_schedule)
    def can_schedule(self: Any, req: Any, running_loras: set) -> bool:
        allowed = original_can_schedule(self, req, running_loras)
        loader = getattr(self, "lora_overlap_loader", None)
        if loader is not None:
            getattr(loader, "_spindle_lora_reservations", set()).discard(req.lora_id)
        return allowed

    Scheduler.can_schedule_lora_req = can_schedule


def install() -> None:
    # SGLang is installed only in sampler images.
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.lora.mem_pool import LoRAMemoryPool

    install_readiness()
    install_admission()
    original_prepare = LoRAManager.prepare_lora_adapter
    if getattr(original_prepare, "_spindle_cpu_packed", False):
        return
    original_upload = LoRAMemoryPool.load_lora_weight_to_buffer

    @same_signature(original_prepare)
    @functools.wraps(original_prepare)
    def prepare(self: Any, lora_ref: Any) -> Any:
        prepared = original_prepare(self, lora_ref)
        if getattr(self, "memory_pool", None) is not None:
            prepared[1]._spindle_upload_plan = build_upload_plan(
                self, prepared[1], original_upload
            )
        return prepared

    @same_signature(original_upload)
    @functools.wraps(original_upload)
    def upload(
        self: Any,
        uid: Any,
        buffer_id: int,
        lora_adapter: Any,
        lora_modules: Any,
        lora_embed_tokens_module: Any,
        lora_lm_head_module: Any,
    ) -> None:
        plan = getattr(lora_adapter, "_spindle_upload_plan", None)
        if uid is not None and plan is not None and plan.copy_to(self, buffer_id):
            return
        original_upload(
            self,
            uid,
            buffer_id,
            lora_adapter,
            lora_modules,
            lora_embed_tokens_module,
            lora_lm_head_module,
        )

    prepare._spindle_cpu_packed = True
    LoRAManager.prepare_lora_adapter = prepare
    LoRAMemoryPool.load_lora_weight_to_buffer = upload
