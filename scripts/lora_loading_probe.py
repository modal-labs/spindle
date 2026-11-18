"""Experimental instrumentation and CPU-packed uploads for a pinned SGLang image."""

# ruff: noqa: ANN401

from __future__ import annotations

import cProfile
import functools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch

from spindle.inference.lora_packing import build_upload_plan

_local = threading.local()
_original_upload: Any = None


def _root() -> Path:
    return Path(os.environ["SPINDLE_LORA_TRACE_ROOT"])


def _emit(event: dict) -> None:
    event.update(
        pid=os.getpid(), thread=threading.current_thread().name, at=time.time()
    )
    with (_root() / f"events-{os.getpid()}.jsonl").open("a") as output:
        output.write(json.dumps(event) + "\n")


def _pack(manager: Any, adapter: Any) -> None:
    plan = build_upload_plan(manager, adapter, _original_upload)
    adapter._spindle_upload_plan = plan
    _emit(
        {
            "phase": "packed_layout",
            "uid": adapter.uid,
            "copies": len(plan.sources),
            "bytes": plan.nbytes,
        }
    )


def _wrap(cls: Any, method: str, label: str, *, profile: bool = False) -> None:
    original = getattr(cls, method)

    @functools.wraps(original)
    def measured(self: Any, *args: Any, **kwargs: Any) -> Any:
        started, cpu = time.perf_counter(), time.thread_time()
        profiler = (
            cProfile.Profile()
            if profile and (_root() / "profile-on").exists()
            else None
        )
        if profiler:
            profiler.enable()
        try:
            return original(self, *args, **kwargs)
        finally:
            if profiler:
                profiler.disable()
                profiler.dump_stats(
                    str(_root() / f"{label}-{os.getpid()}-{time.time_ns()}.pstats")
                )
            _emit(
                {
                    "phase": label,
                    "wall_s": time.perf_counter() - started,
                    "thread_cpu_s": time.thread_time() - cpu,
                }
            )

    setattr(cls, method, measured)


def patch_adapter(cls: Any) -> None:
    _wrap(cls, "initialize_weights", "read_weights")
    _wrap(cls, "_normalize_weights", "normalize")


def patch_manager(cls: Any) -> None:
    original = cls.prepare_lora_adapter

    @functools.wraps(original)
    def prepare(self: Any, lora_ref: Any) -> Any:
        started, cpu = time.perf_counter(), time.thread_time()
        profiler = cProfile.Profile() if (_root() / "profile-on").exists() else None
        if profiler:
            profiler.enable()
        try:
            result = original(self, lora_ref)
            prepared = time.perf_counter()
            mode_file = os.environ.get("SPINDLE_LORA_BENCH_MODE_FILE")
            mode = (
                Path(mode_file).read_text().strip()
                if mode_file
                else lora_ref.lora_name.split("-", 1)[0]
            )
            if mode not in {"baseline", "pinned", "packed"}:
                if mode_file:
                    raise ValueError(f"Unknown adapter loading mode: {mode}")
                mode = "baseline"
            if mode == "packed":
                _pack(self, result[1])
            elif mode == "pinned":
                result[1].pin_weights_in_cpu()
            _emit(
                {
                    "phase": "prepare",
                    "mode": mode,
                    "name": lora_ref.lora_name,
                    "uid": lora_ref.lora_id,
                    "normal_prepare_s": prepared - started,
                    "pack_s": time.perf_counter() - prepared,
                    "wall_s": time.perf_counter() - started,
                    "thread_cpu_s": time.thread_time() - cpu,
                }
            )
            return result
        finally:
            if profiler:
                profiler.disable()
                profiler.dump_stats(
                    str(_root() / f"prepare-{os.getpid()}-{time.time_ns()}.pstats")
                )

    cls.prepare_lora_adapter = prepare
    _wrap(cls, "install_prepared_lora_adapter", "install")


def patch_pool(cls: Any, module: dict) -> None:
    global _original_upload
    _original_upload = cls.load_lora_weight_to_buffer
    _wrap(cls, "retain_local_expert_weights", "retain")
    original_copy = module["copy_weight_into_buffer"]

    @functools.wraps(original_copy)
    def transfer(destination: Any, source: Any) -> None:
        stats = getattr(_local, "transfers", None)
        if stats is not None and destination.device.type == "cuda":
            stats["copies"] += 1
            stats["bytes"] += source.numel() * source.element_size()
        return original_copy(destination, source)

    module["copy_weight_into_buffer"] = transfer

    @functools.wraps(_original_upload)
    def upload(
        self: Any,
        uid: Any,
        buffer_id: int,
        lora_adapter: Any,
        lora_modules: Any,
        lora_embed_tokens_module: Any,
        lora_lm_head_module: Any,
    ) -> None:
        if uid is None:
            return _original_upload(
                self,
                uid,
                buffer_id,
                lora_adapter,
                lora_modules,
                lora_embed_tokens_module,
                lora_lm_head_module,
            )
        started, cpu = time.perf_counter(), time.thread_time()
        stats = {"copies": 0, "bytes": 0}
        _local.transfers = stats
        upload_plan = getattr(lora_adapter, "_spindle_upload_plan", None)
        plan = upload_plan.resolve(self) if upload_plan is not None else None
        profiler = cProfile.Profile() if (_root() / "profile-on").exists() else None
        if profiler:
            profiler.enable()
        try:
            if plan is None:
                _original_upload(
                    self,
                    uid,
                    buffer_id,
                    lora_adapter,
                    lora_modules,
                    lora_embed_tokens_module,
                    lora_lm_head_module,
                )
            else:
                if (_root() / "verify-on").exists():
                    for destination, _ in plan:
                        destination[buffer_id].zero_()
                    _original_upload(
                        self,
                        uid,
                        buffer_id,
                        lora_adapter,
                        lora_modules,
                        lora_embed_tokens_module,
                        lora_lm_head_module,
                    )
                    torch.cuda.synchronize()
                    for destination, source in plan:
                        if not torch.equal(destination[buffer_id].cpu(), source):
                            raise AssertionError(
                                f"packed upload mismatch: {uid} {destination.shape}"
                            )
                    _emit(
                        {"phase": "packed_verified", "uid": uid, "buffers": len(plan)}
                    )
                for destination, source in plan:
                    destination[buffer_id].copy_(source, non_blocking=True)
                stats = {
                    "copies": len(plan),
                    "bytes": sum(
                        source.numel() * source.element_size() for _, source in plan
                    ),
                }
        finally:
            _local.transfers = None
            if profiler:
                profiler.disable()
                profiler.dump_stats(
                    str(_root() / f"activate-{os.getpid()}-{time.time_ns()}.pstats")
                )
            _emit(
                {
                    "phase": "activate",
                    "uid": uid,
                    "slot": buffer_id,
                    "packed": plan is not None,
                    "submit_s": time.perf_counter() - started,
                    "thread_cpu_s": time.thread_time() - cpu,
                    **stats,
                }
            )

    cls.load_lora_weight_to_buffer = upload


def patch_scheduler(cls: Any) -> None:
    original = cls._poll_lora_prepares

    @functools.wraps(original)
    def poll(self: Any) -> None:
        pending = getattr(self, "_pending_lora_prepares", None)
        if not pending:
            return original(self)
        started = time.perf_counter()
        stats = getattr(
            self, "_spindle_prepare_poll", {"calls": 0, "wall_s": 0.0, "max_s": 0.0}
        )
        result = original(self)
        elapsed = time.perf_counter() - started
        stats["calls"] += 1
        stats["wall_s"] += elapsed
        stats["max_s"] = max(stats["max_s"], elapsed)
        if not self._pending_lora_prepares:
            _emit({"phase": "readiness_poll", **stats})
            stats = {"calls": 0, "wall_s": 0.0, "max_s": 0.0}
        self._spindle_prepare_poll = stats
        return result

    cls._poll_lora_prepares = poll
