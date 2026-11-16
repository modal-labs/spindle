"""Opt-in observation of stock SDK results without retaining training payloads."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from concurrent.futures import Future
from types import ModuleType
from pathlib import Path

import tinker
from tinker.lib.public_interfaces.api_future import APIFuture

import spindle.client as spindle_client

from .store import json_safe

logger = logging.getLogger(__name__)
_active = ContextVar("spindle_sdk_observer", default=False)


class ObservedFuture(APIFuture[object]):
    def __init__(
        self, future: APIFuture[object], observe: Callable[[object], None]
    ) -> None:
        self._wrapped = future
        self._observe = observe
        self._lock = threading.Lock()
        self._recorded = False

    def _result(self, result: object) -> object:
        with self._lock:
            if not self._recorded:
                self._recorded = True
                try:
                    self._observe(result)
                except Exception:
                    logger.warning("SDK benchmark result could not be observed")
        return result

    def result(self, timeout: float | None = None) -> object:
        return self._result(self._wrapped.result(timeout=timeout))

    async def result_async(self, timeout: float | None = None) -> object:
        return self._result(await self._wrapped.result_async(timeout=timeout))

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)


class ClientMetrics:
    def __init__(self, directory: str | Path, run_id: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"sdk-{os.getpid()}-metrics.jsonl"
        self.run_id = run_id
        self.lock = threading.Lock()
        self.patches = []

    def emit(self, event: str, **fields: object) -> None:
        try:
            row = json_safe(
                {"event": event, "time": time.time(), "run_id": self.run_id, **fields}
            )
            with self.lock, self.path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        except Exception:
            logger.warning("SDK benchmark observation could not be saved")

    def _observe(
        self, method: str, bound: inspect.BoundArguments, started: float
    ) -> Callable[[object], None]:
        owner = bound.arguments.get("self")
        counts = {}
        data = bound.arguments.get("data")
        if isinstance(data, (list, tuple)):
            counts = {
                "sequences": len(data),
                "input_tokens": sum(d.model_input.length for d in data),
            }
        prompt = bound.arguments.get("prompt")
        if prompt is not None:
            counts["input_tokens"] = prompt.length * bound.arguments.get(
                "num_samples", 1
            )

        def observed(result: object) -> None:
            metrics = dict(getattr(result, "metrics", {}) or {})
            sequences = getattr(result, "sequences", None)
            if sequences is not None:
                counts.update(
                    output_tokens=sum(len(s.tokens) for s in sequences),
                    sequences=len(sequences),
                )
                counts["prompt_cache_hit_tokens"] = getattr(
                    result, "prompt_cache_hit_tokens", 0
                )
            self.emit(
                "sdk_result",
                operation=method.removesuffix("_async"),
                model_id=getattr(owner, "model_id", None),
                sdk={
                    "metrics": metrics,
                    "observed_latency_s": time.monotonic() - started,
                    **counts,
                },
            )

        return observed

    def _wrap(self, result: object, observe: Callable[[object], None]) -> object:
        if isinstance(result, ObservedFuture):
            return result
        if isinstance(result, APIFuture):
            return ObservedFuture(result, observe)
        if isinstance(result, Future):

            def completed(future: Future) -> None:
                if not future.cancelled() and future.exception() is None:
                    try:
                        observe(future.result())
                    except Exception:
                        logger.warning("SDK benchmark result could not be observed")

            result.add_done_callback(completed)
            return result
        try:
            observe(result)
        except Exception:
            logger.warning("SDK benchmark result could not be observed")
        return result

    def _patch(
        self, cls: type | ModuleType, name: str, *, metadata: bool = False
    ) -> None:
        original = getattr(cls, name, None)
        if original is None:
            return
        signature = inspect.signature(original)

        def arguments(
            args: tuple[object, ...], kwargs: dict[str, object]
        ) -> inspect.BoundArguments:
            bound = signature.bind(*args, **kwargs)
            if metadata:
                bound.arguments["user_metadata"] = {
                    **(bound.arguments.get("user_metadata") or {}),
                    "run_id": self.run_id,
                    "attempt_id": self.run_id,
                }
            return bound

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapped(*args: object, **kwargs: object) -> object:
                if _active.get():
                    return await original(*args, **kwargs)
                bound = arguments(args, kwargs)
                started = time.monotonic()
                token = _active.set(True)
                try:
                    result = await original(*bound.args, **bound.kwargs)
                finally:
                    _active.reset(token)
                if not metadata:
                    try:
                        return self._wrap(result, self._observe(name, bound, started))
                    except Exception:
                        logger.warning(
                            "SDK benchmark observation could not be attached"
                        )
                return result
        else:

            @functools.wraps(original)
            def wrapped(*args: object, **kwargs: object) -> object:
                if _active.get():
                    return original(*args, **kwargs)
                bound = arguments(args, kwargs)
                started = time.monotonic()
                token = _active.set(True)
                try:
                    result = original(*bound.args, **bound.kwargs)
                finally:
                    _active.reset(token)
                if not metadata:
                    try:
                        return self._wrap(result, self._observe(name, bound, started))
                    except Exception:
                        logger.warning(
                            "SDK benchmark observation could not be attached"
                        )
                return result

        self.patches.append((cls, name, original, wrapped))
        setattr(cls, name, wrapped)

    def install(self) -> ClientMetrics:
        for name in (
            "create_lora_training_client",
            "create_lora_training_client_async",
            "create_training_client_from_state",
            "create_training_client_from_state_async",
        ):
            self._patch(tinker.ServiceClient, name, metadata=True)
        for name in (
            "create_full_training_client",
            "create_full_training_client_async",
        ):
            self._patch(spindle_client, name, metadata=True)
        for operation in (
            "forward",
            "forward_backward",
            "forward_backward_custom",
            "optim_step",
            "save_state",
            "save_weights_for_sampler",
        ):
            for suffix in ("", "_async"):
                self._patch(tinker.TrainingClient, operation + suffix)
        for name in ("sample", "sample_async"):
            self._patch(tinker.SamplingClient, name)
        return self

    def close(self) -> None:
        for cls, name, original, wrapped in reversed(self.patches):
            if getattr(cls, name) is wrapped:
                setattr(cls, name, original)
        self.patches.clear()


@functools.cache
def install_from_env() -> ClientMetrics | None:
    directory = os.getenv("SPINDLE_BENCHMARK_LOG_DIR")
    run_id = os.getenv("SPINDLE_BENCHMARK_RUN_ID")
    if directory and run_id:
        return ClientMetrics(directory, run_id).install()
    return None
