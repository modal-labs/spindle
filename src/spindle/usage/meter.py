from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from .rows import COUNTERS, ROW_NAMESPACE, UsageRow, add_counters
from .store import DirectorySink

log = logging.getLogger(__name__)

WINDOW_S = 60


class UsageMeter:
    def __init__(
        self,
        sink: DirectorySink,
        *,
        container_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.sink = sink
        self.container_id = (
            container_id or os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
        )
        self.clock = clock
        self._counts: dict[tuple, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(
        self,
        *,
        app_name: str,
        model_id: str,
        session_id: str | None = None,
        training_model_id: str | None = None,
        **counters: float,
    ) -> None:
        unknown = counters.keys() - set(COUNTERS)
        if unknown:
            raise ValueError(f"unknown usage counters {sorted(unknown)}")
        if not any(value > 0 for value in counters.values()):
            return
        window = math.floor(self.clock() / WINDOW_S) * WINDOW_S
        key = (window, app_name, model_id, session_id, training_model_id)
        with self._lock:
            add_counters(self._counts.setdefault(key, {}), counters)

    def flush(self, *, include_open: bool = False) -> int:
        cutoff = math.inf if include_open else self.clock() - WINDOW_S
        with self._lock:
            ready = {key: v for key, v in self._counts.items() if key[0] <= cutoff}
            for key in ready:
                del self._counts[key]
        if not ready:
            return 0
        rows = [self._row(key, counters) for key, counters in ready.items()]
        try:
            self.sink.write(rows)
        except Exception:
            with self._lock:
                for key, counters in ready.items():
                    add_counters(self._counts.setdefault(key, {}), counters)
            raise
        return len(rows)

    def start(self) -> UsageMeter:
        def loop() -> None:
            while not self._stop.wait(WINDOW_S):
                try:
                    self.flush()
                except Exception:  # noqa: BLE001 - usage must not stop training
                    log.warning("Usage flush failed; retrying next interval")

        self._thread = threading.Thread(target=loop, name="usage-flush", daemon=True)
        self._thread.start()
        # Runs before Python joins busy non-daemon threads, which plain atexit waits on.
        threading._register_atexit(self.close)
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Long enough for a slow commit, so its rows aren't requeued after the final flush.
            self._thread.join(timeout=30)
            self._thread = None
        try:
            self.flush(include_open=True)
        except Exception:  # noqa: BLE001 - usage must not stop shutdown
            log.warning("Final usage flush failed")

    def _row(self, key: tuple, counters: Mapping[str, float]) -> UsageRow:
        window, app_name, model_id, session_id, training_model_id = key
        # The same container and minute always get the same id, so a retry is counted once.
        identity = json.dumps([self.container_id, *key])
        values: dict[str, Any] = dict(counters)
        return UsageRow(
            timestamp=window,
            model_id=model_id,
            request_uuid=str(uuid.uuid5(ROW_NAMESPACE, identity)),
            app_name=app_name,
            session_id=session_id,
            training_model_id=training_model_id,
            **values,
        )


class TrainerUsage:
    """A trainer's usage. None of these raise, so usage can't stop training."""

    def __init__(
        self,
        meter: UsageMeter,
        *,
        app_name: str,
        base_model: str,
        started_at: float,
    ) -> None:
        self.meter = meter
        self.app_name = app_name
        self.base_model = base_model
        self._add(cold_start_seconds=max(0.0, time.time() - started_at))

    def trained(self, model_id: str, payload: object) -> None:
        try:
            tokens = _input_tokens(payload)
        except Exception:  # noqa: BLE001 - usage must not stop training
            log.warning("Couldn't count the tokens of a training command")
            return
        if tokens:
            self._add(training_model_id=model_id, train_tokens=tokens)

    def busy(self, seconds: float) -> None:
        self._add(busy_seconds=seconds)

    def idle(self, seconds: float) -> None:
        self._add(idle_seconds=seconds)

    def _add(self, **values) -> None:
        self.meter.add(app_name=self.app_name, model_id=self.base_model, **values)


def meter_sample(
    meter: UsageMeter,
    *,
    app_name: str,
    base_model: str,
    session_id: str,
    stats: Mapping[str, Any],
) -> None:
    prefill = int(stats.get("prefill_tokens") or 0)
    meter.add(
        app_name=app_name,
        model_id=base_model,
        session_id=session_id,
        prompt_tokens=prefill,
        cached_prompt_tokens=min(int(stats.get("cached_prompt_tokens") or 0), prefill),
        completion_tokens=int(stats.get("generated_tokens") or 0),
    )


def _input_tokens(payload: object) -> int:
    return sum(
        len(tokens)
        if (tokens := getattr(chunk, "tokens", None)) is not None
        else getattr(chunk, "expected_tokens", None) or 0
        for example in getattr(payload, "data", None) or ()
        for chunk in example.model_input.chunks
    )
