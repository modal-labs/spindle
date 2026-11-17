from __future__ import annotations

import contextlib
import gc
import json
import os
import signal
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

import torch

PROFILE_STEP_ENV = "SPINDLE_TORCH_PROFILE_STEP"
PROFILE_DIR_ENV = "SPINDLE_TORCH_PROFILE_DIR"
PROFILE_RANKS_ENV = "SPINDLE_TORCH_PROFILE_RANKS"
PROFILE_SHAPES_ENV = "SPINDLE_TORCH_PROFILE_SHAPES"
PROFILE_MEMORY_ENV = "SPINDLE_TORCH_PROFILE_MEMORY"

_MAX_TRACKED_STEPS = 8
_WRITE_TIMEOUT_S = 15 * 60


@dataclass(frozen=True, slots=True)
class TorchProfileConfig:
    """Opt-in torch.profiler capture for one optimizer step."""

    step: int | None
    output_dir: str
    ranks: frozenset[int] | None = frozenset({0})
    """Trainer ranks that record a trace; ``None`` means every rank."""
    record_shapes: bool = False
    profile_memory: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> TorchProfileConfig:
        raw_step = env.get(PROFILE_STEP_ENV)
        step: int | None = None
        if raw_step:
            step = int(raw_step)
            if step < 0:
                raise ValueError(f"{PROFILE_STEP_ENV} must be non-negative")
        raw_ranks = env.get(PROFILE_RANKS_ENV, "0").strip()
        ranks: frozenset[int] | None
        if raw_ranks == "all":
            ranks = None
        else:
            ranks = frozenset(int(part) for part in raw_ranks.split(",") if part)
        checkpoint_root = env.get("SPINDLE_CHECKPOINT_ROOT") or "/checkpoints"
        output_dir = env.get(PROFILE_DIR_ENV) or os.path.join(
            checkpoint_root,
            "torch-profile",
            env.get("SPINDLE_DEFINITION_ID", "trainer"),
        )
        return cls(
            step=step,
            output_dir=output_dir,
            ranks=ranks,
            record_shapes=env.get(PROFILE_SHAPES_ENV) == "1",
            profile_memory=env.get(PROFILE_MEMORY_ENV) == "1",
        )

    @property
    def enabled(self) -> bool:
        return self.step is not None

    def profiles_rank(self, rank: int) -> bool:
        return self.ranks is None or rank in self.ranks


class StepPhaseTimer:
    """Always-on, cheap per-phase wall-clock timer for backend operations.

    Thread-safe: backend methods may run on different threads via
    ``asyncio.to_thread``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: dict[int, dict[str, float]] = {}
        self._last_activity_end: float | None = None

    def record(
        self,
        phase: str,
        seconds: float,
        *,
        step: int,
        model_id: str | None = None,
        **extra: object,
    ) -> None:
        with self._lock:
            accumulators = self._steps.setdefault(step, {})
            self._prune_locked()
            accumulators[phase] = accumulators.get(phase, 0.0) + seconds
            calls_key = f"{phase}_calls"
            accumulators[calls_key] = accumulators.get(calls_key, 0.0) + 1
            self._last_activity_end = time.perf_counter()
        print(
            json.dumps(
                {
                    "event": "spindle_step_timing",
                    "phase": phase,
                    "step": step,
                    "seconds": round(seconds, 6),
                    "model_id": model_id,
                    "ts": time.time(),
                    **extra,
                }
            ),
            flush=True,
        )

    def note_request(self, step: int) -> None:
        """Account for idle time since the previous backend op ended."""
        with self._lock:
            now = time.perf_counter()
            if self._last_activity_end is not None:
                gap = now - self._last_activity_end
                if gap > 0:
                    accumulators = self._steps.setdefault(step, {})
                    self._prune_locked()
                    accumulators["idle_wait"] = accumulators.get("idle_wait", 0.0) + gap
                    accumulators["idle_wait_calls"] = (
                        accumulators.get("idle_wait_calls", 0.0) + 1
                    )

    @contextlib.contextmanager
    def phase(
        self,
        name: str,
        step: int,
        model_id: str | None = None,
        **extra: object,
    ):
        self.note_request(step)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                name, time.perf_counter() - start, step=step, model_id=model_id, **extra
            )

    def metrics_for_step(self, step: int) -> dict[str, float]:
        """Return ``timing/*`` metrics for one optimizer step.

        Save/publish work for step k runs *after* ``optim_step`` k returns, so
        it is attributed to the model's current optimizer step at the time it
        runs (k + 1). Values logged under step k + 1 therefore refer to the
        checkpoint taken after step k.
        """
        with self._lock:
            accumulators = dict(self._steps.get(step, {}))
        metrics: dict[str, float] = {}
        for key, value in accumulators.items():
            if key.endswith("_calls"):
                metrics[f"timing/{key}"] = float(value)
            else:
                metrics[f"timing/{key}_s"] = float(value)
        forward_backward = accumulators.get("forward_backward")
        optim_step = accumulators.get("optim_step")
        if forward_backward is not None and optim_step is not None:
            metrics["timing/trainer_step_s"] = forward_backward + optim_step
        return metrics

    def pop_step(self, step: int) -> None:
        with self._lock:
            self._steps.pop(step, None)

    def _prune_locked(self) -> None:
        while len(self._steps) > _MAX_TRACKED_STEPS:
            self._steps.pop(min(self._steps))


class RankProfiler:
    """Rank-local torch.profiler wrapper usable in any process."""

    _writer: threading.Thread | None = None

    def __init__(
        self, *, record_shapes: bool = False, profile_memory: bool = False
    ) -> None:
        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._profile: torch.profiler.profile | None = None

    @classmethod
    def wait_for_writer(cls) -> None:
        if cls._writer is not None:
            cls._writer.join()

    def start(self) -> None:
        RankProfiler.wait_for_writer()
        self._profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=self._record_shapes,
            profile_memory=self._profile_memory,
            with_stack=False,
        )
        self._profile.__enter__()

    def stop(self, output_dir: str, name: str, *, fork: bool = True) -> dict[str, str]:
        """Stop on the thread that called ``start``; a child process writes the files unless ``fork`` is false."""
        if self._profile is None:
            raise RuntimeError("RankProfiler.stop() called without start()")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profile = self._profile
        self._profile = None
        profile.__exit__(None, None, None)
        paths = {
            "trace": os.path.join(output_dir, f"{name}.trace.json.gz"),
            "table": os.path.join(output_dir, f"{name}.key_averages.txt"),
        }
        if not fork:
            _write(profile, output_dir, name, paths)
            return paths
        # A child process keeps the GIL-bound export off the training thread.
        pid = os.fork()
        if pid == 0:
            _write_in_child(profile, output_dir, name, paths)
        RankProfiler._writer = threading.Thread(
            target=_report_when_written, args=(pid, paths), daemon=True
        )
        RankProfiler._writer.start()
        return paths


def _write(
    profile: torch.profiler.profile, output_dir: str, name: str, paths: dict[str, str]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    partial = os.path.join(output_dir, f".{name}.trace.json.gz")
    profile.export_chrome_trace(partial)
    os.replace(partial, paths["trace"])
    table = profile.key_averages().table(sort_by="cuda_time_total", row_limit=60)
    with open(paths["table"], "w", encoding="utf-8") as file:
        file.write(table)


def _write_in_child(
    profile: torch.profiler.profile, output_dir: str, name: str, paths: dict[str, str]
) -> NoReturn:
    # A collection here would copy the parent's heap and run its finalizers.
    gc.disable()
    os.nice(10)
    code = 0
    try:
        _write(profile, output_dir, name, paths)
    except BaseException:
        traceback.print_exc()
        code = 1
    os._exit(code)


def _report_when_written(pid: int, paths: dict[str, str]) -> None:
    started = time.perf_counter()
    while not (finished := os.waitpid(pid, os.WNOHANG))[0]:
        if time.perf_counter() - started > _WRITE_TIMEOUT_S:
            os.kill(pid, signal.SIGKILL)
        time.sleep(1)
    exit_code = os.waitstatus_to_exitcode(finished[1])
    print(
        json.dumps(
            {
                "event": "spindle_torch_profile",
                "state": "written" if exit_code == 0 else "error",
                "exit_code": exit_code,
                "files": list(paths.values()),
                "seconds": round(time.perf_counter() - started, 3),
            }
        ),
        flush=True,
    )
