import gzip
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from test_miles import FakeMilesRuntime, _backend, _datum, _spec
from tinker import AdamParams

from spindle.backends import ForwardBatch, ForwardItem
from spindle.backends.miles_runtime.profiling import (
    RankProfiler,
    StepPhaseTimer,
    TorchProfileConfig,
)


def test_torch_profile_config_disabled_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("SPINDLE_TORCH_PROFILE_STEP", raising=False)
    monkeypatch.delenv("SPINDLE_TORCH_PROFILE_DIR", raising=False)

    config = TorchProfileConfig.from_env({})

    assert not config.enabled
    assert config.step is None


def test_torch_profile_config_reads_step_and_dir() -> None:
    config = TorchProfileConfig.from_env(
        {"SPINDLE_TORCH_PROFILE_STEP": "2", "SPINDLE_TORCH_PROFILE_DIR": "/traces"}
    )

    assert config.enabled
    assert config.step == 2
    assert config.output_dir == "/traces"
    assert config.profiles_rank(0)
    assert not config.profiles_rank(3)
    assert not config.record_shapes and not config.profile_memory


def test_torch_profile_config_rank_selection() -> None:
    all_ranks = TorchProfileConfig.from_env(
        {"SPINDLE_TORCH_PROFILE_STEP": "2", "SPINDLE_TORCH_PROFILE_RANKS": "all"}
    )
    some = TorchProfileConfig.from_env(
        {"SPINDLE_TORCH_PROFILE_STEP": "2", "SPINDLE_TORCH_PROFILE_RANKS": "0,4"}
    )

    assert all_ranks.ranks is None and all_ranks.profiles_rank(7)
    assert some.ranks == frozenset({0, 4})
    assert some.profiles_rank(4) and not some.profiles_rank(1)


def test_torch_profile_config_rejects_bad_step() -> None:
    with pytest.raises(ValueError):
        TorchProfileConfig.from_env({"SPINDLE_TORCH_PROFILE_STEP": "abc"})
    with pytest.raises(ValueError):
        TorchProfileConfig.from_env({"SPINDLE_TORCH_PROFILE_STEP": "-1"})


def test_torch_profile_config_defaults_dir_to_checkpoint_volume() -> None:
    config = TorchProfileConfig.from_env(
        {
            "SPINDLE_TORCH_PROFILE_STEP": "1",
            "SPINDLE_CHECKPOINT_ROOT": "/mnt/ckpt",
            "SPINDLE_DEFINITION_ID": "def-a",
        }
    )

    assert config.output_dir == "/mnt/ckpt/torch-profile/def-a"


def test_timer_records_phases_and_metrics(capsys) -> None:
    timer = StepPhaseTimer()
    with timer.phase("forward_backward", 0, model_id="model-a"):
        pass
    with timer.phase("optim_step", 0, model_id="model-a"):
        pass

    metrics = timer.metrics_for_step(0)

    assert "timing/forward_backward_s" in metrics
    assert metrics["timing/forward_backward_calls"] == 1.0
    assert "timing/optim_step_s" in metrics
    assert metrics["timing/trainer_step_s"] == (
        metrics["timing/forward_backward_s"] + metrics["timing/optim_step_s"]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert all(line["event"] == "spindle_step_timing" for line in lines)
    assert lines[0]["phase"] == "forward_backward"
    assert lines[0]["step"] == 0
    assert lines[0]["model_id"] == "model-a"


def test_timer_records_on_exception(capsys) -> None:
    timer = StepPhaseTimer()
    with pytest.raises(RuntimeError), timer.phase("forward_backward", 3):
        raise RuntimeError("boom")

    assert "timing/forward_backward_s" in timer.metrics_for_step(3)


def test_timer_accumulates_idle_between_ops() -> None:
    timer = StepPhaseTimer()
    with timer.phase("forward_backward", 0):
        pass
    time.sleep(0.01)
    timer.note_request(0)

    metrics = timer.metrics_for_step(0)
    assert metrics["timing/idle_wait_s"] >= 0.01


def test_rank_profiler_stop_returns_before_the_files_are_written(
    tmp_path, monkeypatch
) -> None:
    gate = tmp_path / "gate"
    export = torch.profiler.profile.export_chrome_trace

    def export_after_gate(self, path):
        deadline = time.monotonic() + 10
        while not gate.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        export(self, path)

    monkeypatch.setattr(
        torch.profiler.profile, "export_chrome_trace", export_after_gate
    )
    profiler = RankProfiler()
    profiler.start()
    torch.randn(64, 64) @ torch.randn(64, 64)
    result = profiler.stop(str(tmp_path / "traces"), "rank0")

    assert not Path(result["trace"]).exists()
    gate.touch()
    RankProfiler._writer.join(timeout=60)
    assert json.loads(gzip.decompress(Path(result["trace"]).read_bytes()))
    assert Path(result["table"]).read_text()


def test_rank_profiler_without_fork_writes_before_returning(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(os, "fork", Mock(side_effect=AssertionError("forked")))
    profiler = RankProfiler()
    profiler.start()
    torch.randn(64, 64) @ torch.randn(64, 64)
    result = profiler.stop(str(tmp_path / "traces"), "rank0", fork=False)

    assert json.loads(gzip.decompress(Path(result["trace"]).read_bytes()))
    assert Path(result["table"]).read_text()


def test_trace_is_written_when_the_trainer_exits_right_after_stop(tmp_path) -> None:
    script = f"""
import torch
from spindle.backends.miles_runtime.profiling import RankProfiler

profiler = RankProfiler()
profiler.start()
torch.randn(64, 64) @ torch.randn(64, 64)
profiler.stop({str(tmp_path)!r}, "rank0")
"""
    assert subprocess.run([sys.executable, "-c", script], timeout=60).returncode == 0

    trace = tmp_path / "rank0.trace.json.gz"
    deadline = time.monotonic() + 60
    while not trace.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert json.loads(gzip.decompress(trace.read_bytes()))


def test_backend_timing_metrics_without_profiling(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SPINDLE_TORCH_PROFILE_STEP", raising=False)
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())

    backend.forward_backward(
        ForwardBatch(
            items=(ForwardItem("model-a", (_datum([1, 2, 3], 4),)),),
            loss_fn="cross_entropy",
        )
    )
    (result,) = backend.optim_step(("model-a",), AdamParams(learning_rate=2e-4))

    assert ("torch_profile_start",) not in runtime.calls
    assert result.metrics["timing/optimizer_step"] == 0.0
    assert "timing/forward_backward_s" in result.metrics
    assert "timing/optim_step_s" in result.metrics
    assert "timing/trainer_step_s" in result.metrics
    assert result.metrics["grad_norm:mean"] == 0.5


def _step(backend, model_id="model-a"):
    backend.forward_backward(
        ForwardBatch(
            items=(ForwardItem(model_id, (_datum([1, 2, 3], 4),)),),
            loss_fn="cross_entropy",
        )
    )
    (result,) = backend.optim_step((model_id,), AdamParams(learning_rate=2e-4))
    return result


def test_backend_starts_and_stops_torch_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPINDLE_TORCH_PROFILE_STEP", "1")
    monkeypatch.setenv("SPINDLE_TORCH_PROFILE_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    runtime = FakeMilesRuntime()
    backend = _backend(tmp_path, runtime)
    backend.accept_model("model-a", _spec())

    _step(backend)  # step 0: no profiling yet
    assert ("torch_profile_start",) not in runtime.calls

    _step(backend)  # step 1: profiler starts at first forward_backward
    starts = [c for c in runtime.calls if c == ("torch_profile_start",)]
    assert len(starts) == 1
    assert backend._profiling_active

    backend.capture_sampler_snapshot("model-a", "capture-a", 1)

    result = _step(backend)  # step 2: profiler stops before this fb
    stops = [c for c in runtime.calls if c[0] == "torch_profile_stop"]
    assert stops == [("torch_profile_stop", str(tmp_path / "traces"))]
    fb_index = max(i for i, c in enumerate(runtime.calls) if c[0] == "forward_backward")
    assert runtime.calls.index(stops[0]) < fb_index
    assert not backend._profiling_active

    # The capture ran while optimizer_step was 2, so it is reported on the
    # step-2 optim response.
    assert "timing/save_sampler_weights_s" in result.metrics

    _step(backend)  # step 3: profiling stays off
    assert len([c for c in runtime.calls if c == ("torch_profile_start",)]) == 1
    assert len([c for c in runtime.calls if c[0] == "torch_profile_stop"]) == 1


def test_backend_stops_torch_profile_once_across_command_and_publish_threads(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SPINDLE_TORCH_PROFILE_STEP", "1")
    monkeypatch.setenv("SPINDLE_TORCH_PROFILE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("SPINDLE_BULLETIN_ROOT", str(tmp_path / "bulletin"))
    monkeypatch.delenv("SPINDLE_BULLETIN_VOLUME", raising=False)
    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    stopping = threading.Event()
    release = threading.Event()

    class SlowStopRuntime(FakeMilesRuntime):
        def torch_profile_stop(self, output_dir, *, fork):
            stopping.set()
            release.wait(timeout=10)
            return super().torch_profile_stop(output_dir, fork=fork)

    runtime = SlowStopRuntime()
    backend = _backend(tmp_path, runtime)
    # The engine runs commands and sampler publishes on different threads.
    commands = ThreadPoolExecutor(1)
    publishes = ThreadPoolExecutor(1)
    commands.submit(backend.accept_model, "model-a", _spec()).result()
    commands.submit(_step, backend).result()
    commands.submit(_step, backend).result()  # profiled step
    commands.submit(
        backend.capture_sampler_snapshot, "model-a", "capture-a", 2
    ).result()

    publish = publishes.submit(backend.publish_sampler_snapshot, "capture-a")
    assert stopping.wait(timeout=10)
    commands.submit(_step, backend).result(timeout=10)  # next step overlaps the stop
    release.set()
    publish.result(timeout=10)

    assert [c for c in runtime.calls if c[0] == "torch_profile_stop"] == [
        ("torch_profile_stop", str(tmp_path / "traces"))
    ]
    assert not commands.submit(torch.autograd._profiler_enabled).result()


def test_trainer_deployment_env_forwards_profile_vars(monkeypatch) -> None:
    monkeypatch.delenv("SPINDLE_TRAINER_MAX_CONTAINERS", raising=False)
    monkeypatch.delenv("SPINDLE_TORCH_PROFILE_STEP", raising=False)
    monkeypatch.delenv("SPINDLE_TORCH_PROFILE_DIR", raising=False)
    monkeypatch.setenv("SPINDLE_TORCH_PROFILE_STEP", "2")
    monkeypatch.setenv("SPINDLE_TORCH_PROFILE_DIR", "/traces")

    from spindle.providers.modal.deployment import trainer_deployment_env

    env = trainer_deployment_env()
    assert env["SPINDLE_TORCH_PROFILE_STEP"] == "2"
    assert env["SPINDLE_TORCH_PROFILE_DIR"] == "/traces"
    assert "SPINDLE_APP_NAME" not in env
