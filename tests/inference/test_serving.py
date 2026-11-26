import contextlib
import io
import threading
import urllib.error
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from spindle.inference.serving import (
    sglang_busy,
    start_fft_sidecar,
    start_sglang,
    supervise_children,
    terminate,
    watch_health,
)


def test_full_sglang_uses_cpu_weight_staging() -> None:
    with patch("subprocess.Popen") as popen:
        start_sglang(
            "/model",
            port=8001,
            context_length=1024,
            max_loras_per_batch=1,
            max_loaded_loras=1,
            max_lora_rank=1,
            max_running_requests=64,
            enable_lora=False,
            weight_update_staging="cpu",
            weight_update_max_compile_group_gb=16,
            schedule_policy="lpm",
        )
    command = popen.call_args.args[0]
    assert "--enable-lora" not in command
    assert command[command.index("--weight-update-staging") + 1] == "cpu"
    assert command[command.index("--weight-version") + 1] == "0"
    assert command[command.index("--weight-update-max-compile-group-gb") + 1] == "16"
    assert command[command.index("--schedule-policy") + 1] == "lpm"


def test_sglang_supports_hybrid_moe_parallelism() -> None:
    with patch("subprocess.Popen") as popen:
        start_sglang(
            "/model",
            port=8001,
            context_length=65_536,
            max_loras_per_batch=1,
            max_loaded_loras=1,
            max_lora_rank=1,
            max_running_requests=64,
            tensor_parallel_size=1,
            expert_parallel_size=4,
            parallel_world_size=4,
            memory_fraction=0.9,
            enable_lora=False,
        )
    command = popen.call_args.args[0]
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--dp-size") + 1] == "4"
    assert command[command.index("--ep-size") + 1] == "4"
    assert command[command.index("--mem-fraction-static") + 1] == "0.9"
    assert "--enable-dp-attention" in command


def test_child_supervisor_terminates_sibling_and_retires_unexpected_exit() -> None:
    exited = Mock()
    exited.poll.return_value = 1
    sibling = Mock()
    sibling.poll.return_value = None
    retired = threading.Event()

    with patch("spindle.inference.serving.terminate") as terminate:
        supervisor = supervise_children(exited, sibling, on_exit=retired.set)
        supervisor.join(timeout=1)

    assert not supervisor.is_alive()
    terminate.assert_called_once_with(sibling)
    assert retired.is_set()

    stop = threading.Event()
    stop.set()
    retired.clear()
    with patch("spindle.inference.serving.terminate") as terminate:
        supervisor = supervise_children(exited, sibling, stop=stop, on_exit=retired.set)
        supervisor.join(timeout=1)
    assert not retired.is_set()
    terminate.assert_not_called()


def test_fft_sidecar_receives_store_scope() -> None:
    with patch("subprocess.Popen") as popen:
        start_fft_sidecar(
            port=8000,
            sglang_port=8001,
            bulletin_root="/bulletin",
            bulletin_volume="bulletin",
            run_id="run-a",
            pinned_version=7,
        )
    command = popen.call_args.args[0]
    assert command[command.index("--run-id") + 1] == "run-a"
    assert command[command.index("--pinned-version") + 1] == "7"


def metrics(running: float, queued: float, finished: float) -> io.BytesIO:
    return io.BytesIO(
        f"""# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{{dp_rank="0"}} {running}
sglang:num_running_reqs 0.0
sglang:num_queue_reqs{{dp_rank="0"}} {queued}
sglang:num_requests_total{{dp_rank="0",is_streaming="false"}} {finished}
""".encode()
    )


def test_sglang_counts_running_queued_and_just_finished_requests_as_busy() -> None:
    pages = [metrics(0, 0, 4), metrics(2, 0, 4), metrics(0, 0, 6), metrics(0, 3, 6)]
    busy = sglang_busy(8001)
    with patch("urllib.request.urlopen", side_effect=[*pages, metrics(0, 0, 6)]):
        assert [busy() for _ in range(5)] == [False, True, True, True, False]


def test_sglang_without_request_metrics_is_an_error_not_idle() -> None:
    with (
        patch("urllib.request.urlopen", return_value=io.BytesIO(b"other 1\n")),
        pytest.raises(ValueError, match="num_running_reqs"),
    ):
        sglang_busy(8001)()


def test_watch_health_resets_on_recovery_and_retires_after_consecutive_failures() -> (
    None
):
    sidecar = Mock(spec=["poll"])
    sidecar.poll.return_value = None
    sglang = Mock(spec=["poll"])
    sglang.poll.return_value = None

    stop = threading.Event()
    events: list[tuple[str, object]] = []
    outcomes = iter(
        [
            urllib.error.HTTPError("http://x/health", 503, "unhealthy", {}, None),
            contextlib.nullcontext(SimpleNamespace(status=200)),
            TimeoutError("timed out"),
            urllib.error.HTTPError("http://x/health", 503, "unhealthy", {}, None),
        ]
    )

    def urlopen(url: str, timeout: float) -> object:
        item = next(outcomes)
        if isinstance(item, BaseException):
            raise item
        return item

    with (
        patch("urllib.request.urlopen", side_effect=urlopen),
        patch("spindle.inference.serving._log_gpu_state"),
        patch(
            "spindle.inference.serving.terminate",
            side_effect=lambda proc: events.append(("terminate", proc)),
        ),
    ):
        thread = watch_health(
            "http://127.0.0.1:8001/health",
            sidecar=sidecar,
            engine=sglang,
            stop=stop,
            interval=0.005,
            probe_timeout=1.0,
            max_failures=2,
            on_unhealthy=lambda: events.append(("retire", stop.is_set())),
        )
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert stop.is_set()
    assert events == [
        ("terminate", sidecar),
        ("retire", True),
    ]


@pytest.mark.parametrize("reason", ["stop_set", "child_exited"])
def test_watch_health_skips_retirement_on_shutdown_race(reason: str) -> None:
    sidecar = Mock(spec=["poll"])
    sidecar.poll.return_value = None
    sglang = Mock(spec=["poll"])
    sglang.poll.return_value = None

    stop = threading.Event()
    probe_started = threading.Event()
    release_probe = threading.Event()
    terminated: list[object] = []
    retired = threading.Event()

    def urlopen(url: str, timeout: float) -> object:
        probe_started.set()
        release_probe.wait(timeout=2)
        raise TimeoutError("interrupted by shutdown")

    with (
        patch("urllib.request.urlopen", side_effect=urlopen),
        patch("spindle.inference.serving.terminate", side_effect=terminated.append),
    ):
        thread = watch_health(
            "http://127.0.0.1:8001/health",
            sidecar=sidecar,
            engine=sglang,
            stop=stop,
            interval=0.005,
            probe_timeout=1.0,
            max_failures=1,
            on_unhealthy=retired.set,
        )
        assert probe_started.wait(timeout=2)
        if reason == "stop_set":
            stop.set()
        else:
            sglang.poll.return_value = 0
        release_probe.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert terminated == []
    assert not retired.is_set()


def test_terminate_ignores_already_exited_process_group() -> None:
    process = Mock(pid=4242)
    process.poll.return_value = None
    with patch("os.killpg", side_effect=ProcessLookupError):
        terminate(process)
