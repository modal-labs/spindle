from __future__ import annotations

import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

SGLANG_ENV = {"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"}


def start_sglang(
    model_path: str,
    *,
    port: int,
    context_length: int,
    max_loras_per_batch: int,
    max_loaded_loras: int,
    max_lora_rank: int,
    max_running_requests: int,
    max_queued_requests: int | None = None,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    parallel_world_size: int | None = None,
    trust_remote_code: bool = False,
    lora_target_modules: Sequence[str] = ("all",),
    enable_lora: bool = True,
    enable_return_routed_experts: bool = False,
    weight_update_staging: str | None = None,
    weight_update_max_compile_group_gb: float | None = None,
    memory_fraction: float = 0.85,
    schedule_policy: str = "fcfs",
) -> subprocess.Popen:
    if enable_lora and max_loaded_loras < max_loras_per_batch:
        raise ValueError("max_loaded_loras must be at least max_loras_per_batch")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    if max_queued_requests is not None and max_queued_requests < 1:
        raise ValueError("max_queued_requests must be at least 1")
    if expert_parallel_size < 1:
        raise ValueError("expert_parallel_size must be at least 1")
    if expert_tensor_parallel_size < 1:
        raise ValueError("expert_tensor_parallel_size must be at least 1")
    if enable_lora and not lora_target_modules:
        raise ValueError("lora_target_modules must not be empty")
    if weight_update_staging not in (None, "cpu"):
        raise ValueError("weight_update_staging must be None or 'cpu'")
    if (
        weight_update_max_compile_group_gb is not None
        and weight_update_staging != "cpu"
    ):
        raise ValueError(
            "weight_update_max_compile_group_gb requires weight_update_staging='cpu'"
        )
    if not 0 < memory_fraction < 1:
        raise ValueError("memory_fraction must be between 0 and 1")
    moe_parallel_size = expert_parallel_size * expert_tensor_parallel_size
    world_size = parallel_world_size or math.lcm(
        tensor_parallel_size,
        moe_parallel_size,
    )
    if world_size < 1:
        raise ValueError("parallel_world_size must be at least 1")
    if world_size % tensor_parallel_size:
        raise ValueError(
            "parallel_world_size must be divisible by tensor_parallel_size"
        )
    if expert_parallel_size > 1 and world_size % moe_parallel_size:
        raise ValueError("parallel_world_size must be divisible by EP times expert TP")
    attention_data_parallel_size = world_size // tensor_parallel_size
    moe_data_parallel_size = (
        world_size // moe_parallel_size if expert_parallel_size > 1 else 1
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        str(world_size),
    ]
    if attention_data_parallel_size > 1:
        command.extend(["--dp-size", str(attention_data_parallel_size)])
        command.append("--enable-dp-attention")
    if expert_parallel_size > 1:
        command.extend(["--ep-size", str(expert_parallel_size)])
    if moe_data_parallel_size > 1:
        command.extend(["--moe-dp-size", str(moe_data_parallel_size)])
    if trust_remote_code:
        command.append("--trust-remote-code")
    if weight_update_staging is not None:
        command.extend(
            ["--weight-update-staging", weight_update_staging, "--weight-version", "0"]
        )
    if weight_update_max_compile_group_gb is not None:
        command.extend(
            [
                "--weight-update-max-compile-group-gb",
                str(weight_update_max_compile_group_gb),
            ]
        )
    command.extend(
        [
            "--context-length",
            str(context_length),
            "--mem-fraction-static",
            str(memory_fraction),
            "--weight-loader-disable-mmap",
        ]
    )
    if enable_return_routed_experts:
        command.append("--enable-return-routed-experts")
    if enable_lora:
        command.extend(
            [
                "--enable-lora",
                "--max-loras-per-batch",
                str(max_loras_per_batch),
                "--max-loaded-loras",
                str(max_loaded_loras),
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-target-modules",
                *lora_target_modules,
            ]
        )
    command.extend(
        [
            "--max-running-requests",
            str(max_running_requests),
            "--schedule-policy",
            schedule_policy,
        ]
    )
    if max_queued_requests is not None:
        command.extend(
            [
                "--max-queued-requests",
                str(max_queued_requests),
            ]
        )
    return subprocess.Popen(command, start_new_session=True)


def start_fft_sidecar(
    *,
    port: int,
    sglang_port: int,
    bulletin_root: str,
    bulletin_volume: str,
    run_id: str,
    pinned_version: int | None = None,
    scoped_registry: str | None = None,
) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "spindle.inference.fft_sidecar",
        "--port",
        str(port),
        "--upstream-url",
        f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root",
        bulletin_root,
        "--bulletin-volume",
        bulletin_volume,
        "--run-id",
        run_id,
    ]
    if scoped_registry is not None:
        command.extend(["--scoped-registry", scoped_registry])
    if pinned_version is not None:
        command.extend(
            [
                "--pinned-version",
                str(pinned_version),
            ]
        )
    return subprocess.Popen(command, start_new_session=True)


def start_lora_sidecar(
    *,
    port: int,
    sglang_port: int,
    bulletin_root: str,
    bulletin_volume: str,
) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "spindle.inference.lora_sidecar",
        "--port",
        str(port),
        "--upstream-url",
        f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root",
        bulletin_root,
        "--bulletin-volume",
        bulletin_volume,
    ]
    return subprocess.Popen(command, start_new_session=True)


def request_activity(metrics: str) -> tuple[float, float]:
    """SGLang's running or queued and finished request counts, summed over its ranks."""
    totals = {"num_running_reqs": 0.0, "num_queue_reqs": 0.0, "num_requests_total": 0.0}
    seen = set()
    for line in metrics.splitlines():
        name = re.split(r"[{\s]", line, maxsplit=1)[0].removeprefix("sglang:")
        if line.startswith("sglang:") and name in totals:
            totals[name] += float(line.rsplit(maxsplit=1)[-1])
            seen.add(name)
    if "num_running_reqs" not in seen:
        raise ValueError("SGLang metrics have no sglang:num_running_reqs")
    active = totals["num_running_reqs"] + totals["num_queue_reqs"]
    return active, totals["num_requests_total"]


def sglang_busy(port: int) -> Callable[[], bool]:
    """Whether SGLang had work since the last call, counting requests that finished in between."""
    last_finished = None

    def busy() -> bool:
        nonlocal last_finished
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=2
        ) as response:
            active, finished = request_activity(response.read().decode())
        before, last_finished = last_finished, finished
        return active > 0 or (before is not None and finished > before)

    return busy


def wait_http(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited with code {process.returncode}: {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        time.sleep(2)
    raise TimeoutError(f"{url}: {error}")


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _probe_health(url: str, timeout: float) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def _log_gpu_state() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if result.stdout.strip():
        logger.error("GPU state before sampler termination:\n%s", result.stdout.strip())


def retire_container() -> None:
    """Signal PID 1 to exit so the platform replaces an unhealthy replica."""
    logger.error("Retiring unhealthy sampler container")
    os.kill(1, signal.SIGTERM)


def watch_health(
    url: str,
    *,
    sidecar: subprocess.Popen,
    engine: subprocess.Popen,
    stop: threading.Event | None = None,
    interval: float = 30.0,
    probe_timeout: float | None = None,
    max_failures: int = 4,
    on_unhealthy: Callable[[], None] | None = None,
) -> threading.Thread:
    """Retire a sampler replica after consecutive health check failures."""
    done = stop or threading.Event()
    if probe_timeout is None:
        sglang_timeout = float(os.environ.get("SGLANG_HEALTH_CHECK_TIMEOUT", "20"))
        probe_timeout = max(30.0, sglang_timeout + 10.0)

    def monitor() -> None:
        failures = 0
        while not done.wait(interval):
            if sidecar.poll() is not None or engine.poll() is not None:
                return
            ok, detail = _probe_health(url, probe_timeout)
            if done.is_set() or sidecar.poll() is not None or engine.poll() is not None:
                return
            if ok:
                failures = 0
                continue
            failures += 1
            logger.warning(
                "Sampler health check %s failed (%d/%d): %s",
                url,
                failures,
                max_failures,
                detail,
            )
            if failures < max_failures:
                continue
            logger.error(
                "Sampler health check %s failed %d consecutive times (%s); "
                "retiring unhealthy replica",
                url,
                failures,
                detail,
            )
            _log_gpu_state()
            if done.is_set() or sidecar.poll() is not None or engine.poll() is not None:
                return
            done.set()
            # Close the proxy port first so in-flight callers retry instead of
            # waiting out exit_grace_period, then retire immediately.
            terminate(sidecar)
            if on_unhealthy is not None:
                try:
                    on_unhealthy()
                except Exception:
                    logger.exception("Unhealthy sampler callback failed")
            else:
                terminate(engine)
            return

    thread = threading.Thread(
        target=monitor,
        name="tinker-rollout-health-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def supervise_children(
    first: subprocess.Popen,
    second: subprocess.Popen,
    *,
    stop: threading.Event | None = None,
    on_exit: Callable[[], None] | None = None,
) -> threading.Thread:
    """Terminate the sibling when either serving process exits unexpectedly."""
    done = stop or threading.Event()

    def monitor() -> None:
        while not done.is_set():
            if first.poll() is not None:
                terminate(second)
                if on_exit is not None and not done.is_set():
                    on_exit()
                return
            if second.poll() is not None:
                terminate(first)
                if on_exit is not None and not done.is_set():
                    on_exit()
                return
            done.wait(1)

    thread = threading.Thread(
        target=monitor,
        name="tinker-rollout-child-supervisor",
        daemon=True,
    )
    thread.start()
    return thread
