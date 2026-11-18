"""Isolated GLM Flash inference: resident serving, registration, activation, churn.

Set SPINDLE_LORA_BENCH_ASSET_VOLUME and SPINDLE_LORA_BENCH_ADAPTER_VOLUME.
Run with Modal; --adapter-path and --settings-path are relative to the latter.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import statistics
import struct
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import modal

from spindle.inference.serving import terminate, wait_http
from spindle.providers.modal.image_dependencies import ignore_config_source
from spindle.providers.modal.rollout_image import runtime_image_01_c4182b2
from spindle.providers.modal.kernel_cache import (
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_ROOT,
    kernel_cache_volume,
)

app = modal.App("spindle-lora-loading-investigation")
BENCH_VOLUME_ENV = {
    name: os.environ[name]
    for name in ("SPINDLE_LORA_BENCH_ASSET_VOLUME", "SPINDLE_LORA_BENCH_ADAPTER_VOLUME")
}
assets = modal.Volume.from_name(
    os.environ["SPINDLE_LORA_BENCH_ASSET_VOLUME"], version=2
)
adapters = modal.Volume.from_name(
    os.environ["SPINDLE_LORA_BENCH_ADAPTER_VOLUME"], version=2
)
output = modal.Volume.from_name(
    "spindle-lora-loading-investigation", version=2, create_if_missing=True
)


HOOKS = {
    "lora/lora_overlap_loader.py": "from spindle.inference.lora_packing import install_readiness; install_readiness()",
    "lora/lora.py": "from .loading_probe import patch_adapter; patch_adapter(LoRAAdapter)",
    "lora/mem_pool.py": "from .loading_probe import patch_pool; patch_pool(LoRAMemoryPool, globals())",
    "lora/lora_manager.py": "from .loading_probe import patch_manager; patch_manager(LoRAManager)",
    "managers/scheduler.py": "from spindle.inference.lora_packing import install_admission; install_admission(); from sglang.srt.lora.loading_probe import patch_scheduler; patch_scheduler(Scheduler)",
}
INSTALL_HOOKS = (
    "python - <<'PY'\nfrom pathlib import Path\n"
    "root = Path('/sgl-workspace/sglang/python/sglang/srt')\n"
    f"hooks = {HOOKS!r}\n"
    "for relative, hook in hooks.items():\n"
    "    path = root / relative\n"
    "    path.write_text(path.read_text() + '\\n' + hook + '\\n')\nPY"
)


image = (
    runtime_image_01_c4182b2.add_local_file(
        Path(__file__).with_name("lora_loading_probe.py"),
        "/sgl-workspace/sglang/python/sglang/srt/lora/loading_probe.py",
        copy=True,
    )
    .add_local_file(Path(__file__), "/opt/lora-loading-benchmark.py", copy=True)
    .run_commands(INSTALL_HOOKS)
    .add_local_python_source("spindle", ignore=ignore_config_source)
)


def quantile(values: list[float], fraction: float) -> float | None:
    return (
        sorted(values)[min(int(len(values) * fraction), len(values) - 1)]
        if values
        else None
    )


async def sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    pending = ""
    async for chunk in response.aiter_text():
        pending += chunk
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            yield line.removesuffix("\r")
    if pending:
        yield pending


async def traffic(
    client: httpx.AsyncClient,
    names: list[str],
    *,
    requests: int,
    concurrency: int,
    output_tokens: int,
    started: asyncio.Event | None = None,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    rows = []
    begin = time.perf_counter()

    async def generate(index: int) -> None:
        async with semaphore:
            sent, first, previous, count = time.perf_counter(), None, None, 0
            gaps = []
            payload = {
                "rid": f"loadbench-{uuid.uuid4().hex}",
                "input_ids": [
                    100 + (index * 13 + token) % 1000 for token in range(256)
                ],
                "lora_path": names[index % len(names)],
                "stream": True,
                "sampling_params": {
                    "max_new_tokens": output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            }
            async with client.stream("POST", "/generate", json=payload) as response:
                response.raise_for_status()
                async for line in sse_lines(response):
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    item = json.loads(line[6:])
                    tokens = item.get("meta_info", {}).get("completion_tokens", 0)
                    if tokens > count:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                            if started is not None:
                                started.set()
                        if previous is not None:
                            gaps.append(now - previous)
                        previous, count = now, tokens
            if count != output_tokens:
                raise AssertionError(
                    f"incomplete generation: {count} != {output_tokens}"
                )
            rows.append(
                {
                    "request": index,
                    "adapter": payload["lora_path"],
                    "start": sent,
                    "end": time.perf_counter(),
                    "tokens": count,
                    "ttft_s": first - sent if first else None,
                    "gaps_s": gaps,
                }
            )

    await asyncio.gather(*(generate(index) for index in range(requests)))
    end = time.perf_counter()
    gaps = [gap for row in rows for gap in row["gaps_s"]]
    return {
        "start": begin,
        "end": end,
        "duration_s": end - begin,
        "output_tokens": sum(row["tokens"] for row in rows),
        "output_tps": sum(row["tokens"] for row in rows) / (end - begin),
        "median_ttft_s": statistics.median(row["ttft_s"] for row in rows),
        "p99_stream_gap_s": quantile(gaps, 0.99),
        "max_stream_gap_s": max(gaps, default=None),
        "requests": rows,
    }


@app.function(
    image=image,
    gpu="B300:8",
    cpu=32,
    memory=524288,
    timeout=5400,
    volumes={
        "/assets": assets,
        "/adapters": adapters,
        "/out": output,
        KERNEL_CACHE_ROOT: kernel_cache_volume,
    },
    env=KERNEL_CACHE_ENV
    | BENCH_VOLUME_ENV
    | {
        "SPINDLE_GLM_ASYNC_LORA_LOADING": "1",
        "SGLANG_LORA_CSGMV_SPLIT_K_MAX_TOKENS": "128",
    },
    experimental_options={"exclude_reserved_capacity": True, "efa_enabled": True},
)
@modal.clustered(size=1, rdma=True)
def benchmark(
    run_id: str,
    adapter_path: str,
    settings_path: str,
    model_path: str,
    modes: list[str],
    requests: int,
    concurrency: int,
    gpu_slots: int,
    eviction: bool,
    compare_slots: int,
) -> dict:
    if not compare_slots:
        return run_benchmark(
            run_id,
            adapter_path,
            settings_path,
            model_path,
            modes,
            requests,
            concurrency,
            gpu_slots,
            eviction,
        )
    results = {}
    for slots in (gpu_slots, compare_slots):
        results[str(slots)] = run_benchmark(
            f"{run_id}-slots{slots}",
            adapter_path,
            settings_path,
            model_path,
            modes,
            requests,
            concurrency,
            slots,
            eviction,
        )
    assert len({result["hardware"] for result in results.values()}) == 1
    return {"capacities": results}


def run_benchmark(
    run_id: str,
    adapter_path: str,
    settings_path: str,
    model_path: str,
    modes: list[str],
    requests: int,
    concurrency: int,
    gpu_slots: int,
    eviction: bool,
) -> dict:
    folder = Path("/out") / run_id
    folder.mkdir(parents=True)
    os.environ["SPINDLE_LORA_TRACE_ROOT"] = str(folder)
    adapter = Path("/adapters") / adapter_path
    weights = adapter / "adapter_model.safetensors"
    with weights.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    settings = json.loads((Path("/adapters") / settings_path).read_text())["settings"]
    settings.update(
        context_length=16384,
        max_running_requests=concurrency,
        max_total_tokens=65536,
        cuda_graph_max_bs_decode=concurrency,
        max_loaded_loras=max(8, gpu_slots),
        max_loras_per_batch=gpu_slots,
        enable_return_routed_experts=False,
        disable_radix_cache=True,
        enable_lora_overlap_loading=True,
        weight_loader_disable_mmap=True,
    )
    report = {
        "run_id": run_id,
        "workload_version": 7,
        "source": "c4182b22139a5730c9f18907cc0b178a305bb0f1",
        "task_id": os.environ.get("MODAL_TASK_ID"),
        "settings": settings,
        "adapter_bytes": weights.stat().st_size,
        "adapter_tensors": len(header) - int("__metadata__" in header),
        "adapter_config": json.loads((adapter / "adapter_config.json").read_text()),
        "hardware": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv"],
            text=True,
        ),
        "trials": [],
    }
    probe = Path(
        "/sgl-workspace/sglang/python/sglang/srt/lora/loading_probe.py"
    ).read_bytes()
    (folder / "loading_probe.py").write_bytes(probe)
    report["probe_sha256"] = hashlib.sha256(probe).hexdigest()
    benchmark_source = Path("/opt/lora-loading-benchmark.py").read_bytes()
    (folder / "benchmark.py").write_bytes(benchmark_source)
    report["benchmark_sha256"] = hashlib.sha256(benchmark_source).hexdigest()
    packing_spec = importlib.util.find_spec("spindle.inference.lora_packing")
    assert packing_spec is not None and packing_spec.origin is not None
    packing_source = Path(packing_spec.origin).read_bytes()
    (folder / "lora_packing.py").write_bytes(packing_source)
    report["packing_sha256"] = hashlib.sha256(packing_source).hexdigest()

    def save() -> None:
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        output.commit()

    process = None
    log = (folder / "server.log").open("w")
    try:
        save()
        print(
            "BENCH_ARTIFACTS", run_id, "tensors", report["adapter_tensors"], flush=True
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spindle.inference.sglang",
                model_path,
                json.dumps(settings),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_http("http://127.0.0.1:8001/ready", process, timeout=2400)

        async def run() -> None:
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:8001",
                timeout=1800,
                limits=httpx.Limits(max_connections=None),
            ) as client:

                async def register(name: str) -> dict:
                    started = time.perf_counter()
                    response = await client.post(
                        "/load_lora_adapter",
                        json={
                            "lora_name": name,
                            "lora_path": str(adapter),
                            "pinned": False,
                        },
                    )
                    response.raise_for_status()
                    assert response.json().get("success"), response.text
                    result = {
                        "name": name,
                        "start": started,
                        "end": time.perf_counter(),
                    }
                    print("LOAD", name, result["end"] - started, flush=True)
                    return result

                async def activate(name: str) -> dict:
                    started = time.perf_counter()
                    response = await client.post(
                        "/generate",
                        json={
                            "input_ids": list(range(100, 164)),
                            "lora_path": name,
                            "sampling_params": {
                                "max_new_tokens": 1,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                    )
                    response.raise_for_status()
                    return {
                        "name": name,
                        "duration_s": time.perf_counter() - started,
                        "result": response.json(),
                    }

                for mode in modes:
                    trial: dict[str, Any] = {
                        "mode": mode,
                        "registrations": [],
                        "activations": [],
                    }
                    report["trials"].append(trial)
                    names = [
                        f"{mode}-{uuid.uuid4().hex[:8]}-{index}"
                        for index in range(12 if eviction else 8)
                    ]
                    for index, name in enumerate(names[:4]):
                        if index == 0:
                            (folder / "profile-on").touch()
                        trial["registrations"].append(await register(name))
                        if mode == "packed":
                            (folder / "verify-on").touch()
                        trial["activations"].append(await activate(name))
                        (folder / "verify-on").unlink(missing_ok=True)
                        (folder / "profile-on").unlink(missing_ok=True)
                    trial["concurrent_warmup"] = await traffic(
                        client,
                        names[:4],
                        requests=concurrency,
                        concurrency=concurrency,
                        output_tokens=8,
                    )
                    save()
                    trial["resident"] = await traffic(
                        client,
                        names[:4],
                        requests=requests,
                        concurrency=concurrency,
                        output_tokens=256,
                    )
                    first_token = asyncio.Event()

                    async def load_under_traffic(
                        event: asyncio.Event = first_token,
                        pending_names: tuple[str, ...] = tuple(names[4:8]),
                    ) -> list[dict]:
                        await asyncio.wait_for(event.wait(), 120)
                        return [await register(name) for name in pending_names]

                    (
                        trial["registration_traffic"],
                        trial["background_loads"],
                    ) = await asyncio.gather(
                        traffic(
                            client,
                            names[:4],
                            requests=requests * 2,
                            concurrency=concurrency,
                            output_tokens=512,
                            started=first_token,
                        ),
                        load_under_traffic(),
                    )
                    save()
                    trial["first_activation_after_registration"] = await activate(
                        names[4]
                    )
                    (folder / "profile-on").touch()
                    trial["profiled_activation"] = await activate(names[5])
                    (folder / "profile-on").unlink()
                    for name in names[:8]:
                        await activate(name)
                    trial["churn_warmup"] = await traffic(
                        client,
                        names[:8],
                        requests=concurrency,
                        concurrency=concurrency,
                        output_tokens=8,
                    )
                    trial["churn"] = await traffic(
                        client,
                        names[:8],
                        requests=requests,
                        concurrency=concurrency,
                        output_tokens=256,
                    )
                    await traffic(
                        client,
                        names[:4],
                        requests=concurrency,
                        concurrency=concurrency,
                        output_tokens=8,
                    )
                    trial["resident_after_churn"] = await traffic(
                        client,
                        names[:4],
                        requests=requests,
                        concurrency=concurrency,
                        output_tokens=256,
                    )
                    if eviction:
                        first_token = asyncio.Event()
                        (
                            trial["eviction_traffic"],
                            trial["evicting_loads"],
                        ) = await asyncio.gather(
                            traffic(
                                client,
                                names[:4],
                                requests=requests * 2,
                                concurrency=concurrency,
                                output_tokens=512,
                                started=first_token,
                            ),
                            load_under_traffic(first_token, tuple(names[8:])),
                        )
                    save()
                    catalog = await client.get("/v1/models")
                    catalog.raise_for_status()
                    registered_names = {item["id"] for item in catalog.json()["data"]}
                    for name in names:
                        if name not in registered_names:
                            continue
                        response = await client.post(
                            "/unload_lora_adapter", json={"lora_name": name}
                        )
                        response.raise_for_status()
                        assert response.json().get("success"), response.text
                    print(
                        "TRIAL",
                        mode,
                        {
                            case: round(trial[case]["output_tps"], 2)
                            for case in ("resident", "registration_traffic", "churn")
                        },
                        flush=True,
                    )
                    save()

        asyncio.run(run())
        started = time.perf_counter()
        digest = hashlib.sha256()
        with weights.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        report["warm_volume_verification"] = {
            "duration_s": time.perf_counter() - started,
            "sha256": digest.hexdigest(),
        }
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        terminate(process)
        log.close()
        save()
    return report


@app.local_entrypoint()
def main(
    adapter_path: str,
    settings_path: str,
    model_path: str = "/assets/model",
    modes: str = "baseline,pinned,packed",
    requests: int = 64,
    concurrency: int = 32,
    gpu_slots: int = 4,
    eviction: bool = False,
    compare_slots: int = 0,
    result_path: str = "/tmp/lora-loading-report.json",
) -> None:
    selected = modes.split(",")
    if not set(selected) <= {"baseline", "pinned", "packed"}:
        raise ValueError(selected)
    if gpu_slots < 4:
        raise ValueError("The resident workload requires at least four GPU slots")
    if compare_slots and (compare_slots < 4 or compare_slots == gpu_slots):
        raise ValueError(
            "The capacity control requires a different size of at least four"
        )
    run_id = "loading-" + uuid.uuid4().hex[:12]
    print("RUN_ID", run_id, flush=True)
    result = benchmark.remote(
        run_id,
        adapter_path,
        settings_path,
        model_path,
        selected,
        requests,
        concurrency,
        gpu_slots,
        eviction,
        compare_slots,
    )
    Path(result_path).write_text(json.dumps(result, indent=2))
