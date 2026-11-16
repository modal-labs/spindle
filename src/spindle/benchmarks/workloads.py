from __future__ import annotations

import asyncio
import ast
import hashlib
import inspect
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import httpx
import tinker
from tinker import types

from spindle.client import create_full_training_client_async


def prompt_tokens(tokenizer, length, client):
    text = f"Benchmark client {client}. Explain how to implement a stable parallel reduction. "
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        raise ValueError("Tokenizer produced an empty prompt")
    return (seed * (length // len(seed) + 1))[:length]


async def sdk_suite(config, base_url, recorder):
    headers = {}
    if os.getenv("MODAL_PROXY_TOKEN_ID"):
        headers = {
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        }
    header_key = (
        "default_headers_overrides"
        if "default_headers_overrides"
        in inspect.signature(tinker.ServiceClient.__init__).parameters
        else "default_headers"
    )
    client_options: dict[str, Any] = {"base_url": base_url, header_key: headers}
    service = tinker.ServiceClient(**client_options)
    for clients in config.clients:
        trainers = []
        try:
            for client in range(clients):
                started = time.monotonic()
                metadata = {
                    "run_id": recorder.run_id,
                    "attempt_id": recorder.run_id,
                }
                if config.parameterization == "full":
                    trainer = await create_full_training_client_async(
                        service, config.model, user_metadata=metadata
                    )
                else:
                    trainer = await service.create_lora_training_client_async(
                        config.model,
                        rank=config.rank,
                        train_unembed=False,
                        user_metadata=metadata,
                    )
                trainers.append(trainer)
                recorder.record(
                    {"client.create_s": time.monotonic() - started},
                    {"client": client, "clients": clients, "phase": "startup"},
                )
            tokenizer = await asyncio.to_thread(trainers[0].get_tokenizer)
            for context in config.context_lengths:

                async def drive(
                    client,
                    trainer,
                    *,
                    context=context,
                    clients=clients,
                    tokenizer=tokenizer,
                ):
                    tokens = prompt_tokens(
                        tokenizer,
                        context + (1 if config.suite != "inference" else 0),
                        client,
                    )
                    labels = {
                        "client": client,
                        "clients": clients,
                        "context_length": context,
                        "batch_size": config.batch_size,
                    }
                    recorder.event(
                        "workload",
                        {
                            "prompt_sha256": hashlib.sha256(
                                json.dumps(tokens).encode()
                            ).hexdigest(),
                            "input_tokens": len(tokens),
                        },
                        labels,
                    )
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": types.TensorData(
                                data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]
                            ),
                            "weights": types.TensorData(
                                data=[1.0] * (len(tokens) - 1),
                                dtype="float32",
                                shape=[len(tokens) - 1],
                            ),
                        },
                    )
                    sampler = None
                    if config.suite == "inference":
                        sampler = (
                            await trainer.save_weights_and_get_sampling_client_async()
                        )
                    measured_start = None
                    measured_end = None
                    measured_tokens = 0
                    for step in range(config.warmup_steps + config.steps):
                        tags = {
                            **labels,
                            "step": step,
                            "phase": "warmup"
                            if step < config.warmup_steps
                            else "measurement",
                        }
                        started = time.monotonic()
                        measured = step >= config.warmup_steps
                        if measured and measured_start is None:
                            measured_start = started
                        if config.suite == "inference":
                            response = await sampler.sample_async(
                                types.ModelInput.from_ints(tokens),
                                num_samples=config.batch_size,
                                sampling_params=types.SamplingParams(
                                    max_tokens=config.output_tokens,
                                    temperature=0,
                                    seed=config.seed,
                                ),
                            )
                            seconds = time.monotonic() - started
                            count = sum(len(s.tokens) for s in response.sequences)
                            if count == 0:
                                raise ValueError("Sampling returned no output tokens")
                            if measured:
                                measured_tokens += count
                            recorder.record(
                                {
                                    "sampling.latency_s": seconds,
                                    "sampling.output_tokens": count,
                                    "sampling.output_tokens_per_s": count / seconds,
                                    "sampling.input_tokens": context
                                    * config.batch_size,
                                },
                                tags,
                            )
                        else:
                            method = (
                                trainer.forward_async
                                if config.suite == "eval"
                                else trainer.forward_backward_async
                            )
                            forward = await method(
                                [datum] * config.batch_size, "cross_entropy"
                            )
                            result = await forward.result_async()
                            forward_seconds = time.monotonic() - started
                            recorder.record(
                                {
                                    "forward"
                                    if config.suite == "eval"
                                    else "forward_backward": result.metrics,
                                    "eval.forward_s"
                                    if config.suite == "eval"
                                    else "training.forward_backward_s": forward_seconds,
                                },
                                tags,
                                raw=result.metrics,
                            )
                            if any(
                                isinstance(v, (int, float)) and not math.isfinite(v)
                                for v in result.metrics.values()
                            ):
                                raise ValueError("Nonfinite training metric")
                            if config.suite == "eval":
                                recorder.record(
                                    {
                                        "eval.input_tokens": context
                                        * config.batch_size,
                                        "eval.input_tokens_per_s": context
                                        * config.batch_size
                                        / forward_seconds,
                                    },
                                    tags,
                                )
                                if measured:
                                    measured_end = time.monotonic()
                                continue
                            optim_started = time.monotonic()
                            optimizer = await trainer.optim_step_async(
                                types.AdamParams(learning_rate=config.learning_rate)
                            )
                            optim_result = await optimizer.result_async()
                            recorder.record(
                                {"optimizer": optim_result.metrics},
                                tags,
                                raw=optim_result.metrics,
                            )
                            if (
                                optim_result.metrics.get("update_successful:mean", 1)
                                != 1
                            ):
                                raise ValueError(
                                    "Optimizer reported an unsuccessful update"
                                )
                            update_finished = time.monotonic()
                            if any(
                                isinstance(v, (int, float)) and not math.isfinite(v)
                                for v in optim_result.metrics.values()
                            ):
                                raise ValueError("Nonfinite training metric")
                            recorder.record(
                                {
                                    "training.optimizer_s": update_finished
                                    - optim_started,
                                    "training.step_s": update_finished - started,
                                    "training.input_tokens": context
                                    * config.batch_size,
                                    "training.input_tokens_per_s": context
                                    * config.batch_size
                                    / (update_finished - started),
                                },
                                tags,
                            )
                            if config.suite == "weight-sync":
                                sampler = await trainer.save_weights_and_get_sampling_client_async()
                                published = time.monotonic()
                                response = await sampler.sample_async(
                                    types.ModelInput.from_ints(tokens[:-1]),
                                    num_samples=1,
                                    sampling_params=types.SamplingParams(
                                        max_tokens=1, temperature=0
                                    ),
                                )
                                if (
                                    not response.sequences
                                    or not response.sequences[0].tokens
                                ):
                                    raise ValueError(
                                        "Published-weight readiness probe returned no tokens"
                                    )
                                recorder.record(
                                    {
                                        "weight_sync.publish_receipt_s": published
                                        - update_finished,
                                        "weight_sync.probe_s": time.monotonic()
                                        - published,
                                        "weight_sync.update_to_first_response_s": time.monotonic()
                                        - update_finished,
                                    },
                                    tags,
                                )
                        if measured:
                            measured_end = time.monotonic()
                    return {
                        "start": measured_start,
                        "end": measured_end,
                        "output_tokens": measured_tokens,
                    }

                tasks = [
                    asyncio.create_task(drive(client, trainer))
                    for client, trainer in enumerate(trainers)
                ]
                try:
                    outcomes = await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                elapsed = max(row["end"] for row in outcomes) - min(
                    row["start"] for row in outcomes
                )
                aggregate = {"aggregate.wall_s": elapsed}
                if config.suite == "inference":
                    total_tokens = sum(row["output_tokens"] for row in outcomes)
                    aggregate.update(
                        {
                            "aggregate.output_tokens": total_tokens,
                            "aggregate.output_tokens_per_s": total_tokens / elapsed,
                        }
                    )
                else:
                    aggregate.update(
                        {
                            "aggregate.updates_per_s": clients * config.steps / elapsed,
                            "aggregate.eval_input_tokens_per_s"
                            if config.suite == "eval"
                            else "aggregate.training_input_tokens_per_s": clients
                            * config.steps
                            * context
                            * config.batch_size
                            / elapsed,
                        }
                    )
                recorder.record(
                    aggregate,
                    {
                        "clients": clients,
                        "context_length": context,
                        "phase": "measurement",
                        "aggregation": "all_clients",
                    },
                )
        finally:
            async with httpx.AsyncClient(
                base_url=base_url.rstrip("/").removesuffix("/api/v1"),
                headers={"X-API-Key": os.environ["TINKER_API_KEY"], **headers},
                timeout=60,
            ) as client:
                for trainer in trainers:
                    try:
                        response = await client.post(
                            "/api/v1/unload_model", json={"model_id": trainer.model_id}
                        )
                        response.raise_for_status()
                    except Exception as exc:
                        recorder.collection_error("unload_model", type(exc).__name__)


async def control_plane_suite(config, base_url, recorder):
    headers = {"X-API-Key": os.environ["TINKER_API_KEY"]}
    if os.getenv("MODAL_PROXY_TOKEN_ID"):
        headers.update(
            {
                "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
                "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
            }
        )
    base = base_url.rstrip("/").removesuffix("/api/v1")
    for clients in config.clients:
        async with httpx.AsyncClient(
            base_url=base,
            headers=headers,
            timeout=60,
            limits=httpx.Limits(
                max_connections=clients, max_keepalive_connections=clients
            ),
        ) as http:
            for phase, count in (
                ("warmup", config.warmup_steps),
                ("measurement", config.requests_per_client),
            ):
                latencies = []
                errors = []
                gate = asyncio.Event()

                async def drive(
                    client,
                    *,
                    gate=gate,
                    count=count,
                    errors=errors,
                    latencies=latencies,
                    phase=phase,
                    clients=clients,
                ):
                    await gate.wait()
                    for request in range(count):
                        started = time.monotonic()
                        status = 0
                        failed = False
                        try:
                            response = await http.get("/api/v1/get_server_capabilities")
                            status = response.status_code
                            response.raise_for_status()
                            if not isinstance(
                                response.json().get("supported_models"), list
                            ):
                                raise ValueError("Invalid capabilities response")
                        except (httpx.HTTPError, ValueError) as exc:
                            failed = True
                            errors.append(type(exc).__name__)
                        elapsed = time.monotonic() - started
                        latencies.append(elapsed)
                        recorder.record(
                            {
                                "control_plane.request_s": elapsed,
                                "control_plane.errors": int(failed),
                            },
                            {
                                "phase": phase,
                                "clients": clients,
                                "client": client,
                                "step": request,
                                "status": status,
                                "endpoint": "get_server_capabilities",
                            },
                        )

                tasks = [
                    asyncio.create_task(drive(client)) for client in range(clients)
                ]
                started = time.monotonic()
                gate.set()
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                elapsed = time.monotonic() - started
                if latencies:
                    recorder.record(
                        {
                            "control_plane.mean_s": float(np.mean(latencies)),
                            "control_plane.p95_s": float(np.quantile(latencies, 0.95)),
                            "control_plane.requests_per_s": len(latencies) / elapsed,
                            "control_plane.request_count": len(latencies),
                            "control_plane.error_count": len(errors),
                        },
                        {
                            "phase": phase,
                            "clients": clients,
                            "endpoint": "get_server_capabilities",
                        },
                    )
                if errors:
                    raise RuntimeError(f"{len(errors)} control-plane requests failed")


def recipe_command(config, root, output, base_url):
    work = output / "recipe"
    work.mkdir(parents=True)
    if config.recipe == "dapo":
        shutil.copytree(
            root / "examples/dapo-27b-4h200-12b200", work, dirs_exist_ok=True
        )
        values = json.loads((work / "config.json").read_text())
        values.update(config.recipe_config)
        values.update(
            model=config.model,
            updates=config.steps,
            context_tokens=max(config.context_lengths),
            max_tokens=config.output_tokens,
        )
        values.update(
            seed=config.seed, rank=config.rank, learning_rate=config.learning_rate
        )
        (work / "config.json").write_text(json.dumps(values))
        command = [
            sys.executable,
            "-u",
            str(work / "launch_clients.py"),
            "--clients",
            str(config.clients[0]),
            "--mode",
            "gate",
            "--stagger-seconds",
            "0",
            "--base-url",
            base_url,
        ]
    elif config.recipe == "swe-gym":
        worker = Path(os.environ.get("SWE_GYM_WORKER", ""))
        if not worker.is_file():
            raise ValueError(
                "Set SWE_GYM_WORKER to a pinned SWE-Gym worker checkout; its dependencies must be installed"
            )
        helpers = worker.parent / "helpers"
        if not helpers.is_dir():
            raise ValueError(
                "SWE_GYM_WORKER must have its helpers directory alongside it"
            )
        shutil.copytree(helpers, work / "helpers")
        tree = ast.parse(worker.read_text())
        found = set()
        for statement in tree.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                name = statement.targets[0].id
                if name == "MODEL":
                    statement.value = ast.Constant(config.model)
                    found.add(name)
                elif name == "TRAINING":
                    values = ast.literal_eval(statement.value)
                    values.update(config.recipe_config)
                    values.update(
                        context_tokens=config.context_lengths[0],
                        turn_tokens=config.output_tokens,
                        seed=config.seed,
                    )
                    statement.value = ast.parse(repr(values), mode="eval").body
                    found.add(name)
        if found != {"MODEL", "TRAINING"}:
            raise ValueError(
                "Unsupported SWE-Gym worker layout: expected MODEL and TRAINING assignments"
            )
        (work / "swe_gym.py").write_text(ast.unparse(ast.fix_missing_locations(tree)))
        command = [
            sys.executable,
            "-u",
            str(work / "swe_gym.py"),
            "--clients",
            str(config.clients[0]),
            "--steps",
            str(config.steps),
            "--groups-per-batch",
            "1",
            "--hours",
            str(config.timeout_s / 3600),
            "--output",
            str(work / "results"),
        ]
    else:
        command = [
            part.replace("{output}", str(work))
            .replace("{base_url}", base_url)
            .replace("{model}", config.model)
            for part in config.command
        ]
    return command, work


def command_suite(config, root, output, base_url, recorder):
    command, work = recipe_command(config, root, output, base_url)
    instrumentation = work / "instrumentation"
    instrumentation.mkdir()
    proxy_helper = root / "examples/shared-server/with_proxy_auth.py"
    if proxy_helper.is_file():
        shutil.copy2(proxy_helper, instrumentation / "with_proxy_auth.py")
        (instrumentation / "sitecustomize.py").write_text(
            'import os\nif os.getenv("MODAL_PROXY_TOKEN_ID"):\n    import with_proxy_auth\n    with_proxy_auth.install()\n'
        )
    recipe_files = {
        str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(work.rglob("*"))
        if p.is_file()
    }
    recorder.manifest["recipe_files"] = recipe_files
    recorder.manifest["recipe_fingerprint"] = hashlib.sha256(
        json.dumps(recipe_files, sort_keys=True).encode()
    ).hexdigest()
    recorder.store.register(recorder.manifest)
    offsets = {}
    process = None
    completed = 0

    def ingest():
        nonlocal completed
        for path in work.rglob("*.jsonl"):
            if "events" not in path.name and "metrics" not in path.name:
                continue
            with path.open() as stream:
                stream.seek(offsets.get(str(path), 0))
                while line := stream.readline():
                    if not line.endswith("\n"):
                        break
                    try:
                        value = json.loads(line)
                    except ValueError:
                        recorder.collection_error(
                            str(path.relative_to(work)), "Invalid JSON line"
                        )
                        offsets[str(path)] = stream.tell()
                        continue
                    offsets[str(path)] = stream.tell()
                    if value.get("event") in ("update_done", "step_done"):
                        completed += 1
                    recorder.record(
                        {"recipe": value},
                        {
                            "source": str(path.relative_to(work)),
                            "event": value.get("event", "metrics"),
                            "step": value.get("update", value.get("step", 0)),
                        },
                        raw=value,
                        stamp=value.get("time"),
                    )

    try:
        with (output / "recipe.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=work,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        [str(instrumentation), os.environ.get("PYTHONPATH", "")]
                    ),
                    "TINKER_BASE_URL": base_url,
                    "SPINDLE_BENCHMARK_RUN_ID": recorder.run_id,
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + config.timeout_s
            while process.poll() is None:
                ingest()
                if time.monotonic() > deadline:
                    raise TimeoutError("RL benchmark exceeded timeout_s")
                time.sleep(1)
            ingest()
            if process.returncode:
                raise RuntimeError(
                    f"RL recipe exited with {process.returncode}; see recipe.log"
                )
            if completed < config.steps * config.clients[0]:
                raise ValueError(
                    "RL recipe did not emit the expected completed updates"
                )
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            ingest()
