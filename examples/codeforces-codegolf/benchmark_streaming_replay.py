"""Pair streamed and batched training on identical recorded rollout schedules."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import modal
import spindle
import tinker
from spindle.client import create_full_training_client_async
from spindle.engines import qwen3_5_9b_full_64k
from spindle.providers.modal.scoped import control_image
from tinker import types

from codegolf.config import BatchConfig
from codegolf.train import (
    forward_backward,
    gather_work,
    proxy_headers,
    release,
    streamed_forward,
    training_datums,
)

APP_NAME = "spindle-codegolf-streaming-replay"
PROXY_SECRET = os.environ.get("CODEGOLF_PROXY_SECRET", "spindle-proxy")
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name("spindle-codegolf-streaming-benchmark")
image = control_image().add_local_python_source("codegolf")


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_cases(source):
    spec = json.loads((source / "spec.json").read_text())
    cfg = BatchConfig(**spec["config"])
    if not cfg.sequence_mean or cfg.stream_minibatches or cfg.async_rollouts:
        raise ValueError("Replay source must use the batched sequence-mean objective")
    cases = []
    for step in (3, 4, 5):
        records = sorted(
            (
                json.loads(path.read_text())
                for path in (source / f"rollouts/step-{step:04d}").glob("*.json")
            ),
            key=lambda row: row["problem_id"],
        )
        if len(records) != cfg.prompts_per_step:
            raise ValueError(f"Step {step} does not contain every problem group")
        if len({record["problem_id"] for record in records}) != len(records):
            raise ValueError(f"Step {step} contains duplicate problem groups")
        delays = []
        for record in records:
            if len(record["rows"]) != cfg.group_size:
                raise ValueError(f"Step {step} contains an incomplete advantage group")
            delay = record["sampling_seconds"] + record["judge_seconds"]
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("Group readiness must be finite and non-negative")
            for row in record["rows"]:
                if not row["tokens"] or len(row["tokens"]) != len(row["logprobs"]):
                    raise ValueError("Replay tokens and log probabilities must align")
            delays.append(delay)
        datums = [
            {
                "tokens": item.model_input.to_ints(),
                "loss_fn_inputs": {
                    key: {
                        "data": list(value.data),
                        "dtype": value.dtype,
                        "shape": value.shape,
                    }
                    for key, value in item.loss_fn_inputs.items()
                },
            }
            for item in training_datums(records, cfg)
        ]
        cases.append(
            {
                "id": f"step-{step}",
                "records": records,
                "delays": delays,
                "datums_sha256": digest(datums),
                "schedule_sha256": digest(delays),
            }
        )
    return cfg, cases


async def arrive(record, deadline):
    await asyncio.sleep(max(0, deadline - time.monotonic()))
    return record


async def trial(training, cfg, case, mode, profile, *, learning_rate=0.0):
    if mode not in {"batch", "stream"} or profile not in {"all_ready", "recorded"}:
        raise ValueError("Unknown replay mode or readiness profile")
    started = time.monotonic()
    delays = case["delays"] if profile == "recorded" else [0] * len(case["records"])
    arrivals = [
        arrive(record, started + delay)
        for record, delay in zip(case["records"], delays, strict=True)
    ]
    if mode == "stream":
        _, _, phases = await streamed_forward(training, arrivals, cfg)
    else:
        wait_started = time.monotonic()
        records = await gather_work(*arrivals)
        waiting = time.monotonic() - wait_started
        forward_started = time.monotonic()
        await forward_backward(training, records, cfg)
        phases = {
            "group_wait_seconds": waiting,
            "forward_backward_seconds": time.monotonic() - forward_started,
            "minibatches": 1,
        }
    optimizer_started = time.monotonic()
    result = await (
        await training.optim_step_async(
            types.AdamParams(learning_rate=learning_rate, grad_clip_norm=1.0)
        )
    ).result_async()
    finished = time.monotonic()
    elapsed = finished - started
    if result.metrics.get("update_successful:mean") != 1:
        raise RuntimeError("Replay optimizer update did not succeed")
    norm = result.metrics["grad_norm:mean"]
    if not math.isfinite(norm):
        raise RuntimeError("Replay gradient norm is not finite")
    return {
        "case": case["id"],
        "mode": mode,
        "profile": profile,
        "seconds": elapsed,
        "optimizer_seconds": finished - optimizer_started,
        "grad_norm": norm,
        "datums_sha256": case["datums_sha256"],
        "schedule_sha256": case["schedule_sha256"]
        if profile == "recorded"
        else digest([0] * len(delays)),
        **phases,
    }


async def probe(training, cfg, case, *, groupwise=False, all_rows=False):
    records = [
        dict(record, rows=record["rows"] if all_rows else record["rows"][:1])
        for record in case["records"]
    ]
    outputs = []
    for batch in ([record] for record in records) if groupwise else [records]:
        response = await (
            await training.forward_async(training_datums(batch, cfg), "cross_entropy")
        ).result_async()
        outputs.extend(list(row["logprobs"].data) for row in response.loss_fn_outputs)
    return outputs


def probe_difference(first, second):
    if len(first) != len(second) or any(
        len(a) != len(b) for a, b in zip(first, second, strict=True)
    ):
        raise ValueError("Probe outputs have different shapes")
    differences = [
        abs(a - b)
        for left, right in zip(first, second, strict=True)
        for a, b in zip(left, right, strict=True)
    ]
    if not differences or not all(math.isfinite(value) for value in differences):
        raise ValueError("Probe outputs must be finite and nonempty")
    return {"max_abs": max(differences), "mean_abs": statistics.fmean(differences)}


def summarize(rows, pairs_per_case, case_ids, *, require_equivalence=True):
    summaries = {}
    for profile in ("all_ready", "recorded"):
        paired = {}
        for row in rows:
            if row["profile"] == profile:
                pair = paired.setdefault(row["pair_id"], {})
                if row["mode"] in pair:
                    raise ValueError("Duplicate mode in a replay pair")
                pair[row["mode"]] = row
        if not paired or any(
            set(pair) != {"batch", "stream"} for pair in paired.values()
        ):
            raise ValueError("Every replay pair must contain both modes")
        savings = {}
        for pair in paired.values():
            batch, stream = pair["batch"], pair["stream"]
            if any(
                batch[key] != stream[key]
                for key in ("case", "datums_sha256", "schedule_sha256")
            ):
                raise ValueError("Replay pair inputs or readiness schedules differ")
            if require_equivalence and not math.isclose(
                batch["grad_norm"], stream["grad_norm"], rel_tol=5e-4, abs_tol=1e-6
            ):
                raise ValueError("Paired GPU gradient norms differ")
            savings.setdefault(batch["case"], []).append(
                batch["seconds"] - stream["seconds"]
            )
        if set(savings) != set(case_ids) or any(
            len(values) != pairs_per_case for values in savings.values()
        ):
            raise ValueError("Replay cases have incomplete repeat counts")
        rng = random.Random(42)
        bootstrap = sorted(
            statistics.fmean(
                value
                for values in savings.values()
                for value in rng.choices(values, k=len(values))
            )
            for _ in range(10_000)
        )
        summaries[profile] = {
            "pairs": len(paired),
            "median_batch_seconds": statistics.median(
                pair["batch"]["seconds"] for pair in paired.values()
            ),
            "median_stream_seconds": statistics.median(
                pair["stream"]["seconds"] for pair in paired.values()
            ),
            "mean_paired_saving_seconds": statistics.fmean(
                value for values in savings.values() for value in values
            ),
            "paired_saving_ci95_seconds": [bootstrap[249], bootstrap[9749]],
            "per_case_mean_saving_seconds": {
                case: statistics.fmean(values) for case, values in savings.items()
            },
        }
    return summaries


@app.function(
    image=image,
    volumes={"/runs": volume},
    secrets=[modal.Secret.from_name(PROXY_SECRET)],
    nonpreemptible=True,
    timeout=3600,
    cpu=4,
    memory=8192,
)
def benchmark(
    label: str,
    commit: str,
    proxy_secret: str,
    pairs: int = 4,
    seed: int = 42,
    timing_only_from: str = "",
):
    if not label or Path(label).name != label or label in {".", ".."}:
        raise ValueError("label must be a single directory name")
    if pairs < 2 or pairs % 2:
        raise ValueError("pairs must be positive and even, with at least two repeats")
    source = Path("/runs/streaming-20261118-210d46d/0-batch-v11")
    cfg, cases = load_cases(source)
    root = Path("/runs") / label
    root.mkdir(exist_ok=False)
    original = qwen3_5_9b_full_64k()
    engine = dataclasses.replace(
        original,
        revision=MODEL_REVISION,
        trainer_cpu=32,
        trainer_memory=262_144,
        backend_env={
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUDA_MODULE_LOADING": "EAGER",
        },
        training=dataclasses.replace(
            original.training,
            seed=42,
            provider_overrides={
                **original.training.provider_overrides,
                "hidden_dropout": 0.0,
                "attention_dropout": 0.0,
            },
        ),
    )
    report = {
        "commit": commit,
        "model_revision": MODEL_REVISION,
        "source_run": str(source.relative_to("/runs")),
        "config": dataclasses.asdict(cfg),
        "engine_training": dataclasses.asdict(engine.training),
        "trainer_gpu": engine.trainer_gpu,
        "trainer_cpu": engine.trainer_cpu,
        "trainer_memory_mib": engine.trainer_memory,
        "backend_env": engine.backend_env,
        "pairs_per_case": pairs,
        "seed": seed,
        "cases": [
            {key: value for key, value in case.items() if key != "records"}
            for case in cases
        ],
        "warmup": [],
        "trials": [],
        "scope_closed": False,
    }

    def write_report():
        (root / "result.json").write_text(json.dumps(report, indent=2))

    def persist():
        write_report()
        volume.commit()

    async def persist_async():
        write_report()
        await volume.commit.aio()

    with spindle.run(
        engine=engine,
        name=APP_NAME,
        requires_proxy_auth=True,
        proxy_secret=modal.Secret.from_name(proxy_secret),
        checkpoint_volume=APP_NAME + "-checkpoints",
        latest=spindle.Pool(min_containers=0, max_containers=1),
    ) as (url, api_key):
        os.environ["TINKER_BASE_URL"] = url
        os.environ["TINKER_API_KEY"] = api_key

        async def execute():
            async def create():
                header_key = (
                    "default_headers_overrides"
                    if "default_headers_overrides"
                    in inspect.signature(tinker.ServiceClient).parameters
                    else "default_headers"
                )
                service = tinker.ServiceClient(
                    base_url=url, api_key=api_key, **{header_key: proxy_headers()}
                )
                return await create_full_training_client_async(service, cfg.model)

            check_case = cases[1]
            if timing_only_from:
                if Path(
                    timing_only_from
                ).name != timing_only_from or timing_only_from in {".", ".."}:
                    raise ValueError("timing_only_from must name a prior check run")
                prior = json.loads(
                    (Path("/runs") / timing_only_from / "result.json").read_text()
                )
                for key in (
                    "model_revision",
                    "config",
                    "engine_training",
                    "trainer_gpu",
                    "backend_env",
                    "cases",
                ):
                    if digest(prior[key]) != digest(report[key]):
                        raise ValueError(f"Prior check inputs differ: {key}")
                if "reset_difference" not in prior["gpu_update_check"]:
                    raise ValueError("Prior GPU checks did not complete")
                report.update(
                    {
                        "timing_only_from": timing_only_from,
                        "check_commit": prior["commit"],
                        "numerical_equivalence_established": False,
                        "frozen_layout_check": prior["frozen_layout_check"],
                        "gpu_update_check": prior["gpu_update_check"],
                    }
                )
                await persist_async()
            else:
                updates, probes, changes, initial_probes = {}, {}, {}, {}
                for mode in ("batch", "stream"):
                    training = await create()
                    try:
                        if mode == "batch":
                            packed = await probe(
                                training, cfg, check_case, all_rows=True
                            )
                            repeated = await probe(
                                training, cfg, check_case, all_rows=True
                            )
                            grouped = await probe(
                                training, cfg, check_case, all_rows=True, groupwise=True
                            )
                            report["frozen_layout_check"] = {
                                "repeat_difference": probe_difference(packed, repeated),
                                "grouping_difference": probe_difference(
                                    packed, grouped
                                ),
                            }
                            print(
                                "FROZEN_LAYOUT_CHECK "
                                + json.dumps(report["frozen_layout_check"]),
                                flush=True,
                            )
                            await persist_async()
                        initial = await probe(training, cfg, check_case)
                        initial_probes[mode] = initial
                        updates[mode] = await trial(
                            training,
                            cfg,
                            check_case,
                            mode,
                            "all_ready",
                            learning_rate=1e-5,
                        )
                        probes[mode] = await probe(training, cfg, check_case)
                        changes[mode] = probe_difference(initial, probes[mode])
                        report["gpu_update_check"] = {
                            "learning_rate": 1e-5,
                            "updates": updates,
                            "policy_changes": changes,
                        }
                        print(
                            "GPU_UPDATE_PROGRESS "
                            + json.dumps(report["gpu_update_check"]),
                            flush=True,
                        )
                        await persist_async()
                        if (
                            changes[mode]["max_abs"] <= 1e-6
                            or updates[mode]["grad_norm"] <= 0
                        ):
                            raise RuntimeError(
                                "GPU correctness check must change the policy"
                            )
                    finally:
                        await release(training)
                difference = probe_difference(probes["batch"], probes["stream"])
                baseline_difference = probe_difference(
                    initial_probes["batch"], initial_probes["stream"]
                )
                report["gpu_update_check"].update(
                    {
                        "probe_difference": difference,
                        "reset_difference": baseline_difference,
                    }
                )
                await persist_async()
                if baseline_difference["max_abs"] > 1e-6:
                    raise RuntimeError(
                        f"Base policy reset differs: {baseline_difference}"
                    )
                if difference["max_abs"] > 1e-3 or difference["mean_abs"] > 1e-4:
                    raise RuntimeError(f"Post-update GPU probes differ: {difference}")
                if not math.isclose(
                    updates["batch"]["grad_norm"],
                    updates["stream"]["grad_norm"],
                    rel_tol=5e-4,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError("Nonzero-update GPU gradient norms differ")
                report["gpu_update_check"].update(
                    {
                        "learning_rate": 1e-5,
                        "updates": updates,
                        "probe_difference": difference,
                        "policy_changes": changes,
                    }
                )
                print(
                    "GPU_UPDATE_CHECK " + json.dumps(report["gpu_update_check"]),
                    flush=True,
                )
                await persist_async()
            training = await create()
            try:
                before = await probe(training, cfg, check_case)
                for _ in range(2):
                    for case in cases:
                        for mode in ("batch", "stream"):
                            report["warmup"].append(
                                await trial(training, cfg, case, mode, "all_ready")
                            )
                schedule = [
                    (case, profile, repeat)
                    for case in cases
                    for profile in ("all_ready", "recorded")
                    for repeat in range(pairs)
                ]
                random.Random(seed).shuffle(schedule)
                for case, profile, repeat in schedule:
                    modes = (
                        ("batch", "stream") if repeat % 2 == 0 else ("stream", "batch")
                    )
                    for mode in modes:
                        row = await trial(training, cfg, case, mode, profile)
                        row["pair_id"] = f"{case['id']}:{profile}:{repeat}"
                        report["trials"].append(row)
                        print("REPLAY_TRIAL " + json.dumps(row), flush=True)
                    await persist_async()
                frozen = probe_difference(
                    before, await probe(training, cfg, check_case)
                )
                if frozen["max_abs"] > 1e-6:
                    raise RuntimeError(f"Frozen policy probe changed: {frozen}")
                report["frozen_probe_difference"] = frozen
                report["summary"] = summarize(
                    report["trials"],
                    pairs,
                    [case["id"] for case in cases],
                    require_equivalence=not timing_only_from,
                )
            finally:
                await release(training)

        try:
            asyncio.run(execute())
        except BaseException as exc:
            report["error_type"] = type(exc).__name__
            persist()
            raise
    report["scope_closed"] = True
    persist()
    print("REPLAY_RESULT " + json.dumps(report["summary"]), flush=True)
    return json.dumps(report)
