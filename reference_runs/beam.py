"""Beam LoRA reference run and incremental result storage.

Run ``uv run python reference_runs/beam.py --help`` for configuration.
Install/run from the repository root:
  uv run --with transformers==5.15.1 --with matplotlib==3.11.2 python reference_runs/beam.py \
    --output reference_runs/results/beam --replay-cache /path/to/beam-replay \
    --deployment-provenance /path/to/validated-deployment.json --stop-after 10
Resume: add --resume CHECKPOINT_URI (the most recent saved checkpoint).
Set TINKER_BASE_URL and TINKER_API_KEY; optional MODAL_KEY and MODAL_SECRET
supply frontend authentication. --plan and --plot-only perform no remote work.
Metrics, configuration and evaluations in --output are intended for Git.
Large routing records live in --replay-cache, outside the checked-in results.
After update10, rebuild/restart the deployment and resume from the printed and journaled
checkpoint URI using the SAME output and cache paths, without --stop-after.
To audit restoration first, resume with --stop-after 10. This saves a roundtrip
checkpoint and exits before publication, evaluation, or training. Compare its
weights and optimizer state with the original checkpoint, then resume the new
checkpoint without --stop-after to continue.
The external deployment provenance JSON records exact Spindle/Miles/SGLang
revisions, resolved recipe and hardware; it must contain no credentials.

Protocol:32 groups x8 samples, 4096 output tokens, GRPO with the upstream
length penalty, one-update asynchronous behavior-policy lag, routing replay,
one-pass CISPO(0,2), rank16/alpha32 and the400-update cosine LR schedule.
The one-pass gradient matches single-update PPO with detached TIS at unchanged
scoring/backward weights; its reported loss value differs from PPO. TIS
statistics use tokens with nonzero advantages. The sampled log-ratio is a
behavior/current diagnostic, not KL against the frozen base model.

Budgets: median of the latest five accepted-update intervals <=900s after
four intervals;30h wall time including evaluation/restart. A breached budget
saves a checkpoint and exits for diagnosis. Evaluation writes each response
before waiting for other groups. Metrics/evaluations/config/summary/plots are
intended to be committed as they accumulate. Checkpoint URI records refer to
remote30-day states; keep replay cache and checkpoint storage independently.
The script never redeploys apps or claims exact optimizer restore identity;
compare checkpoint/roundtrip state independently during the restart test.
"""

import argparse
import asyncio
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import signal
import sys
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tinker
from huggingface_hub import hf_hub_download
from tinker import types
from transformers import PreTrainedTokenizerFast

import spindle
from spindle.replay import ReplaySequence


@dataclasses.dataclass(frozen=True)
class Config:
    model: str = "reflection/Beam-1.0-501B-A23B"
    model_revision: str = "12950ae640b48cbbe9b5bb7735bf197507d6383f"
    training_dataset: str = "zhuzilin/dapo-math-17k"
    training_revision: str = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
    evaluation_dataset: str = "zhuzilin/aime-2025"
    evaluation_revision: str = "3d2ffb246285920641112d03a9c23218db4de9b7"
    updates: int = 100
    groups: int = 32
    samples_per_prompt: int = 8
    response_tokens: int = 4096
    prompt_tokens: int = 1024
    seed: int = 1234
    evaluation_interval: int = 5
    checkpoint_interval: int = 10
    rank: int = 16
    alpha: int = 32


def learning_rate(step: int) -> float:
    if not 1 <= step <= 400:
        raise ValueError("Update must be in the reference schedule's range 1–400")
    if step <= 10:
        return 1e-4 * step / 10
    return 1e-5 + 9e-5 * 0.5 * (1 + math.cos(math.pi * (step - 10) / 390))


def read_records(path: Path) -> list[dict]:
    """Read committed lines; an interrupted final write is not a record."""
    if not path.exists():
        return []
    records = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object in {path}")
        records.append(record)
    return records


class Journal:
    """Exclusive writer with a durable record boundary after each append."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.deadline = None
        self.refresh = False
        self.mutex = threading.Lock()
        self.lock = (directory / ".writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(
                "Another reference-run writer owns this directory"
            ) from None

    def timeout(self, limit, *, checkpoint=False):
        if self.deadline is None:
            return limit
        remaining = self.deadline - time.time() - (0 if checkpoint else 1800)
        if remaining <= 0:
            raise TimeoutError(
                "Run time budget exhausted; checkpoint/recovery required"
            )
        return min(limit, remaining)

    def close(self):
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            if self.refresh:
                try:
                    plot(self.directory, journal=self)
                except Exception as error:
                    if exc_type is None:
                        raise
                    print(
                        f"Artifact refresh failed: {type(error).__name__}",
                        file=sys.stderr,
                    )
        finally:
            self.close()

    def append(self, filename: str, record: dict):
        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise ValueError("Journal filename must be a local .jsonl file")
        encoded = json.dumps(record, allow_nan=False).encode() + b"\n"
        path = self.directory / filename
        with self.mutex, path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end:
                stream.seek(end - 1)
                if stream.read(1) != b"\n":
                    position = end
                    while position:
                        start = max(0, position - 8192)
                        stream.seek(start)
                        chunk = stream.read(position - start)
                        boundary = chunk.rfind(b"\n")
                        if boundary >= 0:
                            stream.truncate(start + boundary + 1)
                            break
                        position = start
                    else:
                        stream.truncate(0)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def event(self, event: str, **values):
        self.append("metrics.jsonl", {"time": time.time(), "event": event, **values})


def grade(text, label):
    # Matches the reference's strict boxed-answer grader, including its suffix limit.
    text = text[-100:]
    start = text.rfind("\\boxed{")
    if start < 0:
        return 0.0
    depth = 1
    for end in range(start + 7, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return float(text[start + 7 : end] == label)
    return 0.0


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_data(config, tokenizer):
    def rows(repo, revision):
        filename = repo.split("/")[-1] + ".jsonl"
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    train = rows(config.training_dataset, config.training_revision)
    random.Random(config.seed).shuffle(train)
    evaluation = rows(config.evaluation_dataset, config.evaluation_revision)
    for row in evaluation:
        row["prompt"][0]["content"] = (
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
            + row["prompt"][0]["content"]
            + '\n\nRemember to put your answer on its own line after "Answer:".'
        )

    def encode(row):
        prompt = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        return row, prompt if isinstance(prompt, list) else prompt["input_ids"]

    train = [encode(row) for row in train]
    return [
        (row, prompt) for row, prompt in train if len(prompt) <= config.prompt_tokens
    ], [encode(row) for row in evaluation]


def sample_group(sampler, item, config, tokenizer, replay, timeout=3600):
    row, prompt = item
    params = types.SamplingParams(
        max_tokens=config.response_tokens,
        temperature=1.0,
        top_p=1.0,
        seed=config.seed,
        stop=[tokenizer.eos_token_id],
    )
    if replay:
        future = spindle.sample_with_replay(
            sampler,
            types.ModelInput.from_ints(prompt),
            sampling_params=params,
            num_samples=config.samples_per_prompt,
            return_routed_experts=True,
        )
    else:
        future = sampler.sample(
            types.ModelInput.from_ints(prompt), config.samples_per_prompt, params
        )
    sequences = future.result(timeout=timeout).sequences
    if len(sequences) != config.samples_per_prompt:
        raise ValueError("Incomplete sampling group")
    rewards = [grade(tokenizer.decode(seq.tokens), row["label"]) for seq in sequences]
    return sequences, rewards


def evaluate(sampler, rows, step, config, tokenizer, journal):
    filename = f"aime25-step-{step:04d}.jsonl"
    existing = read_records(journal.directory / filename)
    indices = {(r["problem"], r["sample"]): r for r in existing}
    expected = {
        (i, j) for i in range(len(rows)) for j in range(config.samples_per_prompt)
    }
    if len(indices) != len(existing) or not indices.keys() <= expected:
        raise ValueError("Duplicate or out-of-range evaluation response")
    started = time.monotonic()

    def group(index_item):
        index, item = index_item
        if all((index, j) in indices for j in range(config.samples_per_prompt)):
            return
        sequences, rewards = sample_group(
            sampler, item, config, tokenizer, False, timeout=journal.timeout(3600)
        )
        for j, (sequence, reward) in enumerate(zip(sequences, rewards, strict=True)):
            if (index, j) not in indices:
                journal.append(
                    filename,
                    dict(
                        step=step,
                        problem=index,
                        sample=j,
                        prompt=item[0]["prompt"],
                        label=item[0]["label"],
                        text=tokenizer.decode(sequence.tokens),
                        tokens=len(sequence.tokens),
                        correct=reward,
                        stop_reason=sequence.stop_reason,
                    ),
                )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(group, enumerate(rows)))
    records = read_records(journal.directory / filename)
    if len(records) != len(rows) * config.samples_per_prompt:
        raise ValueError("Evaluation coverage mismatch")
    journal.event(
        "evaluation",
        step=step,
        accuracy=statistics.mean(r["correct"] for r in records),
        mean_tokens=statistics.mean(r["tokens"] for r in records),
        capped=sum(r["stop_reason"] == "length" for r in records),
        completions=len(records),
        seconds=time.monotonic() - started,
    )


def capture(sampler, rows, step, policy, config, tokenizer, journal, cache):
    started = time.monotonic()
    directory = cache / f"step-{step:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["policy"] != policy:
            raise ValueError("Cached rollout has different behavior weights")
        for name, digest in manifest["sha256"].items():
            if (
                Path(name).name != name
                or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
            ):
                raise ValueError("Cached rollout checksum mismatch")
        return manifest

    def group(index_item):
        index, item = index_item
        path = directory / f"group-{index:02d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["policy"] != policy or saved["prompt"] != item[1]:
                raise ValueError("Partial rollout provenance mismatch")
            return saved
        sequences, rewards = sample_group(
            sampler, item, config, tokenizer, True, timeout=journal.timeout(3600)
        )
        saved = dict(
            policy=policy,
            prompt=item[1],
            rewards=rewards,
            sequences=[s.model_dump() for s in sequences],
        )
        atomic_json(path, saved)
        journal.event(
            "rollout_group",
            step=step,
            group=index,
            behavior_policy_step=policy,
            rewards=rewards,
            lengths=[len(s.tokens) for s in sequences],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return saved

    with ThreadPoolExecutor(max_workers=8) as pool:
        groups = list(pool.map(group, enumerate(rows)))
    manifest = dict(
        step=step,
        policy=policy,
        groups=len(groups),
        seconds=time.monotonic() - started,
        sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.glob("group-*.json")
        },
    )
    atomic_json(manifest_path, manifest)
    journal.event("rollout", **manifest)
    return manifest


def build_datums(cache, step, config):
    groups = [
        json.loads(path.read_text())
        for path in sorted((cache / f"step-{step:04d}").glob("group-*.json"))
    ]
    if len(groups) != config.groups:
        raise ValueError("Incomplete training batch")
    total = sum(len(s["tokens"]) for g in groups for s in g["sequences"])
    datums = []
    for group in groups:
        sequences = [ReplaySequence.model_validate(s) for s in group["sequences"]]
        scores = np.array(
            [
                2 * r - 1 + min(-(len(s.tokens) - 3277) / 819, 0.0)
                for s, r in zip(sequences, group["rewards"], strict=True)
            ]
        )
        advantages = (scores - scores.mean()) / (scores.std(ddof=1) + 1e-6)
        for sequence, advantage in zip(sequences, advantages, strict=True):
            if (
                not sequence.tokens
                or sequence.logprobs is None
                or not all(map(math.isfinite, sequence.logprobs))
            ):
                raise ValueError("Invalid rollout logprobs/tokens")
            if sequence.replay.routed_experts is None:
                raise ValueError("Beam training requires captured expert routes")
            full = group["prompt"] + sequence.tokens
            padding = len(group["prompt"]) - 1
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": full[1:],
                        "logprobs": [0.0] * padding + sequence.logprobs,
                        "advantages": [0.0] * padding
                        + [float(advantage) / total] * len(sequence.tokens),
                        **sequence.replay.training_inputs(
                            len(sequence.tokens), num_layers=52, experts_per_token=8
                        ),
                    },
                )
            )
    lengths = [len(s["tokens"]) for g in groups for s in g["sequences"]]
    return datums, dict(
        generated_tokens=total,
        mean_tokens=statistics.mean(lengths),
        accuracy=statistics.mean(r for g in groups for r in g["rewards"]),
        token_limit_fraction=statistics.mean(
            n >= config.response_tokens for n in lengths
        ),
    )


def train(training, datums, step, journal, **batch_metrics):
    journal.timeout(1)
    started = time.monotonic()
    journal.event("update_started", step=step)
    result = training.forward_backward(
        datums,
        "cispo",
        loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 2.0},
    ).result(timeout=journal.timeout(7200))
    journal.event(
        "backward",
        step=step,
        seconds=time.monotonic() - started,
        metrics=result.metrics,
    )
    deltas = []
    for datum, output in zip(datums, result.loss_fn_outputs, strict=True):
        active = datum.loss_fn_inputs["advantages"].to_numpy() != 0
        delta = (
            output["logprobs"].to_numpy() - datum.loss_fn_inputs["logprobs"].to_numpy()
        )
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite scoring logprobs; optimizer was not submitted")
        deltas.extend(delta[active].tolist())
    optimizer_started = time.monotonic()
    journal.timeout(1)
    journal.event("optimizer_submitted", step=step)
    optimizer = training.optim_step(
        types.AdamParams(
            learning_rate=learning_rate(step),
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            grad_clip_norm=1.0,
        )
    ).result(timeout=journal.timeout(1800))
    journal.event(
        "optimizer_result",
        step=step,
        metrics=optimizer.metrics,
        seconds=time.monotonic() - optimizer_started,
    )
    if optimizer.metrics.get("update_successful:mean") != 1.0:
        raise RuntimeError("Optimizer update rejected")
    delta = np.array(deltas)
    journal.event(
        "update",
        step=step,
        **batch_metrics,
        learning_rate=learning_rate(step),
        seconds=time.monotonic() - started,
        metrics=result.metrics,
        optimizer=optimizer.metrics,
        active_tokens=len(deltas),
        tis_mean=float(np.exp(np.minimum(delta, math.log(2))).mean())
        if len(delta)
        else None,
        tis_clip_fraction=float((delta > math.log(2)).mean()) if len(delta) else None,
        behavior_to_current_sampled_log_ratio_mean=float((-delta).mean())
        if len(delta)
        else None,
    )


def optimizer_outcome(events, step):
    outcome = None
    for event in events:
        if event.get("step") != step:
            continue
        if event["event"] == "optimizer_submitted":
            outcome = None
        elif event["event"] == "optimizer_result":
            outcome = event["metrics"].get("update_successful:mean")
    return outcome


def run(args, config):
    if args.replay_cache.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("Replay cache must be outside checked-in results")
    with Journal(args.output) as journal, Journal(args.replay_cache):
        journal.refresh = True
        config_path = args.output / "config.json"
        if config_path.exists() and json.loads(
            config_path.read_text()
        ) != dataclasses.asdict(config):
            raise ValueError("Recorded configuration differs from this invocation")
        events = read_records(args.output / "metrics.jsonl")
        checkpoints = [r for r in events if r["event"] == "checkpoint"]
        updates = [r for r in events if r["event"] == "update"]
        if not args.resume and (
            checkpoints
            or updates
            or any(r["event"] == "optimizer_submitted" for r in events)
        ):
            raise ValueError(
                "Existing training state requires an explicit checkpoint resume"
            )
        completed = 0
        if args.resume:
            if not checkpoints or checkpoints[-1]["path"] != args.resume:
                raise ValueError("Resume must use the latest recorded checkpoint")
            completed = checkpoints[-1]["step"]
            if (
                getattr(args, "stop_after", None) is not None
                and args.stop_after < completed
            ):
                raise ValueError(
                    "--stop-after precedes the restored checkpoint's update"
                )
            if updates and updates[-1]["step"] > completed:
                raise ValueError(
                    "Updates after checkpoint require explicit recovery; refusing silent rollback"
                )
            for event in events:
                if (
                    event["event"] == "optimizer_submitted"
                    and event["step"] > completed
                ):
                    if optimizer_outcome(events, event["step"]) != 0:
                        raise ValueError(
                            "Uncheckpointed update outcome requires explicit recovery"
                        )
        atomic_json(config_path, dataclasses.asdict(config))
        args.replay_cache.mkdir(parents=True, exist_ok=True)
        identity_path = args.output / "identity.json"
        if not identity_path.exists():
            atomic_json(
                identity_path, {"run_id": uuid.uuid4().hex, "started": time.time()}
            )
        identity = json.loads(identity_path.read_text())
        journal.deadline = identity["started"] + 30 * 3600
        journal.timeout(1)
        cache_identity = args.replay_cache / "identity.json"
        if (
            cache_identity.exists()
            and json.loads(cache_identity.read_text()) != identity
        ):
            raise ValueError("Replay cache belongs to another run")
        atomic_json(cache_identity, identity)
        journal.event(
            "segment",
            segment=uuid.uuid4().hex,
            resume_step=completed,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            spindle_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            packages={
                name: importlib.metadata.version(name)
                for name in ["tinker", "numpy", "huggingface-hub", "transformers"]
            },
            deployment=json.loads(args.deployment_provenance.read_text()),
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            config.model, revision=config.model_revision
        )
        train_rows, eval_rows = load_data(config, tokenizer)
        if len(train_rows) < config.updates * config.groups or len(eval_rows) != 30:
            raise ValueError("Dataset coverage differs from reference")
        journal.event(
            "datasets", train_prompts=len(train_rows), eval_prompts=len(eval_rows)
        )

        headers = {
            name: os.environ[var]
            for name, var in [
                ("Modal-Key", "MODAL_KEY"),
                ("Modal-Secret", "MODAL_SECRET"),
            ]
            if var in os.environ
        }
        service = tinker.ServiceClient(
            base_url=os.environ["TINKER_BASE_URL"],
            api_key=os.environ["TINKER_API_KEY"],
            default_headers_overrides=headers,
        )

        async def create_training():
            operation = (
                service.create_training_client_from_state_with_optimizer_async(
                    args.resume
                )
                if args.resume
                else service.create_lora_training_client_async(
                    base_model=config.model,
                    rank=config.rank,
                    seed=config.seed,
                    train_unembed=False,
                )
            )
            return await asyncio.wait_for(operation, timeout=journal.timeout(7200))

        training = asyncio.run(create_training())
        pending_descriptor = (
            checkpoints[-1].get("pending_rollout") if args.resume else None
        )

        def checkpoint(step):
            started = time.monotonic()
            result = training.save_state(
                f"beam-step-{step:04d}-{uuid.uuid4().hex[:8]}", ttl_seconds=30 * 86400
            ).result(timeout=journal.timeout(1800, checkpoint=True))
            journal.event(
                "checkpoint",
                step=step,
                path=result.path,
                pending_rollout=pending_descriptor,
                seconds=time.monotonic() - started,
            )
            print(f"Checkpoint update{step}: {result.path}", flush=True)
            plot(args.output, journal=journal)
            return result.path

        def sampling_client(path):
            async def create_sampling():
                return await asyncio.wait_for(
                    service.create_sampling_client_async(model_path=path),
                    timeout=journal.timeout(1800),
                )

            return asyncio.run(create_sampling())

        def publish(step):
            started = time.monotonic()
            journal.timeout(1)
            result = training.save_weights_for_sampler(
                f"beam-policy-{step:04d}-{uuid.uuid4().hex[:8]}", ttl_seconds=30 * 86400
            ).result(timeout=journal.timeout(1800))
            sampler = sampling_client(result.path)
            journal.event(
                "publication",
                step=step,
                path=result.path,
                seconds=time.monotonic() - started,
            )
            return sampler, result.path

        if args.resume and args.stop_after == completed:
            path = checkpoint(completed)
            journal.event(
                "restore_roundtrip", step=completed, reference=args.resume, path=path
            )
            return
        if not args.resume:
            checkpoint(0)
        sampler, sampler_path = publish(completed)
        if completed % config.evaluation_interval == 0:
            evaluate(sampler, eval_rows, completed, config, tokenizer, journal)
            plot(args.output, journal=journal)
        if completed == config.updates:
            journal.event("completed", step=completed)
            plot(args.output, journal=journal)
            return

        def capture_step(sampling, step, policy):
            journal.timeout(1)
            offset = (step - 1) * config.groups
            return capture(
                sampling,
                train_rows[offset : offset + config.groups],
                step,
                policy,
                config,
                tokenizer,
                journal,
                args.replay_cache,
            )

        first = completed + 1
        if args.resume and completed > 0 and not pending_descriptor:
            raise ValueError(
                "Checkpoint lacks durable pending-rollout behavior identity"
            )
        if pending_descriptor:
            if pending_descriptor["step"] != first:
                raise ValueError("Pending batch step does not match checkpoint")
            behavior = sampling_client(pending_descriptor["sampler_path"])
            manifest = capture_step(behavior, first, pending_descriptor["policy"])
        else:
            manifest = capture_step(sampler, first, completed)
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            for step in range(first, config.updates + 1):
                cycle = time.monotonic()
                current_descriptor = dict(
                    step=step,
                    policy=manifest["policy"],
                    sampler_path=pending_descriptor["sampler_path"]
                    if pending_descriptor
                    else sampler_path,
                )
                pending_descriptor = (
                    dict(step=step + 1, policy=step - 1, sampler_path=sampler_path)
                    if step < config.updates
                    else None
                )
                pending = (
                    prefetch.submit(capture_step, sampler, step + 1, step - 1)
                    if pending_descriptor
                    else None
                )
                try:
                    datums, batch_metrics = build_datums(
                        args.replay_cache, step, config
                    )
                    train(
                        training,
                        datums,
                        step,
                        journal,
                        behavior_policy_step=manifest["policy"],
                        **batch_metrics,
                    )
                except Exception:
                    latest = read_records(args.output / "metrics.jsonl")
                    submitted = any(
                        r["event"] == "optimizer_submitted" and r["step"] == step
                        for r in latest
                    )
                    rejected = optimizer_outcome(latest, step) == 0
                    if not submitted or rejected:
                        pending_descriptor = current_descriptor
                        journal.event(
                            "update_rejected"
                            if rejected
                            else "backward_failed_before_optimizer",
                            step=step,
                        )
                        checkpoint(step - 1)
                    raise
                del datums
                accepted = [
                    r
                    for r in read_records(args.output / "metrics.jsonl")
                    if r["event"] == "update"
                ]
                intervals = [
                    b["time"] - a["time"] for a, b in zip(accepted, accepted[1:])
                ]
                slow = len(intervals) >= 4 and statistics.median(intervals[-5:]) > 900
                stop = step == args.stop_after or slow
                saved = (
                    step % config.checkpoint_interval == 0
                    or step == config.updates
                    or stop
                )
                if saved:
                    checkpoint(step)
                try:
                    following = (
                        pending.result(timeout=journal.timeout(7200))
                        if pending
                        else None
                    )
                except Exception:
                    journal.event("prefetch_failed", step=step)
                    if not saved:
                        checkpoint(step)
                    raise
                journal.event(
                    "cycle",
                    step=step,
                    seconds=time.monotonic() - cycle,
                    behavior_policy_step=manifest["policy"],
                )
                if stop:
                    journal.event(
                        "stopped",
                        step=step,
                        reason="runtime_budget" if slow else "restart_boundary",
                    )
                    plot(args.output, journal=journal)
                    return
                try:
                    sampler, sampler_path = publish(step)
                    if step % config.evaluation_interval == 0:
                        evaluate(sampler, eval_rows, step, config, tokenizer, journal)
                        plot(args.output, journal=journal)
                except Exception as error:
                    if not saved:
                        checkpoint(step)
                    journal.event("stopped", step=step, reason=type(error).__name__)
                    raise
                manifest = following
        journal.event("completed", step=config.updates)
        plot(args.output, journal=journal)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument(
        "--deployment-provenance",
        type=Path,
        help="JSON containing validated runtime revisions, configuration and hardware (no secrets)",
    )
    parser.add_argument(
        "--resume", help="Latest recorded checkpoint URI; restores optimizer too"
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Save and exit at this accepted update for deployment restart testing",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Summarize and plot existing partial/final output without remote calls",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = Config()
    if args.plan:
        print(json.dumps(dataclasses.asdict(config), indent=2))
        return
    if args.plot_only:
        if args.output is None:
            parser.error("--output is required for plotting")
        plot(args.output)
        return
    if not all([args.output, args.replay_cache, args.deployment_provenance]):
        parser.error(
            "--output, --replay-cache and --deployment-provenance are required"
        )
    if args.stop_after is not None and not 1 <= args.stop_after <= config.updates:
        parser.error("--stop-after must be between 1 and100")
    if not args.worker:
        identity = args.output / "identity.json"
        started = (
            json.loads(identity.read_text())["started"]
            if identity.exists()
            else time.time()
        )
        seconds = started + 30 * 3600 - time.time()
        if seconds <= 0:
            parser.error("The original run's30-hour budget is exhausted")
        code = supervise(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
            seconds,
        )
        if code == 124:
            print(
                "Reference run reached its deadline; regenerate partial plots with --plot-only.",
                file=sys.stderr,
            )
        raise SystemExit(code)
    run(args, config)


def supervise(command, seconds, termination_grace=30):
    deadline = time.monotonic() + seconds
    process = subprocess.Popen(command, start_new_session=True)

    def kill_group(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def terminate():
        cleanup_deadline = min(deadline, time.monotonic() + termination_grace)
        kill_group(signal.SIGTERM)
        try:
            process.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        finally:
            kill_group(signal.SIGKILL)
        try:
            process.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        try:
            return process.wait(timeout=max(0, seconds - termination_grace))
        except subprocess.TimeoutExpired:
            terminate()
            return 124
        except BaseException:
            terminate()
            raise
    finally:
        kill_group(signal.SIGKILL)
        signal.signal(signal.SIGTERM, previous)


def summarize(directory):
    evaluations = []
    records_by_step = {}
    for path in sorted(directory.glob("aime25-step-*.jsonl")):
        records = read_records(path)
        if not records:
            continue
        step = int(path.stem.rsplit("-", 1)[1])
        records_by_step[step] = records
        evaluations.append(
            dict(
                step=step,
                completions=len(records),
                expected_completions=240,
                complete=len(records) == 240
                and {(r["problem"], r["sample"]) for r in records}
                == {(p, s) for p in range(30) for s in range(8)},
                accuracy=statistics.mean(r["correct"] for r in records),
                mean_tokens=statistics.mean(r["tokens"] for r in records),
            )
        )
    improvement = None
    if 0 in records_by_step and 100 in records_by_step:
        before, after = records_by_step[0], records_by_step[100]
        expected = {(p, s) for p in range(30) for s in range(8)}
        if {(r["problem"], r["sample"]) for r in before} == expected == {
            (r["problem"], r["sample"]) for r in after
        } and len(before) == len(after) == 240:
            delta = np.array(
                [
                    statistics.mean(r["correct"] for r in after if r["problem"] == p)
                    - statistics.mean(r["correct"] for r in before if r["problem"] == p)
                    for p in range(30)
                ]
            )
            rng = np.random.default_rng(1234)
            draws = delta[rng.integers(0, 30, size=(20000, 30))].mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975]).tolist()
            improvement = dict(
                accuracy_delta=float(delta.mean()),
                paired_problem_bootstrap_95=[lo, hi],
                learning_demonstrated=lo > 0,
                draws=20000,
                seed=1234,
            )
    events = read_records(directory / "metrics.jsonl")
    updates = [r for r in events if r["event"] == "update"]
    intervals = [b["time"] - a["time"] for a, b in zip(updates, updates[1:])]
    identity_path = directory / "identity.json"
    elapsed = None
    if identity_path.exists():
        end = next(
            (r["time"] for r in reversed(events) if r["event"] == "completed"),
            time.time(),
        )
        elapsed = end - json.loads(identity_path.read_text())["started"]
    median = statistics.median(intervals) if intervals else None
    return dict(
        evaluations=evaluations,
        final_comparison=improvement,
        accepted_updates=len(updates),
        median_seconds_between_updates=median,
        elapsed_seconds=elapsed,
        update_time_target_met=median <= 900 if median is not None else None,
        wall_time_budget_met=elapsed <= 30 * 3600 if elapsed is not None else None,
        learning_demonstrated=improvement["learning_demonstrated"]
        if improvement
        else None,
        horizon_complete=[r["step"] for r in updates] == list(range(1, 101))
        and bool(improvement),
    )


def plot(directory, journal=None):
    if journal is None:
        with Journal(directory) as locked:
            return plot(directory, journal=locked)
    # Plotting is optional for CPU artifact tests and --plan.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not directory.exists():
        return
    summary = summarize(directory)
    atomic_json(directory / "summary.json", summary)
    reference = json.loads(
        Path(__file__).with_name("beam_blog_reference.json").read_text()
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, field, reference_field, label, scale in [
        (axes[0], "accuracy", "accuracy_percent", "AIME25 accuracy (%)", 100),
        (axes[1], "mean_tokens", "mean_tokens", "Mean response tokens", 1),
    ]:
        rows = summary["evaluations"]
        for complete, marker in [(True, "o"), (False, "x")]:
            values = [r for r in rows if r["complete"] == complete]
            ax.plot(
                [r["step"] for r in values],
                [r[field] * scale for r in values],
                marker=marker,
                label="Reference run" if complete else "Partial evaluation",
                color="#176b9b",
            )
        series = reference[reference_field]
        ax.plot(
            [p[0] for p in series],
            [p[1] for p in series],
            label="Blog (approximate digitization)",
            color="#79934d",
        )
        ax.set_ylabel(label)
        ax.set_xlabel("Reported step; blog axis is not verified as optimizer updates")
        ax.legend(fontsize=8)
    events = read_records(directory / "metrics.jsonl")
    for event, label in [
        ("cycle", "Training + sampling join"),
        ("update", "Trainer API wall time"),
    ]:
        values = [r for r in events if r["event"] == event]
        axes[2].plot(
            [r["step"] for r in values],
            [r["seconds"] for r in values],
            ".-",
            label=label,
        )
    accepted = [r for r in events if r["event"] == "update"]
    axes[2].plot(
        [r["step"] for r in accepted[1:]],
        [b["time"] - a["time"] for a, b in zip(accepted, accepted[1:])],
        ".-",
        label="Between accepted updates (includes evaluation/publication)",
    )
    axes[2].axhline(900, color="gray", linestyle="--", label="15-minute target")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xlabel("Optimizer update")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Beam LoRA reference run — {summary['accepted_updates']}/100 updates")
    fig.tight_layout()
    fig.savefig(directory / "learning.tmp.png", dpi=160)
    (directory / "learning.tmp.png").replace(directory / "learning.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
