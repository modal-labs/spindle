"""GRPO on DAPO-Math through the stock Tinker SDK.

Commands, the record format and the restart rules are in reference_runs/README.md.
"""

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tinker
from huggingface_hub import hf_hub_download
from tinker import types
from transformers import AutoTokenizer

REQUEST_TIMEOUT_S = 3600
UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class RunConfig:
    model: str = "Qwen/Qwen3.5-4B"
    tokenizer_revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    enable_thinking: bool = False
    dataset: str = "zhuzilin/dapo-math-17k"
    dataset_revision: str = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
    seed: int = 1234
    max_prompt_tokens: int = 1024
    max_response_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    updates: int = 30
    groups: int = 32
    samples_per_group: int = 8
    eval_problems: int = 128
    eval_samples: int = 2
    evaluation_interval: int = 5
    checkpoint_interval: int = 10
    lora_rank: int = 32
    loss: str = "importance_sampling"
    learning_rate: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


SMOKE = {
    "updates": 3,
    "groups": 4,
    "eval_problems": 8,
    "eval_samples": 1,
    "evaluation_interval": 1,
    "checkpoint_interval": 1,
}

PROTOCOL = {
    "selection": (
        "Shuffle the dataset rows with the seed. The first eval_problems rows whose "
        "prompt fits max_prompt_tokens are held out for evaluation. Update n trains "
        "on the next `groups` remaining rows, starting at (n - 1) * groups."
    ),
    "prompt": "The dataset's chat messages through the tokenizer's chat template.",
    "reward": "1 if the last \\boxed{...} in the response equals the label, else 0.",
    "advantage": (
        "Reward minus the mean reward of the prompt's samples, on every response "
        "token. Prompt tokens have zero advantage."
    ),
    "sampling_seed": "seed + dataset row index, for training and evaluation.",
    "evaluation": (
        "eval_samples responses per held-out problem with the training sampling "
        "settings. Accuracy is the mean reward over all responses."
    ),
}


def read_records(path: Path) -> list[dict]:
    """Complete records in a metrics file; an unterminated last line is not one."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_bytes().splitlines(keepends=True)
        if line.endswith(b"\n")
    ]


def optimizer_outcomes(records: list[dict]) -> dict[int, str]:
    """Each submitted optimizer request as accepted, rejected or unknown."""
    outcomes = {}
    for record in records:
        if record["event"] == "optimizer_submitted":
            outcomes[record["step"]] = UNKNOWN
        elif record["event"] == "optimizer_result":
            outcomes[record["step"]] = "accepted" if record["accepted"] else "rejected"
    return outcomes


class Journal:
    """The only writer of a run directory. Every record is durable when written."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "metrics.jsonl"
        self.lock = (directory / ".writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise SystemExit(f"Another process is writing to {directory}") from None
        if self.path.exists():
            data = self.path.read_bytes()
            complete = data.rfind(b"\n") + 1
            if complete != len(data):
                with self.path.open("r+b") as stream:
                    stream.truncate(complete)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.lock.close()

    def event(self, event: str, **values) -> dict:
        record = {"time": time.time(), "event": event, **values}
        with self.path.open("ab") as stream:
            stream.write(json.dumps(record, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def records(self) -> list[dict]:
        return read_records(self.path)


def write_json(path: Path, value, indent=None) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=indent, allow_nan=False) + "\n")
    temporary.replace(path)


def recorded_config(output: Path) -> RunConfig:
    path = output / "config.json"
    if not path.exists():
        raise SystemExit(f"{output} has no run to resume")
    return RunConfig(**json.loads(path.read_text())["config"])


def grade(text: str, label: str) -> float:
    start = text.rfind("\\boxed{")
    if start < 0:
        return 0.0
    depth = 1
    for end in range(start + 7, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return float(text[start + 7 : end].strip() == label.strip())
    return 0.0


def load_rows(config: RunConfig) -> list[dict]:
    path = hf_hub_download(
        config.dataset,
        config.dataset.split("/")[-1] + ".jsonl",
        repo_type="dataset",
        revision=config.dataset_revision,
    )
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def select(config: RunConfig, rows: list[dict], tokenizer) -> tuple[list, list]:
    """Held-out evaluation problems and the ordered training problems."""
    problems = []
    for index in random.Random(config.seed).sample(range(len(rows)), len(rows)):
        prompt = tokenizer.apply_chat_template(
            rows[index]["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
            return_dict=False,
        )
        if len(prompt) <= config.max_prompt_tokens:
            problems.append(
                {"id": index, "prompt": list(prompt), "label": rows[index]["label"]}
            )
        if len(problems) == config.eval_problems + config.updates * config.groups:
            break
    else:
        raise SystemExit("The dataset has too few problems for this configuration")
    return problems[: config.eval_problems], problems[config.eval_problems :]


def sample(sampler, problems, count, config, tokenizer) -> list[dict]:
    """Sampled responses per problem, graded. Requests run concurrently."""
    stop = list(
        dict.fromkeys(
            [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        )
    )

    def one(problem):
        sequences = (
            sampler.sample(
                types.ModelInput.from_ints(problem["prompt"]),
                count,
                types.SamplingParams(
                    max_tokens=config.max_response_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed + problem["id"],
                    stop=stop,
                ),
            )
            .result(timeout=REQUEST_TIMEOUT_S)
            .sequences
        )
        if len(sequences) != count:
            raise RuntimeError("The sampler returned too few sequences")
        samples = []
        for sequence in sequences:
            tokens, logprobs = list(sequence.tokens), list(sequence.logprobs or [])
            if (
                not tokens
                or len(logprobs) != len(tokens)
                or not all(map(math.isfinite, logprobs))
            ):
                raise RuntimeError("A sampled response lacks tokens or logprobs")
            samples.append(
                {
                    "tokens": tokens,
                    "logprobs": logprobs,
                    "truncated": sequence.stop_reason == "length",
                    "reward": grade(tokenizer.decode(tokens), problem["label"]),
                }
            )
        return {**problem, "samples": samples}

    with ThreadPoolExecutor(len(problems)) as pool:
        return list(pool.map(one, problems))


def response_stats(groups: list[dict]) -> dict:
    samples = [s for group in groups for s in group["samples"]]
    lengths = [len(s["tokens"]) for s in samples]
    return {
        "responses": len(samples),
        "reward": statistics.mean(s["reward"] for s in samples),
        "response_tokens": sum(lengths),
        "response_tokens_mean": statistics.mean(lengths),
        "response_tokens_max": max(lengths),
        "truncated_fraction": statistics.mean(s["truncated"] for s in samples),
    }


def build_datums(groups: list[dict]) -> tuple[list[types.Datum], list[np.ndarray]]:
    """Training data and, per datum, the sampler's logprobs of the response."""
    datums, behavior = [], []
    for group in groups:
        mean_reward = statistics.mean(s["reward"] for s in group["samples"])
        prefix = len(group["prompt"]) - 1
        for s in group["samples"]:
            tokens = group["prompt"] + s["tokens"]
            advantage = s["reward"] - mean_reward
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tokens[1:],
                        "logprobs": [0.0] * prefix + s["logprobs"],
                        "advantages": [0.0] * prefix + [advantage] * len(s["tokens"]),
                    },
                )
            )
            behavior.append(np.array(s["logprobs"]))
    return datums, behavior


def divergence(outputs, behavior: list[np.ndarray]) -> dict:
    """Trainer-versus-sampler diagnostics, when the service returns logprobs."""
    if len(outputs) != len(behavior) or any("logprobs" not in o for o in outputs):
        return {}
    ratios = np.concatenate(
        [
            o["logprobs"].to_numpy()[-len(b) :] - b
            for o, b in zip(outputs, behavior, strict=True)
        ]
    )
    if not np.isfinite(ratios).all():
        return {}
    return {
        "sampled_log_ratio_mean": float(ratios.mean()),
        "importance_weight_mean": float(np.exp(ratios).mean()),
    }


def proxy_headers() -> dict[str, str]:
    names = {
        "Modal-Key": "MODAL_PROXY_TOKEN_ID",
        "Modal-Secret": "MODAL_PROXY_TOKEN_SECRET",
    }
    return {h: os.environ[v] for h, v in names.items() if v in os.environ}


def connect() -> tinker.ServiceClient:
    headers = proxy_headers()
    option = (
        "default_headers_overrides"
        if "default_headers_overrides"
        in inspect.signature(tinker.ServiceClient).parameters
        else "default_headers"
    )
    return tinker.ServiceClient(**({option: headers} if headers else {}))


def served_revisions() -> dict:
    """Server-reported revisions. Services that report none are recorded as unknown."""
    served = {}
    base_url = os.environ.get("TINKER_BASE_URL")
    if base_url:
        request = urllib.request.Request(
            base_url.rstrip("/").removesuffix("/api/v1")
            + "/api/v1/get_server_capabilities",
            headers={"x-api-key": os.environ["TINKER_API_KEY"], **proxy_headers()},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                served = json.load(response)
        except (OSError, ValueError):
            pass
    return {k: served.get(k) or UNKNOWN for k in ["spindle_version", "spindle_commit"]}


def client_revision() -> str:
    def git(*command):
        return subprocess.run(
            ["git", *command],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return git("rev-parse", "HEAD") + (
            "-dirty" if git("status", "--porcelain", "--untracked-files=no") else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN


def run(
    config: RunConfig,
    output: Path,
    *,
    service,
    tokenizer,
    rows: list[dict],
    resume: bool = False,
    stop_after: int | None = None,
    deployment: dict | None = None,
) -> None:
    config_path = output / "config.json"
    if resume:
        if config != recorded_config(output):
            raise SystemExit("--resume uses the recorded configuration")
    elif config_path.exists() or (output / "metrics.jsonl").exists():
        raise SystemExit(f"{output} already holds a run; pass --resume or a new path")
    if stop_after is not None and not 0 <= stop_after <= config.updates:
        raise SystemExit("--stop-after is outside the configured updates")

    with Journal(output) as journal:
        local = output / "local"
        local.mkdir(exist_ok=True)
        completed, checkpoint = 0, None
        if resume:
            records = journal.records()
            checkpoints = [r for r in records if r["event"] == "checkpoint"]
            if not checkpoints:
                raise SystemExit("The run saved no checkpoint to resume from")
            checkpoint = checkpoints[-1]
            completed = checkpoint["step"]
            if stop_after is not None and stop_after < completed:
                raise SystemExit("--stop-after precedes the latest checkpoint")
            lost = {
                step: outcome
                for step, outcome in optimizer_outcomes(records).items()
                if step > completed and outcome != "rejected"
            }
            if lost:
                raise SystemExit(
                    f"Checkpoint {completed} is older than optimizer requests {lost}, "
                    "so this run cannot continue. Start a new run."
                )
        else:
            write_json(
                config_path,
                {"config": dataclasses.asdict(config), "protocol": PROTOCOL},
                indent=1,
            )

        evaluation, training_problems = select(config, rows, tokenizer)
        journal.event(
            "segment",
            resumed_from=checkpoint and checkpoint["path"],
            client_revision=client_revision(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            packages={
                name: importlib.metadata.version(name)
                for name in ["tinker", "transformers", "numpy", "huggingface-hub"]
            },
            deployment=deployment
            or dict.fromkeys(["spindle_version", "spindle_commit"], UNKNOWN),
            evaluation_ids=[p["id"] for p in evaluation],
            prompts_sha256=hashlib.sha256(
                json.dumps(
                    [p["prompt"] for p in evaluation + training_problems]
                ).encode()
            ).hexdigest(),
        )

        run_started = journal.records()[0]["time"]
        started = time.monotonic()
        if checkpoint:
            training = service.create_training_client_from_state_with_optimizer(
                checkpoint["path"]
            )
            sampler_path = checkpoint["sampler_path"]
            sampler = service.create_sampling_client(model_path=sampler_path)
            journal.event(
                "restore",
                step=completed,
                path=checkpoint["path"],
                sampler_path=sampler_path,
                model_id=training.model_id,
                seconds=time.monotonic() - started,
            )
        else:
            training = service.create_lora_training_client(
                base_model=config.model, rank=config.lora_rank
            )
            journal.event(
                "created",
                model_id=training.model_id,
                seconds=time.monotonic() - started,
            )

        def publish(step):
            started = time.monotonic()
            path = (
                training.save_weights_for_sampler(f"step-{step:04d}")
                .result(timeout=REQUEST_TIMEOUT_S)
                .path
            )
            sampler = service.create_sampling_client(model_path=path)
            journal.event(
                "publication", step=step, path=path, seconds=time.monotonic() - started
            )
            return sampler, path

        def save(step):
            started = time.monotonic()
            path = (
                training.save_state(f"step-{step:04d}")
                .result(timeout=REQUEST_TIMEOUT_S)
                .path
            )
            journal.event(
                "checkpoint",
                step=step,
                path=path,
                sampler_path=sampler_path,
                next_cursor=step * config.groups,
                seconds=time.monotonic() - started,
            )

        def evaluate(step):
            if step % config.evaluation_interval and step != config.updates:
                return
            if any(
                r["event"] == "evaluation" and r["step"] == step
                for r in journal.records()
            ):
                return
            started = time.monotonic()
            groups = sample(sampler, evaluation, config.eval_samples, config, tokenizer)
            write_json(
                local / f"evaluation-{step:04d}.json",
                [
                    {
                        "id": group["id"],
                        "label": group["label"],
                        "responses": [
                            tokenizer.decode(s["tokens"]) for s in group["samples"]
                        ],
                    }
                    for group in groups
                ],
            )
            stats = response_stats(groups)
            journal.event(
                "evaluation",
                step=step,
                sampler_path=sampler_path,
                problems=len(groups),
                accuracy=stats.pop("reward"),
                **stats,
                correct_per_problem=[
                    int(sum(s["reward"] for s in group["samples"])) for group in groups
                ],
                seconds=time.monotonic() - started,
            )

        def rollout(step):
            """This update's batch: the recorded one if any, else a new sample."""
            cursor = (step - 1) * config.groups
            path = local / f"rollout-{step:04d}.json"
            recorded = [
                r
                for r in journal.records()
                if r["event"] == "rollout" and r["step"] == step
            ]
            if recorded:
                if (
                    not path.exists()
                    or hashlib.sha256(path.read_bytes()).hexdigest()
                    != recorded[-1]["sha256"]
                ):
                    raise SystemExit(
                        f"The batch recorded for update {step} is missing or changed "
                        f"at {path}. Restore it or start a new run."
                    )
                return json.loads(path.read_text()), recorded[-1]
            started = time.monotonic()
            groups = sample(
                sampler,
                training_problems[cursor : cursor + config.groups],
                config.samples_per_group,
                config,
                tokenizer,
            )
            write_json(path, groups)
            return groups, journal.event(
                "rollout",
                step=step,
                behavior_policy_step=step - 1,
                sampler_path=sampler_path,
                cursor=cursor,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                seconds=time.monotonic() - started,
                **response_stats(groups),
            )

        def update(step, previous_update_time):
            groups, sampled = rollout(step)
            datums, behavior = build_datums(groups)
            started = time.monotonic()
            forward = training.forward_backward(datums, config.loss).result(
                timeout=REQUEST_TIMEOUT_S
            )
            forward_seconds = time.monotonic() - started
            journal.event(
                "forward_backward",
                step=step,
                seconds=forward_seconds,
                metrics=forward.metrics,
            )
            if not all(math.isfinite(v) for v in forward.metrics.values()):
                raise SystemExit(f"Update {step} has nonfinite forward metrics")
            started = time.monotonic()
            journal.event("optimizer_submitted", step=step)
            try:
                optimizer = training.optim_step(
                    types.AdamParams(
                        learning_rate=config.learning_rate,
                        beta1=config.beta1,
                        beta2=config.beta2,
                        eps=config.eps,
                        weight_decay=config.weight_decay,
                        grad_clip_norm=config.grad_clip_norm,
                    )
                ).result(timeout=REQUEST_TIMEOUT_S)
            except BaseException as error:
                raise SystemExit(
                    f"The outcome of optimizer request {step} is unknown "
                    f"({type(error).__name__}). It was not sent again, and this run "
                    "cannot continue. Start a new run."
                ) from error
            optimizer_seconds = time.monotonic() - started
            # Services that report no acceptance metric apply every successful request.
            accepted = optimizer.metrics.get("update_successful:mean", 1.0) == 1.0
            journal.event(
                "optimizer_result",
                step=step,
                accepted=accepted,
                seconds=optimizer_seconds,
                metrics=optimizer.metrics,
            )
            if not accepted:
                raise SystemExit(f"The service rejected optimizer request {step}")
            now = time.time()
            optional = {
                "loss_sum": forward.metrics.get("loss:sum"),
                "grad_norm": optimizer.metrics.get("grad_norm:mean"),
                "seconds_since_previous_update": previous_update_time
                and now - previous_update_time,
                **divergence(forward.loss_fn_outputs, behavior),
            }
            journal.event(
                "update",
                time=now,
                step=step,
                elapsed_seconds=now - run_started,
                learning_rate=config.learning_rate,
                behavior_policy_step=sampled["behavior_policy_step"],
                sampler_path=sampled["sampler_path"],
                cursor=sampled["cursor"],
                sampling_seconds=sampled["seconds"],
                forward_backward_seconds=forward_seconds,
                optimizer_seconds=optimizer_seconds,
                **response_stats(groups),
                **{k: v for k, v in optional.items() if v is not None},
            )
            return now

        if not checkpoint:
            sampler, sampler_path = publish(0)
            save(0)
        evaluate(completed)
        previous_update_time = None
        for step in range(completed + 1, config.updates + 1):
            if stop_after is not None and step > stop_after:
                journal.event("stopped", step=step - 1)
                return
            previous_update_time = update(step, previous_update_time)
            sampler, sampler_path = publish(step)
            if step % config.checkpoint_interval == 0 or step in (
                config.updates,
                stop_after,
            ):
                save(step)
            evaluate(step)
        journal.event("completed", step=config.updates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help=f"use {SMOKE}")
    parser.add_argument("--stop-after", type=int, metavar="UPDATE")
    for field in dataclasses.fields(RunConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=json.loads if field.type is bool else field.type,
            help=f"default {field.default}",
        )
    args = parser.parse_args()
    overrides = {
        **(SMOKE if args.smoke else {}),
        **{
            f.name: getattr(args, f.name)
            for f in dataclasses.fields(RunConfig)
            if getattr(args, f.name) is not None
        },
    }
    if args.resume:
        if overrides:
            parser.error("--resume uses the recorded configuration")
        config = recorded_config(args.output)
    else:
        config = RunConfig(**overrides)
    run(
        config,
        args.output,
        service=connect(),
        tokenizer=AutoTokenizer.from_pretrained(
            config.model, revision=config.tokenizer_revision
        ),
        rows=load_rows(config),
        resume=args.resume,
        stop_after=args.stop_after,
        deployment=served_revisions(),
    )


if __name__ == "__main__":
    main()
