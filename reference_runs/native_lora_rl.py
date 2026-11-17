"""Short arithmetic GRPO and optimizer-resume validation through stock Tinker."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
from pathlib import Path
import re
from statistics import mean
import threading
import time
from typing import Protocol
from collections.abc import Callable

import tinker
from tinker import types

LOCK = threading.Lock()


def record(path: Path, event: str, **values: object) -> None:
    row = {
        "time": time.time(),
        "event": event,
        **values,
    }
    with LOCK:
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)


class Waitable[T](Protocol):
    def result(self, timeout: float | None = None) -> T: ...


def wait[T](future: Waitable[T]) -> T:
    return future.result(timeout=3600)


def score(training: tinker.TrainingClient, datum: types.Datum) -> list[float]:
    result = wait(training.forward([datum], "cross_entropy"))
    return [float(v) for v in result.loss_fn_outputs[0]["logprobs"].data]


def delta(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right)
    return max(abs(a - b) for a, b in zip(left, right))


def make_datum(
    prompt: list[int], sequence: types.SampledSequence, advantage: float
) -> types.Datum:
    assert sequence.tokens and len(sequence.tokens) == len(sequence.logprobs or [])
    assert sequence.logprobs is not None
    assert all(math.isfinite(p) for p in sequence.logprobs)
    full = prompt + list(sequence.tokens)
    prefix = [0.0] * (len(prompt) - 1)
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=full[1:], dtype="int64", shape=[len(full) - 1]
            ),
            "logprobs": types.TensorData(
                data=prefix + list(sequence.logprobs),
                dtype="float32",
                shape=[len(full) - 1],
            ),
            "advantages": types.TensorData(
                data=prefix + [advantage] * len(sequence.tokens),
                dtype="float32",
                shape=[len(full) - 1],
            ),
        },
    )


def train_client(
    service: tinker.ServiceClient,
    recipe: str,
    client_index: int,
    steps: int,
    output: Path,
    train_unembed: bool,
    require_exact_replay: bool,
) -> Callable[[], dict[str, object]]:
    path = output / f"client-{client_index}.jsonl"
    training = service.create_lora_training_client(
        recipe,
        rank=8 if client_index == 0 else 16,
        train_attn=True,
        train_mlp=True,
        train_unembed=train_unembed,
    )
    tokenizer = training.get_tokenizer()
    probe_tokens = tokenizer.encode(
        "The answer to two plus two is four. Two times three is six."
    )
    probe = types.Datum(
        model_input=types.ModelInput.from_ints(probe_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=probe_tokens[1:], dtype="int64", shape=[len(probe_tokens) - 1]
            ),
            "weights": types.TensorData(
                data=[1.0] * (len(probe_tokens) - 1),
                dtype="float32",
                shape=[len(probe_tokens) - 1],
            ),
        },
    )
    initial = score(training, probe)
    assert all(map(math.isfinite, initial))
    initial_repeat_diff = delta(initial, score(training, probe))
    assert initial_repeat_diff <= 1e-5, (
        "repeated forward changed scores",
        initial_repeat_diff,
    )
    record(
        path,
        "client",
        recipe=recipe,
        client=client_index,
        model_id=training.model_id,
        rank=8 if client_index == 0 else 16,
        tinker=importlib.metadata.version("tinker"),
        initial_repeat_max_diff=initial_repeat_diff,
        require_exact_replay=require_exact_replay,
    )
    adam = types.AdamParams(learning_rate=1e-4)
    grad_norms = []
    last_datums = None
    for step in range(steps):
        started = time.monotonic()
        publication = wait(
            training.save_weights_for_sampler(f"client{client_index}-step{step}")
        ).path
        sampler = service.create_sampling_client(model_path=publication)
        groups = []
        for group in range(4):
            a = 137 + 13 * step + 29 * group + 7 * client_index
            b = 97 + 11 * step + 17 * group
            question = (
                f"What is {a} * {b}? Give only the integer answer, without explanation."
            )
            prompt_text = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                if tokenizer.chat_template
                else "Question: " + question + "\nAnswer:"
            )
            assert isinstance(prompt_text, str)
            prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
            sequences = wait(
                sampler.sample(
                    types.ModelInput.from_ints(prompt),
                    8,
                    types.SamplingParams(
                        max_tokens=64,
                        temperature=1.0,
                        seed=1000 + step * 4 + group,
                        stop=[tokenizer.eos_token_id],
                    ),
                )
            ).sequences
            rewards = []
            for sequence in sequences:
                text = tokenizer.decode(sequence.tokens)
                answers = re.findall(r"-?\d+", text)
                rewards.append(float(bool(answers) and int(answers[-1]) == a * b))
            groups.append((prompt, sequences, rewards))
        rewards = [r for _, _, rs in groups for r in rs]
        # A small length penalty keeps an all-correct batch informative, while preserving correctness as the main reward.
        shaped = [
            [r - 0.001 * len(s.tokens) for s, r in zip(sequences, rs)]
            for _, sequences, rs in groups
        ]
        datums = [
            make_datum(prompt, s, (r - mean(rs)) / 32)
            for (prompt, sequences, _), rs in zip(groups, shaped)
            for s, r in zip(sequences, rs)
        ]
        forward = wait(training.forward_backward(datums, "importance_sampling"))
        optimizer = wait(training.optim_step(adam))
        metrics = optimizer.metrics or {}
        assert metrics.get("update_successful:mean", 1) == 1, metrics
        assert all(math.isfinite(float(v)) for v in metrics.values()), metrics
        grad = float(metrics.get("grad_norm:mean", 0))
        grad_norms.append(grad)
        record(
            path,
            "update",
            recipe=recipe,
            client=client_index,
            step=step + 1,
            reward=mean(rewards),
            nonzero_advantages=sum(
                any(v != 0 for v in d.loss_fn_inputs["advantages"].data) for d in datums
            ),
            response_tokens=mean(len(s.tokens) for _, seqs, _ in groups for s in seqs),
            forward_metrics=forward.metrics,
            optimizer_metrics=metrics,
            seconds=time.monotonic() - started,
        )
        last_datums = datums

    def verify_resume() -> dict[str, object]:
        trained = score(training, probe)
        change = delta(initial, trained)
        assert change > 1e-5, ("adapter did not change probe logprobs", change)
        assert any(g > 0 for g in grad_norms), ("no nonzero gradients", grad_norms)
        assert last_datums is not None
        checkpoint = wait(training.save_state(f"client{client_index}-resume")).path
        record(
            path,
            "checkpoint",
            recipe=recipe,
            client=client_index,
            checkpoint=checkpoint,
        )
        wait(training.forward_backward(last_datums, "importance_sampling"))
        next_metrics = wait(training.optim_step(adam)).metrics
        next_scores = score(training, probe)
        reference_checkpoint = wait(
            training.save_state(f"client{client_index}-reference-next")
        ).path
        wait(training.load_state_with_optimizer(checkpoint))
        restored = score(training, probe)
        restore_diff = delta(restored, trained)
        roundtrip_checkpoint = wait(
            training.save_state(f"client{client_index}-roundtrip")
        ).path
        assert restore_diff <= 1e-5, ("restore changed scores", restore_diff)
        wait(training.forward_backward(last_datums, "importance_sampling"))
        replay_metrics = wait(training.optim_step(adam)).metrics
        replay_scores = score(training, probe)
        replay_checkpoint = wait(
            training.save_state(f"client{client_index}-replayed-next")
        ).path
        replay_diff = delta(replay_scores, next_scores)
        record(
            path,
            "resume_check",
            recipe=recipe,
            client=client_index,
            checkpoint=checkpoint,
            roundtrip_checkpoint=roundtrip_checkpoint,
            reference_checkpoint=reference_checkpoint,
            replay_checkpoint=replay_checkpoint,
            restore_max_diff=restore_diff,
            resumed_step_max_diff=replay_diff,
            next_optimizer_metrics=next_metrics,
            replay_optimizer_metrics=replay_metrics,
        )
        for metrics in (next_metrics, replay_metrics):
            assert metrics is not None
            assert metrics.get("update_successful:mean", 1) == 1, metrics
            assert all(math.isfinite(float(v)) for v in metrics.values()), metrics
        snapshot = wait(
            training.save_weights_for_sampler(f"client{client_index}-final")
        ).path
        sampler = service.create_sampling_client(model_path=snapshot)
        served = wait(
            sampler.compute_logprobs(types.ModelInput.from_ints(probe_tokens))
        )
        assert all(p is None or math.isfinite(p) for p in served)
        greedy = wait(
            sampler.sample(
                types.ModelInput.from_ints(probe_tokens[:-1]),
                1,
                types.SamplingParams(max_tokens=16, temperature=0),
            )
        )
        record(
            path,
            "completed",
            exact_replay=replay_diff <= 1e-5,
            recipe=recipe,
            client=client_index,
            probe_change=change,
            checkpoint=checkpoint,
            restore_max_diff=restore_diff,
            resumed_step_max_diff=replay_diff,
            next_optimizer_metrics=next_metrics,
            replay_optimizer_metrics=replay_metrics,
            sampler_logprobs=served,
            sample=tokenizer.decode(greedy.sequences[0].tokens),
        )
        if require_exact_replay:
            assert replay_diff <= 1e-5, ("resumed update differs", replay_diff)
        return {
            "client": client_index,
            "passed": True,
            "probe_change": change,
            "exact_replay": replay_diff <= 1e-5,
            "resumed_step_max_diff": replay_diff,
        }

    return verify_resume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--train-unembed", action="store_true")
    parser.add_argument("--require-exact-replay", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.clients <= 2:
        parser.error("steps must be positive and clients must be 1 or 2")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    service = tinker.ServiceClient()
    with ThreadPoolExecutor(args.clients) as pool:
        futures = [
            pool.submit(
                train_client,
                service,
                args.recipe,
                i,
                args.steps,
                output,
                args.train_unembed,
                args.require_exact_replay,
            )
            for i in range(args.clients)
        ]
        checks = [f.result() for f in futures]
    results = [check() for check in checks]
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
