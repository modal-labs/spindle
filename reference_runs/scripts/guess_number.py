"""Multi-turn guess-the-number GRPO through the stock Tinker SDK."""

import argparse
import dataclasses
import importlib.metadata
import json
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import cast

import numpy as np
import tinker
from tinker import types
from transformers import PreTrainedTokenizerFast

from dapo_math import datum

SYSTEM_PROMPT = (
    "Your job is to guess a integer between 0 and 1024. You can output your guess "
    "with the format Guess: <guess> (without the angle brackets), and say nothing "
    "else. Then we will tell you whether you guess is too high, too low, or correct."
)
FORMAT_FEEDBACK = (
    "Failed to parse. Please output in the format Guess: <guess> "
    "(without the angle brackets), and say nothing else."
)


@dataclasses.dataclass(frozen=True)
class Config:
    model: str
    tokenizer_revision: str
    seed: int = 0
    total_steps: int = 3
    save_step: int | None = None
    restore_step: int | None = None
    groups: int = 8
    samples_per_group: int = 4
    max_turns: int = 10
    max_response_tokens: int = 64
    eval_problems: int = 8
    lora_rank: int = 32
    learning_rate: float = 3e-5


@dataclasses.dataclass
class Episode:
    turns: list[tuple[list[int], types.SampledSequence]]
    reward: float
    solved: bool


def feedback(text: str, answer: int) -> tuple[str, float]:
    match = re.match(r"Guess: (.*)", text)
    try:
        guess = int(match[1]) if match else None
    except ValueError:
        guess = None
    if guess is None:
        return FORMAT_FEEDBACK, -1.0
    if guess == answer:
        return "Correct", 1.0
    return ("Too low" if guess < answer else "Too high"), 0.0


def response_text(text: str) -> str:
    # Harmony and Qwen templates keep reasoning separate from the final answer.
    text = text.rsplit("<|channel|>final<|message|>", 1)[-1]
    text = text.rsplit("</think>", 1)[-1]
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


def play(
    sampler: tinker.SamplingClient,
    tokenizer: PreTrainedTokenizerFast,
    answer: int,
    seed: int,
    config: Config,
) -> Episode:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode = Episode([], 0.0, False)
    for turn in range(config.max_turns):
        prompt = cast(
            list[int],
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                reasoning_effort="low",
                return_dict=False,
            ),
        )
        result = sampler.sample(
            types.ModelInput.from_ints(prompt),
            1,
            types.SamplingParams(
                max_tokens=config.max_response_tokens,
                temperature=1.0,
                seed=seed + turn,
                stop=[tokenizer.eos_token_id],
            ),
        ).result()
        if len(result.sequences) != 1:
            raise ValueError("Expected one sampled sequence")
        sequence = result.sequences[0]
        if (
            not sequence.tokens
            or sequence.logprobs is None
            or len(sequence.logprobs) != len(sequence.tokens)
            or not all(math.isfinite(value) for value in sequence.logprobs)
        ):
            raise ValueError("Sampling returned empty tokens or invalid logprobs")
        episode.turns.append((prompt, sequence))
        text = response_text(cast(str, tokenizer.decode(sequence.tokens)))
        reply, reward = feedback(text, answer)
        episode.reward += reward
        episode.solved = reward == 1
        if episode.solved:
            break
        messages.extend(
            [{"role": "assistant", "content": text}, {"role": "user", "content": reply}]
        )
    return episode


def sample(
    sampler: tinker.SamplingClient,
    tokenizer: PreTrainedTokenizerFast,
    answers: list[int],
    count: int,
    seed: int,
    config: Config,
) -> list[list[Episode]]:
    def one(item: tuple[int, int]) -> Episode:
        index, answer = item
        return play(sampler, tokenizer, answer, seed + index * config.max_turns, config)

    with ThreadPoolExecutor(len(answers) * count) as pool:
        episodes = list(
            pool.map(one, enumerate(a for a in answers for _ in range(count)))
        )
    return [episodes[i : i + count] for i in range(0, len(episodes), count)]


def summarize(groups: list[list[Episode]]) -> dict[str, float | int]:
    episodes = [episode for group in groups for episode in group]
    turns = [sequence for episode in episodes for _, sequence in episode.turns]
    return {
        "responses": len(episodes),
        "reward": mean(episode.reward for episode in episodes),
        "success_fraction": mean(episode.solved for episode in episodes),
        "turns_mean": mean(len(episode.turns) for episode in episodes),
        "response_tokens_mean": mean(len(sequence.tokens) for sequence in turns),
        "truncated_fraction": mean(
            sequence.stop_reason == "length" for sequence in turns
        ),
    }


def scores(sampler: tinker.SamplingClient, prompt: types.ModelInput) -> np.ndarray:
    result = sampler.compute_logprobs(prompt).result()
    values = np.asarray([value for value in result if value is not None])
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Checkpoint probe returned empty or nonfinite scores")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", required=True, nargs=2, metavar=("MODEL", "REVISION")
    )
    parser.add_argument("--total-steps", type=int, default=Config.total_steps)
    parser.add_argument("--save-step", type=int)
    parser.add_argument("--restore-step", type=int)
    args = parser.parse_args()
    config = Config(
        model=args.tokenizer[0],
        tokenizer_revision=args.tokenizer[1],
        total_steps=args.total_steps,
        save_step=args.save_step,
        restore_step=args.restore_step,
    )
    max_steps = int(1024 * 0.9) // config.groups
    if not 1 <= config.total_steps <= max_steps:
        parser.error(f"--total-steps must be between 1 and {max_steps}")
    if (config.save_step is not None or config.restore_step is not None) and not (
        config.save_step is not None
        and config.restore_step is not None
        and 0 < config.save_step < config.restore_step < config.total_steps
    ):
        parser.error(
            "Checkpoint schedule requires 0 < save_step < restore_step < total_steps"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(exist_ok=False)

    def record(event: str, **fields: object) -> None:
        line = {"event": event, "time": time.time(), **fields}
        with args.output.open("a") as stream:
            stream.write(json.dumps(line, allow_nan=False) + "\n")
        print(json.dumps(line, allow_nan=False), flush=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.model, revision=config.tokenizer_revision
    )
    answers = random.Random(config.seed).sample(range(1024), 1024)
    held_out = answers[int(1024 * 0.9) :][: config.eval_problems]
    service = tinker.ServiceClient()
    training = service.create_lora_training_client(args.base_model, config.lora_rank)
    adam = types.AdamParams(learning_rate=config.learning_rate, beta2=0.95, eps=1e-8)
    record(
        "run",
        config=dataclasses.asdict(config),
        adam=adam.model_dump(),
        tinker_sdk=importlib.metadata.version("tinker"),
        recipe=args.base_model,
    )
    previous = None
    step = attempt = 0
    restored = False
    changed = 0.0
    checkpoint: str | None = None
    saved_scores: np.ndarray | None = None
    probe: types.ModelInput | None = None
    restore_started: float | None = None
    load_seconds = 0.0
    last_batch: list[types.Datum] = []
    while True:
        start = time.monotonic()
        path = (
            training.save_weights_for_sampler(f"step-{step}-attempt-{attempt}")
            .result()
            .path
        )
        sampler = service.create_sampling_client(model_path=path)
        record(
            "publication", step=step, attempt=attempt, seconds=time.monotonic() - start
        )
        if restore_started is not None:
            assert probe is not None and saved_scores is not None
            loaded = scores(sampler, probe)
            np.testing.assert_allclose(loaded, saved_scores, rtol=2e-4, atol=2e-4)
            record(
                "restore_validation",
                step=step,
                from_step=config.restore_step,
                path=checkpoint,
                attempt=attempt,
                max_abs_error=float(np.max(np.abs(loaded - saved_scores))),
                discarded_score_change=changed,
                seconds=time.monotonic() - restore_started,
                load_seconds=load_seconds,
            )
            restore_started = None
        if step == config.save_step and not restored:
            probe = last_batch[0].model_input
            saved_scores = scores(sampler, probe)
            start = time.monotonic()
            checkpoint = training.save_state(f"step-{step}").result().path
            record(
                "checkpoint",
                step=step,
                path=checkpoint,
                seconds=time.monotonic() - start,
            )
        if step == config.restore_step and not restored:
            assert checkpoint is not None and saved_scores is not None
            restore_started = time.monotonic()
            assert probe is not None
            before_restore = scores(sampler, probe)
            changed = float(np.max(np.abs(before_restore - saved_scores)))
            if np.allclose(before_restore, saved_scores, rtol=2e-4, atol=2e-4):
                raise ValueError(
                    "Training did not change probe scores beyond restore tolerance"
                )
            start = time.monotonic()
            training.load_state_with_optimizer(checkpoint).result()
            load_seconds = time.monotonic() - start
            restored = True
            step = config.save_step
            previous = None
            continue
        if step in (0, config.total_steps):
            evaluated = sample(sampler, tokenizer, held_out, 1, config.seed, config)
            record("evaluation", step=step, **summarize(evaluated))
        if step == config.total_steps:
            start = time.monotonic()
            checkpoint = training.save_state("final").result().path
            record(
                "checkpoint",
                step=step,
                path=checkpoint,
                seconds=time.monotonic() - start,
            )
            break
        start = time.monotonic()
        groups = sample(
            sampler,
            tokenizer,
            answers[step * config.groups : (step + 1) * config.groups],
            config.samples_per_group,
            config.seed + (step + 1) * 10000,
            config,
        )
        sampled = time.monotonic()
        batch = [
            datum(prompt, sequence, episode.reward - mean(e.reward for e in group))
            for group in groups
            for episode in group
            for prompt, sequence in episode.turns
        ]
        last_batch = batch
        forward = training.forward_backward(batch, "importance_sampling").result()
        trained = time.monotonic()
        turns = [
            turn for group in groups for episode in group for turn in episode.turns
        ]
        differences = [
            float(actual) - float(expected)
            for (prompt, sequence), output in zip(
                turns, forward.loss_fn_outputs, strict=True
            )
            for actual, expected in zip(
                output["logprobs"].data[len(prompt) - 1 :],
                sequence.logprobs or [],
                strict=True,
            )
        ]
        if not all(math.isfinite(value) for value in differences):
            raise ValueError("Training returned nonfinite logprobs")
        optimizer = training.optim_step(adam).result()
        metrics = optimizer.metrics or {}
        if metrics.get("update_successful:mean", 1.0) != 1.0:
            raise ValueError(f"Update {step + 1} was rejected: {metrics}")
        now = time.monotonic()
        timing = (
            {}
            if previous is None
            else {"seconds_since_previous_update": now - previous}
        )
        previous = now
        step += 1
        attempt += 1
        record(
            "update",
            step=step,
            attempt=attempt,
            discarded=not restored
            and config.save_step is not None
            and step > config.save_step,
            **summarize(groups),
            **timing,
            update_seconds=now - start,
            sampling_seconds=sampled - start,
            forward_backward_seconds=trained - sampled,
            optimizer_seconds=now - trained,
            kl_sample_train_v2=mean(0.5 * value**2 for value in differences)
            if differences
            else None,
            forward_metrics=forward.metrics,
            optimizer_metrics=metrics,
        )


if __name__ == "__main__":
    main()
