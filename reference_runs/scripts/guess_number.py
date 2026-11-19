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
    updates: int = 3
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", required=True, nargs=2, metavar=("MODEL", "REVISION")
    )
    parser.add_argument("--updates", type=int, default=Config.updates)
    args = parser.parse_args()
    config = Config(
        model=args.tokenizer[0],
        tokenizer_revision=args.tokenizer[1],
        updates=args.updates,
    )
    max_updates = int(1024 * 0.9) // config.groups
    if not 1 <= config.updates <= max_updates:
        parser.error(f"--updates must be between 1 and {max_updates}")
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
    for step in range(config.updates + 1):
        start = time.monotonic()
        path = training.save_weights_for_sampler(f"step-{step}").result().path
        sampler = service.create_sampling_client(model_path=path)
        record("publication", step=step, seconds=time.monotonic() - start)
        if step in (0, config.updates):
            evaluated = sample(sampler, tokenizer, held_out, 1, config.seed, config)
            record("evaluation", step=step, **summarize(evaluated))
        if step == config.updates:
            checkpoint = training.save_state("final").result().path
            record("checkpoint", step=step, path=checkpoint)
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
        record(
            "update",
            step=step + 1,
            **summarize(groups),
            **timing,
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
