"""GRPO on DAPO-Math through the stock Tinker SDK. See reference_runs/README.md."""

import argparse
import dataclasses
import importlib.metadata
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import tinker
from huggingface_hub import hf_hub_download
from tinker import types
from transformers import PreTrainedTokenizerFast


@dataclasses.dataclass(frozen=True)
class Config:
    model: str = "Qwen/Qwen3.5-4B"
    tokenizer_revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    enable_thinking: bool = False
    dataset: str = "zhuzilin/dapo-math-17k"
    dataset_revision: str = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
    seed: int = 1234
    max_response_tokens: int = 2048
    temperature: float = 1.0
    updates: int = 20
    groups: int = 32
    samples_per_group: int = 8
    eval_problems: int = 128
    eval_samples: int = 2
    evaluation_interval: int = 5
    checkpoint_interval: int = 10
    lora_rank: int = 32
    loss: types.LossFnType = "importance_sampling"
    learning_rate: float = 4e-5


SMOKE = Config(updates=2, groups=4, eval_problems=8, evaluation_interval=1)

Problem = tuple[list[int], str]
Group = tuple[list[int], list[types.SampledSequence], list[float]]


def sample(
    sampler: tinker.SamplingClient,
    problems: list[Problem],
    count: int,
    config: Config,
    tokenizer: PreTrainedTokenizerFast,
) -> list[Group]:
    """Graded responses to each problem. Requests run concurrently."""
    params = types.SamplingParams(
        max_tokens=config.max_response_tokens,
        temperature=config.temperature,
        seed=config.seed,
        stop=[tokenizer.eos_token_id],
    )

    def one(problem: Problem) -> Group:
        prompt, label = problem
        model_input = types.ModelInput.from_ints(prompt)
        sequences = sampler.sample(model_input, count, params).result().sequences
        answers = [
            re.findall(r"\\boxed\{([^{}]*)\}", tokenizer.decode(s.tokens))
            for s in sequences
        ]
        return prompt, sequences, [float(a[-1:] == [label]) for a in answers]

    with ThreadPoolExecutor(len(problems)) as pool:
        return list(pool.map(one, problems))


def summarize(groups: list[Group]) -> dict[str, float]:
    sequences = [s for _, sequences, _ in groups for s in sequences]
    return {
        "responses": len(sequences),
        "reward": mean(r for _, _, rewards in groups for r in rewards),
        "response_tokens_mean": mean(len(s.tokens) for s in sequences),
        "truncated_fraction": mean(s.stop_reason == "length" for s in sequences),
    }


def datum(
    prompt: list[int], sequence: types.SampledSequence, advantage: float
) -> types.Datum:
    tokens = prompt + list(sequence.tokens)
    prefix = [0.0] * (len(prompt) - 1)
    floats = {
        "logprobs": prefix + list(sequence.logprobs or []),
        "advantages": prefix + [advantage] * len(sequence.tokens),
    }
    size = [len(tokens) - 1]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=tokens[1:], dtype="int64", shape=size
            ),
            **{
                name: types.TensorData(data=values, dtype="float32", shape=size)
                for name, values in floats.items()
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help=f"use {SMOKE}")
    args = parser.parse_args()
    config = SMOKE if args.smoke else Config()
    args.output.mkdir(parents=True)

    def record(event: str, **fields: object) -> None:
        with (args.output / "metrics.jsonl").open("a") as stream:
            line = {"time": time.time(), "event": event, **fields}
            stream.write(json.dumps(line, allow_nan=False) + "\n")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.model, revision=config.tokenizer_revision
    )
    dataset = hf_hub_download(
        config.dataset,
        config.dataset.split("/")[-1] + ".jsonl",
        repo_type="dataset",
        revision=config.dataset_revision,
    )
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines()]
    count = config.eval_problems + config.updates * config.groups
    problems: list[Problem] = [
        (
            tokenizer.apply_chat_template(
                row["prompt"],
                add_generation_prompt=True,
                enable_thinking=config.enable_thinking,
                return_dict=False,
            ),
            row["label"],
        )
        for row in random.Random(config.seed).sample(rows, count)
    ]
    held_out = problems[: config.eval_problems]
    service = tinker.ServiceClient()
    training = service.create_lora_training_client(config.model, config.lora_rank)
    adam = types.AdamParams(learning_rate=config.learning_rate)
    record(
        "run",
        config=dataclasses.asdict(config),
        adam=adam.model_dump(),
        tinker_sdk=importlib.metadata.version("tinker"),
    )
    clock, previous = time.monotonic, None
    for step in range(config.updates + 1):
        start = clock()
        path = training.save_weights_for_sampler(f"step-{step}").result().path
        sampler = service.create_sampling_client(model_path=path)
        record("publication", step=step, seconds=clock() - start)
        last = step == config.updates
        if last or step % config.evaluation_interval == 0:
            start = clock()
            evaluated = sample(
                sampler, held_out, config.eval_samples, config, tokenizer
            )
            record(
                "evaluation", step=step, **summarize(evaluated), seconds=clock() - start
            )
        if last or step % config.checkpoint_interval == 0:
            start = clock()
            training.save_state(f"step-{step}").result()
            record("checkpoint", step=step, seconds=clock() - start)
        if last:
            break
        start = clock()
        first = config.eval_problems + step * config.groups
        batch = problems[first : first + config.groups]
        sampled = sample(sampler, batch, config.samples_per_group, config, tokenizer)
        sampling_done = clock()
        datums = [
            datum(prompt, sequence, reward - mean(rewards))
            for prompt, sequences, rewards in sampled
            for sequence, reward in zip(sequences, rewards)
        ]
        forward = training.forward_backward(datums, config.loss).result()
        forward_done = clock()
        optimizer = training.optim_step(adam).result()
        metrics = optimizer.metrics or {}
        if metrics.get("update_successful:mean", 1.0) != 1.0:
            raise SystemExit(f"Update {step + 1} was rejected: {metrics}")
        timing = {
            "sampling_seconds": sampling_done - start,
            "forward_backward_seconds": forward_done - sampling_done,
            "optimizer_seconds": clock() - forward_done,
        }
        if previous:
            timing["seconds_since_previous_update"] = clock() - previous
        previous = clock()
        record(
            "update",
            step=step + 1,
            **summarize(sampled),
            **timing,
            forward_metrics=forward.metrics,
            optimizer_metrics=metrics,
        )


if __name__ == "__main__":
    main()
