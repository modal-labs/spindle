"""GRPO on DAPO-Math through the stock Tinker SDK. See reference_runs/README.md."""

import argparse
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
from transformers import AutoTokenizer

CONFIG = {
    "model": "Qwen/Qwen3.5-4B",
    "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "enable_thinking": False,
    "dataset": "zhuzilin/dapo-math-17k",
    "dataset_revision": "2e65612930298bde4c5d58fd97b3f23a483aaff9",
    "seed": 1234,
    "max_response_tokens": 2048,
    "temperature": 1.0,
    "updates": 30,
    "groups": 32,
    "samples_per_group": 8,
    "eval_problems": 128,
    "eval_samples": 2,
    "evaluation_interval": 5,
    "checkpoint_interval": 10,
    "lora_rank": 32,
    "loss": "importance_sampling",
    "learning_rate": 4e-5,
}
SMOKE = {"updates": 2, "groups": 4, "eval_problems": 8, "evaluation_interval": 1}


def sample(sampler, problems, count, config, tokenizer):
    """(prompt, sequences, rewards) per problem. Requests run concurrently."""
    params = types.SamplingParams(
        max_tokens=config["max_response_tokens"],
        temperature=config["temperature"],
        seed=config["seed"],
        stop=[tokenizer.eos_token_id],
    )

    def one(problem):
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


def summarize(groups):
    sequences = [s for _, sequences, _ in groups for s in sequences]
    return {
        "responses": len(sequences),
        "reward": mean(r for _, _, rewards in groups for r in rewards),
        "response_tokens_mean": mean(len(s.tokens) for s in sequences),
        "truncated_fraction": mean(s.stop_reason == "length" for s in sequences),
    }


def datum(prompt, sequence, advantage):
    tokens = prompt + list(sequence.tokens)
    prefix = [0.0] * (len(prompt) - 1)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "logprobs": prefix + list(sequence.logprobs),
            "advantages": prefix + [advantage] * len(sequence.tokens),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help=f"use {SMOKE}")
    args = parser.parse_args()
    config = CONFIG | (SMOKE if args.smoke else {})
    args.output.mkdir(parents=True)

    def record(event, **fields):
        with (args.output / "metrics.jsonl").open("a") as stream:
            line = {"time": time.time(), "event": event, **fields}
            stream.write(json.dumps(line, allow_nan=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["tokenizer_revision"]
    )
    rows = hf_hub_download(
        config["dataset"],
        config["dataset"].split("/")[-1] + ".jsonl",
        repo_type="dataset",
        revision=config["dataset_revision"],
    )
    rows = [json.loads(line) for line in Path(rows).read_text().splitlines()]
    held_out, groups = config["eval_problems"], config["groups"]
    problems = [
        (
            tokenizer.apply_chat_template(
                row["prompt"],
                add_generation_prompt=True,
                enable_thinking=config["enable_thinking"],
                return_dict=False,
            ),
            row["label"],
        )
        for row in random.Random(config["seed"]).sample(
            rows, held_out + config["updates"] * groups
        )
    ]
    service = tinker.ServiceClient()
    training = service.create_lora_training_client(config["model"], config["lora_rank"])
    adam = types.AdamParams(learning_rate=config["learning_rate"])
    sdk = importlib.metadata.version("tinker")
    record("run", config=config, adam=adam.model_dump(), tinker_sdk=sdk)
    clock, previous = time.monotonic, None
    for step in range(config["updates"] + 1):
        start = clock()
        path = training.save_weights_for_sampler(f"step-{step}").result().path
        sampler = service.create_sampling_client(model_path=path)
        record("publication", step=step, seconds=clock() - start)
        last = step == config["updates"]
        if last or step % config["evaluation_interval"] == 0:
            start = clock()
            evaluated = sample(
                sampler, problems[:held_out], config["eval_samples"], config, tokenizer
            )
            record(
                "evaluation", step=step, **summarize(evaluated), seconds=clock() - start
            )
        if last or step % config["checkpoint_interval"] == 0:
            start = clock()
            training.save_state(f"step-{step}").result()
            record("checkpoint", step=step, seconds=clock() - start)
        if last:
            break
        start = clock()
        batch = problems[held_out + step * groups :][:groups]
        sampled = sample(sampler, batch, config["samples_per_group"], config, tokenizer)
        sampling_done = clock()
        datums = [
            datum(prompt, sequence, reward - mean(rewards))
            for prompt, sequences, rewards in sampled
            for sequence, reward in zip(sequences, rewards)
        ]
        forward = training.forward_backward(datums, config["loss"]).result()
        forward_done = clock()
        optimizer = training.optim_step(adam).result()
        if optimizer.metrics.get("update_successful:mean", 1.0) != 1.0:
            raise SystemExit(f"Update {step + 1} was rejected: {optimizer.metrics}")
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
            optimizer_metrics=optimizer.metrics,
        )


if __name__ == "__main__":
    main()
