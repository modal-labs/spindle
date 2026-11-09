from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import tinker
from tinker import types
import spindle


def timed(name, call):
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    print(json.dumps({"phase": name, "seconds": elapsed}), flush=True)
    return (result, elapsed)


def resolve(value):
    return value.result(timeout=3600) if hasattr(value, "result") else value


TIMEOUT = 3600
MODEL = "Qwen/Qwen3.5-35B-A3B"


def benchmark(url: str, mode: str, steps: int = 10):
    headers = {}
    if os.environ.get("MODAL_PROXY_TOKEN_ID"):
        headers = {
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        }
    service = tinker.ServiceClient(
        base_url=url, api_key=os.environ["TINKER_API_KEY"], default_headers=headers
    )
    training, create_s = timed(
        "create_training_client",
        lambda: spindle.create_full_training_client(
            service,
            MODEL,
            rollout={"min_containers": 1, "max_containers": 1, "scaledown_window": 300},
        ),
    )
    print(json.dumps({"model_id": training.model_id, "mode": mode}), flush=True)
    tokenizer = training.get_tokenizer()
    prompts = []
    for a, b in [
        (17, 26),
        (34, 19),
        (27, 48),
        (56, 37),
        (73, 18),
        (46, 59),
        (82, 39),
        (64, 47),
    ]:
        text = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"What is {a} + {b}? Explain your reasoning in about 80 words, then finish with 'Answer: N' where N is the integer answer.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append((tokenizer.encode(text, add_special_tokens=False), a + b))
    sampler, initial_publish_s = timed(
        "initial_publication",
        lambda: resolve(training.save_weights_and_get_sampling_client()),
    )
    records = []
    for step in range(steps):
        started = time.perf_counter()
        params = types.SamplingParams(
            max_tokens=128, temperature=1.0, top_p=0.95, top_k=64, seed=1234 + step
        )
        futures = [
            spindle.sample_with_replay(
                sampler,
                types.ModelInput.from_ints(prompt),
                sampling_params=params,
                num_samples=4,
                return_sampling_probabilities=True,
            )
            for prompt, _ in prompts
        ]
        sampled = [future.result(timeout=TIMEOUT) for future in futures]
        rollout_s = time.perf_counter() - started
        prepared_at = time.perf_counter()
        datums, rewards, configs, texts = ([], [], [], [])
        generated_tokens = 0
        trained_tokens = 0
        support_entries = 0
        for (prompt, answer), response in zip(prompts, sampled, strict=True):
            for sequence in response.sequences:
                if not sequence.tokens or sequence.replay is None:
                    raise RuntimeError("missing generated tokens or replay")
                text = tokenizer.decode(sequence.tokens)
                numbers = re.findall("-?\\d+", text)
                reward = 1.0 if numbers and int(numbers[-1]) == answer else -1.0
                tokens = prompt + list(sequence.tokens)
                replay = sequence.replay.training_inputs(len(sequence.tokens))
                support_entries += len(replay["sampling_mask_ids"].data)
                datums.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={
                            "target_tokens": tokens[1:],
                            "weights": [0.0] * (len(prompt) - 1)
                            + [1.0] * len(sequence.tokens),
                            "advantages": [0.0] * (len(prompt) - 1)
                            + [reward] * len(sequence.tokens),
                            **replay,
                        },
                    )
                )
                configs.append(sequence.replay.loss_fn_config())
                rewards.append(reward)
                texts.append(text)
                generated_tokens += len(sequence.tokens)
                trained_tokens += len(tokens) - 1
        if any((config != configs[0] for config in configs)):
            raise RuntimeError("inconsistent replay configurations")
        preparation_s = time.perf_counter() - prepared_at
        forward, forward_s = timed(
            "forward_backward",
            lambda: training.forward_backward(
                datums, "score_centered_mis", loss_fn_config=configs[0]
            ).result(timeout=TIMEOUT),
        )
        optim, optimizer_s = timed(
            "optimizer",
            lambda: training.optim_step(types.AdamParams(learning_rate=1e-06)).result(
                timeout=TIMEOUT
            ),
        )
        if (
            optim.metrics.get("update_successful:mean") != 1
            or optim.metrics.get("grad_norm:mean", 0) <= 0
        ):
            raise RuntimeError(f"unsuccessful optimizer update: {optim.metrics}")
        for metrics in [forward.metrics, optim.metrics]:
            if any(
                (
                    isinstance(value, (int, float)) and (not math.isfinite(value))
                    for value in metrics.values()
                )
            ):
                raise RuntimeError("nonfinite training metrics")
        fixed_prompt = types.ModelInput.from_ints(
            prompts[0][0] + list(sampled[0].sequences[0].tokens)
        )
        old_logprobs = sampler.compute_logprobs(fixed_prompt).result(timeout=TIMEOUT)
        publication_started = time.perf_counter()
        sampler, publication_s = timed(
            "publication",
            lambda: resolve(training.save_weights_and_get_sampling_client()),
        )
        new_logprobs = sampler.compute_logprobs(fixed_prompt).result(timeout=TIMEOUT)
        publication_to_serving_s = time.perf_counter() - publication_started
        changes = [
            abs(a - b)
            for a, b in zip(old_logprobs, new_logprobs, strict=True)
            if a is not None and b is not None
        ]
        record = {
            "step": step,
            "rollout_s": rollout_s,
            "preparation_s": preparation_s,
            "forward_backward_s": forward_s,
            "optimizer_s": optimizer_s,
            "publication_s": publication_s,
            "publication_to_serving_s": publication_to_serving_s,
            "total_s": time.perf_counter() - started,
            "generated_tokens": generated_tokens,
            "model_input_tokens": trained_tokens,
            "sequences": len(datums),
            "support_entries": support_entries,
            "reward_mean": statistics.mean(rewards),
            "max_logprob_change": max(changes, default=0),
            "forward_metrics": forward.metrics,
            "optimizer_metrics": optim.metrics,
            "sample_text": texts[0],
        }
        if not changes:
            raise RuntimeError("publication returned no fixed-sequence probabilities")
        print(json.dumps(record), flush=True)
        records.append(record)
    if not any((record["max_logprob_change"] > 0 for record in records)):
        raise RuntimeError("no published update changed fixed-sequence probabilities")
    steady = records[2:]
    timing_keys = [
        "rollout_s",
        "preparation_s",
        "forward_backward_s",
        "optimizer_s",
        "publication_s",
        "publication_to_serving_s",
        "total_s",
    ]
    result = {
        "mode": mode,
        "model": MODEL,
        "model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "gpu": "B300",
        "trainer_gpus": 4,
        "sampler_gpus": 1,
        "trainer_precision": "BF16",
        "sampler_precision": mode,
        "loss": "score_centered_mis",
        "batch_size": 32,
        "max_tokens": 128,
        "create_s": create_s,
        "initial_publish_s": initial_publish_s,
        "excluded_warmup_steps": 2,
        "median_seconds": {
            key: statistics.median((row[key] for row in steady)) for key in timing_keys
        },
        "records": records,
    }
    print(
        json.dumps(
            {
                "summary": {
                    key: value for key, value in result.items() if key != "records"
                }
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare matched BF16-training deployments with BF16 or MXFP8 samplers."
    )
    parser.add_argument("--mode", choices=("bf16", "mxfp8"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("at least three steps are required to exclude two warmup steps")
    url = os.environ.get("TINKER_BASE_URL")
    if not url or not os.environ.get("TINKER_API_KEY"):
        parser.error("set TINKER_BASE_URL and TINKER_API_KEY")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = benchmark(url, args.mode, args.steps)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
