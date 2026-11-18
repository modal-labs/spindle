"""Run sampled rollouts through PPO on Inkling with the stock SDK and renderer."""

from __future__ import annotations

from collections.abc import Callable

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import time
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "thinkingmachines/Inkling-2-Small"
PPO = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}


def timed[T](stage: str, call: Callable[[], T]) -> T:
    started = time.monotonic()
    result = call()
    print(
        json.dumps({"stage": stage, "seconds": time.monotonic() - started}), flush=True
    )
    return result


def score_delta(before: list, after: list) -> float:
    pairs = [
        (a, b)
        for a, b in zip(before, after, strict=True)
        if a is not None and b is not None
    ]
    assert pairs and all(math.isfinite(x) for pair in pairs for x in pair)
    return max(abs(a - b) for a, b in pairs)


def main(steps: int, group_size: int, output: Path) -> None:
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    tokenizer = get_tokenizer(MODEL)
    renderer = get_renderer(renderer_name, tokenizer)
    prompt = renderer.build_generation_prompt(
        [
            {
                "role": "user",
                "content": "Write a friendly one-sentence greeting for a new teammate.",
            }
        ],
        effort=0.0,
    )
    prompt_tokens = prompt.to_ints()
    evidence = {
        "model": MODEL,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("tinker", "tinker-cookbook", "torch", "tml-renderers")
        },
        "renderer": renderer_name,
        "effort": 0.0,
        "rank": 32,
        "learning_rate": 1e-5,
        "loss": "ppo",
        "loss_fn_config": PPO,
        "reward": "-UTF8 response bytes / 128; -4 for empty or malformed responses",
        "group_size": group_size,
        "prompt_tokens": len(prompt_tokens),
        "steps": [],
    }
    headers = {
        key: os.environ[value]
        for key, value in (
            ("Modal-Key", "MODAL_PROXY_TOKEN_ID"),
            ("Modal-Secret", "MODAL_PROXY_TOKEN_SECRET"),
        )
        if value in os.environ
    }
    with tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
        default_headers_overrides=headers,
    ) as service:
        training, control = [
            timed(
                f"create_adapter_{seed}",
                lambda seed=seed: service.create_lora_training_client(
                    base_model=MODEL,
                    rank=32,
                    seed=seed,
                    train_mlp=False,
                    train_attn=True,
                    train_unembed=True,
                ).result(),
            )
            for seed in (41, 42)
        ]
        paths = set()

        def publish(client: tinker.TrainingClient, name: str) -> tinker.SamplingClient:
            publication = timed(
                name, lambda: client.save_weights_for_sampler(name).result()
            )
            assert publication.path not in paths
            paths.add(publication.path)
            return service.create_sampling_client(model_path=publication.path).result()

        sampler = publish(training, "rl-policy-0")
        frozen = publish(control, "rl-control")
        canary = None
        initial_scores = []
        frozen_scores = []
        final_delta = 0.0
        for step in range(steps):
            for attempt in range(3):
                result = timed(
                    f"rollout_{step}_{attempt}",
                    lambda sampler=sampler: sampler.sample(
                        prompt=prompt,
                        num_samples=group_size,
                        sampling_params=types.SamplingParams(
                            max_tokens=64,
                            temperature=1.0,
                            top_p=1.0,
                            stop=renderer.get_stop_sequences(),
                        ),
                    ).result(),
                )
                assert len(result.sequences) == group_size
                rewards = []
                responses = []
                terminations = []
                for sequence in result.sequences:
                    assert sequence.tokens and sequence.logprobs
                    assert len(sequence.tokens) == len(sequence.logprobs)
                    assert all(math.isfinite(x) for x in sequence.logprobs)
                    message, termination = renderer.parse_response(sequence.tokens)
                    text = get_text_content(message).strip()
                    responses.append(text)
                    terminations.append(str(termination))
                    rewards.append(
                        -len(text.encode("utf-8")) / 128
                        if text and str(termination) == "stop_sequence"
                        else -4.0
                    )
                if max(rewards) > min(rewards):
                    break
                print(
                    json.dumps(
                        {
                            "step": step,
                            "constant_reward_attempt": attempt,
                            "responses": responses,
                        }
                    ),
                    flush=True,
                )
            else:
                raise RuntimeError(
                    "No reward variation after three sampled groups; refusing a zero-advantage update"
                )
            mean = statistics.mean(rewards)
            scale = max(statistics.pstdev(rewards), 0.5)
            advantages = [(reward - mean) / scale for reward in rewards]
            assert min(advantages) < 0 < max(advantages)
            if canary is None:
                canary = types.ModelInput.from_ints(
                    prompt_tokens + result.sequences[0].tokens
                )
                initial_scores = timed(
                    "initial_policy_scores",
                    lambda sampler=sampler, canary=canary: sampler.compute_logprobs(
                        canary
                    ).result(),
                )
                frozen_scores = timed(
                    "initial_control_scores",
                    lambda canary=canary: frozen.compute_logprobs(canary).result(),
                )
            total_tokens = sum(len(sequence.tokens) for sequence in result.sequences)
            data = []
            sampled_payload = []
            trained_payload = []
            prefix = len(prompt_tokens) - 1
            for sequence, advantage in zip(result.sequences, advantages, strict=True):
                tokens = list(sequence.tokens)
                logprobs = list(sequence.logprobs)
                sampled_payload.append([tokens, logprobs])
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(prompt_tokens + tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": prompt_tokens[1:] + tokens,
                        "logprobs": [0.0] * prefix + logprobs,
                        "advantages": [0.0] * prefix + [advantage] * len(tokens),
                        "weights": [0.0] * prefix
                        + [total_tokens / (group_size * len(tokens))] * len(tokens),
                    },
                )
                assert all(
                    weight == 0
                    for weight in datum.loss_fn_inputs["weights"].data[:prefix]
                )
                trained_payload.append(
                    [
                        datum.loss_fn_inputs["target_tokens"].data[prefix:],
                        datum.loss_fn_inputs["logprobs"].data[prefix:],
                    ]
                )
                data.append(datum)
            assert sampled_payload == trained_payload
            rollout_digest = hashlib.sha256(
                json.dumps(sampled_payload).encode()
            ).hexdigest()
            backward = timed(
                f"ppo_backward_{step}",
                lambda data=data: training.forward_backward(
                    data, "ppo", loss_fn_config=PPO
                ).result(),
            )
            assert math.isfinite(backward.metrics["loss:sum"]), backward.metrics
            optimizer = timed(
                f"optimizer_{step}",
                lambda: training.optim_step(
                    types.AdamParams(learning_rate=1e-5)
                ).result(),
            )
            assert all(
                math.isfinite(value)
                for value in optimizer.metrics.values()
                if isinstance(value, (float, int))
            )
            sampler = publish(training, f"rl-policy-{step + 1}")
            updated_scores = timed(
                f"updated_policy_scores_{step}",
                lambda sampler=sampler, canary=canary: sampler.compute_logprobs(
                    canary
                ).result(),
            )
            final_delta = score_delta(initial_scores, updated_scores)
            assert final_delta > 1e-6, final_delta
            unchanged_scores = timed(
                f"control_scores_{step}",
                lambda canary=canary: frozen.compute_logprobs(canary).result(),
            )
            control_delta = score_delta(frozen_scores, unchanged_scores)
            assert control_delta < 1e-5, control_delta
            row = {
                "step": step,
                "sampled_policy_version": step,
                "published_policy_version": step + 1,
                "responses": responses,
                "terminations": terminations,
                "rewards": rewards,
                "advantages": advantages,
                "sampled_tokens": total_tokens,
                "rollout_and_training_payload_sha256": rollout_digest,
                "loss": backward.metrics["loss:sum"],
                "optimizer_metrics": optimizer.metrics,
                "policy_logprob_delta_from_initial": final_delta,
                "control_logprob_delta": control_delta,
            }
            evidence["steps"].append(row)
            print(json.dumps(row), flush=True)
        final = timed(
            "final_updated_policy_rollout",
            lambda sampler=sampler: sampler.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=64,
                    temperature=1.0,
                    top_p=1.0,
                    stop=renderer.get_stop_sequences(),
                ),
            ).result(),
        )
        assert final.sequences[0].tokens and final.sequences[0].logprobs
        assert all(math.isfinite(x) for x in final.sequences[0].logprobs)
        timed("save_rl_state", lambda: training.save_state("rl-final").result())
        evidence["status"] = "passed"
        evidence["final_policy_logprob_delta"] = final_delta
        output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "rl_updates": steps,
                    "sampled_tokens_trained": sum(
                        row["sampled_tokens"] for row in evidence["steps"]
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("inkling2-rl-results.json"))
    args = parser.parse_args()
    if args.steps < 2 or args.group_size < 2:
        parser.error("use at least two updates and two samples per group")
    main(args.steps, args.group_size, args.output)
