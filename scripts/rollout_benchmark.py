# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["jinja2", "tinker", "transformers", "wandb"]
# ///

"""Time sampling rounds against any Tinker-compatible URL with the stock Tinker SDK.

Each round sends every prompt --copies times at once with fixed seeds and waits
for all of them, so every round and every run does the same work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import tinker
import wandb


def load_prompts(path: str, client: tinker.SamplingClient) -> list[list[int]]:
    """Each JSONL line has "tokens" (token ids) or "messages" (chat messages)."""
    with open(path, encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines if line.strip()]
    tokenizer = None if all("tokens" in row for row in rows) else client.get_tokenizer()
    return [
        row["tokens"]
        if "tokens" in row
        else tokenizer.encode(  # ty: ignore[possibly-missing-attribute]
            tokenizer.apply_chat_template(  # ty: ignore[possibly-missing-attribute]
                row["messages"], add_generation_prompt=True, tokenize=False
            ),
            add_special_tokens=False,
        )
        for row in rows
    ]


async def run_round(
    client: tinker.SamplingClient, prompts: list[list[int]], args: argparse.Namespace
) -> dict[str, float]:
    async def one(seed: int, tokens: list[int]) -> tuple[float, int]:
        started = time.perf_counter()
        response = await client.sample_async(
            prompt=tinker.ModelInput.from_ints(tokens),
            num_samples=args.num_samples,
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_tokens, temperature=args.temperature, seed=seed
            ),
        )
        return time.perf_counter() - started, sum(
            len(sequence.tokens) for sequence in response.sequences
        )

    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            one(args.seed + copy * len(prompts) + index, tokens)
            for copy in range(args.copies)
            for index, tokens in enumerate(prompts)
        )
    )
    wall = time.perf_counter() - started
    latencies = sorted(latency for latency, _ in results)
    generated = sum(tokens for _, tokens in results)
    return {
        "wall_s": wall,
        "p50_s": latencies[round(0.5 * (len(latencies) - 1))],
        "p95_s": latencies[round(0.95 * (len(latencies) - 1))],
        "max_s": latencies[-1],
        "generated_tokens": generated,
        "s_per_1k_generated": 1000 * wall / max(generated, 1),
    }


async def run_rounds(
    client: tinker.SamplingClient,
    prompts: list[list[int]],
    args: argparse.Namespace,
    run: wandb.Run | None,
) -> list[dict[str, float]]:
    rounds = []
    for index in range(args.rounds):
        result = await run_round(client, prompts, args)
        rounds.append(result)
        print(
            f"{index:5d}  {result['wall_s']:6.1f}  {result['p50_s']:5.1f}  "
            f"{result['p95_s']:5.1f}  {result['max_s']:5.1f}  "
            f"{result['generated_tokens']:10d}  {result['s_per_1k_generated']:8.3f}"
        )
        if run is not None:
            run.log(
                {f"rollout/{key}": value for key, value in result.items()}, step=index
            )
    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Spindle URL; leave out for Tinker")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model-path", help="tinker://.../sampler_weights/...")
    target.add_argument("--base-model")
    parser.add_argument("--prompts", required=True, help="JSONL file of prompts")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--copies", type=int, default=1, help="Copies sent at once, e.g. one per run"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", help="Write per-round results to this file")
    parser.add_argument("--wandb-project", help="Log each round to this W&B project")
    args = parser.parse_args()

    service = tinker.ServiceClient(
        **({"base_url": args.base_url} if args.base_url else {})
    )
    client = service.create_sampling_client(
        model_path=args.model_path, base_model=args.base_model
    )
    prompts = load_prompts(args.prompts, client)
    run = (
        wandb.init(project=args.wandb_project, config=vars(args))
        if args.wandb_project
        else None
    )
    print("round  wall_s  p50_s  p95_s  max_s  gen_tokens  s/1k_gen")
    rounds = asyncio.run(run_rounds(client, prompts, args, run))
    # The first round includes cold starts and adapter loading.
    steady = rounds[1:] or rounds
    wall = statistics.median(r["wall_s"] for r in steady)
    per_1k = statistics.median(r["s_per_1k_generated"] for r in steady)
    print(f"median after round 0: wall {wall:.1f} s, {per_1k:.3f} s per 1k generated")
    if run is not None:
        run.finish()
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as out:
            json.dump({"args": vars(args), "rounds": rounds}, out, indent=2)


if __name__ == "__main__":
    main()
