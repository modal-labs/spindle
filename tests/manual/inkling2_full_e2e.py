"""Validate full Inkling weights with stock Tinker; set TINKER_BASE_URL/API_KEY."""

from collections.abc import Callable

import json
import math
import os
import time

import tinker
from tinker import types
from transformers import AutoTokenizer


def timed[T](label: str, call: Callable[[], T]) -> T:
    started = time.monotonic()
    result = call()
    print(
        json.dumps({"stage": label, "seconds": time.monotonic() - started}), flush=True
    )
    return result


def main() -> None:
    headers = {
        "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
        "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
    }
    service = tinker.ServiceClient(
        base_url=os.environ["TINKER_BASE_URL"],
        api_key=os.environ["TINKER_API_KEY"],
        default_headers=headers,
    )
    clients = [
        timed(
            f"create_adapter_{seed}",
            lambda seed=seed: service.create_lora_training_client(
                base_model="thinkingmachines/Inkling-2-Small",
                rank=8,
                seed=seed,
                train_mlp=False,
                train_attn=True,
                train_unembed=True,
            ),
        )
        for seed in (41, 42)
    ]
    a, b = clients
    tokenizer = AutoTokenizer.from_pretrained(
        "thinkingmachines/Inkling-2-Small",
        revision="1df2d97d45210d94f781f42a6521392e5a539e46",
    )
    tokens = tokenizer.encode("The capital of France is Paris.")
    prompt = types.ModelInput.from_ints(tokens[:-1])
    datum = types.Datum(
        model_input=prompt,
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )

    def loss(client: tinker.TrainingClient, backward: bool = False) -> float:
        method = client.forward_backward if backward else client.forward
        result = method([datum], "cross_entropy").result()
        value = result.metrics["loss:sum"] / (len(tokens) - 1)
        assert math.isfinite(value), result
        print(json.dumps({"loss": value, "backward": backward}), flush=True)
        return value

    def publish_and_sample(
        client: tinker.TrainingClient, name: str
    ) -> tinker.SamplingClient:
        publication = timed(
            name + "_publish", lambda: client.save_weights_for_sampler(name).result()
        )
        sampler = service.create_sampling_client(model_path=publication.path)
        result = timed(
            name + "_sample",
            lambda: sampler.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=16, temperature=0),
            ).result(),
        )
        sequence = result.sequences[0]
        assert sequence.tokens and sequence.logprobs
        assert all(math.isfinite(x) for x in sequence.logprobs)
        print(
            json.dumps({"publication": publication.path, "tokens": sequence.tokens}),
            flush=True,
        )
        return sampler

    before_b = timed("adapter_b_initial_forward", lambda: loss(b))
    publish_and_sample(b, "inkling-b-before")
    for step in range(2):
        timed(f"adapter_a_backward_{step}", lambda: loss(a, True))
        timed(
            f"adapter_a_step_{step}",
            lambda: a.optim_step(types.AdamParams(learning_rate=1e-4)).result(),
        )
    after_b = timed("adapter_b_isolation_forward", lambda: loss(b))
    assert abs(before_b - after_b) < 1e-5, (before_b, after_b)
    publish_and_sample(a, "inkling-a-trained")
    saved_loss = timed("adapter_a_saved_forward", lambda: loss(a))
    checkpoint = timed("save_state", lambda: a.save_state("inkling-a-state").result())
    timed("adapter_a_mutate_backward", lambda: loss(a, True))
    timed(
        "adapter_a_mutate_step",
        lambda: a.optim_step(types.AdamParams(learning_rate=1e-4)).result(),
    )
    timed(
        "restore_state_and_optimizer",
        lambda: a.load_state_with_optimizer(checkpoint.path).result(),
    )
    restored_loss = timed("adapter_a_restored_forward", lambda: loss(a))
    assert abs(saved_loss - restored_loss) < 1e-5, (saved_loss, restored_loss)
    timed("adapter_b_backward", lambda: loss(b, True))
    timed(
        "adapter_b_step",
        lambda: b.optim_step(types.AdamParams(learning_rate=1e-4)).result(),
    )
    publish_and_sample(b, "inkling-b-trained")
    print(json.dumps({"status": "passed", "checkpoint": checkpoint.path}), flush=True)


if __name__ == "__main__":
    main()
