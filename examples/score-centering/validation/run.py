"""Real sampler/reward/SDK/server/native-training validation for PR 173."""

import json
import math
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import spindle
import tinker
from tinker import types

from spindle.version import source_commit

SOURCE = (
    Path(__file__).resolve().parents[3] / "src" / "spindle"
    if modal.is_local()
    else Path("/root/spindle")
)
FRONTEND = os.environ.get("SPINDLE_VALIDATION_FRONTEND", "score-centering-validation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "modal==1.6.3",
        "tinker==0.25.0",
        "numpy",
        "httpx",
        "pydantic>=2.13.4",
        "transformers[chat-template]==5.17.0",
        "huggingface-hub",
    )
    .env(
        {
            "SPINDLE_COMMIT": source_commit() or "unknown",
            "SPINDLE_VALIDATION_FRONTEND": FRONTEND,
        }
    )
    .add_local_dir(
        SOURCE, "/root/spindle", ignore=["**/__pycache__/**", "**/skills/**"]
    )
)
volume = modal.Volume.from_name(
    f"{FRONTEND}-results", create_if_missing=True, version=2
)
app = modal.App(f"{FRONTEND}-driver")


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=14400,
    secrets=[
        modal.Secret.from_name("spindle-api"),
        modal.Secret.from_name("spindle-proxy"),
    ],
    volumes={"/results": volume},
)
def validate(case: str, url: str):
    root = Path("/results") / case
    root.mkdir(parents=True, exist_ok=True)

    def log(event, **fields):
        row = dict(time=time.time(), case=case, event=event, **fields)
        with (root / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        volume.commit()
        print(json.dumps(row, default=str), flush=True)

    def finite(value):
        if isinstance(value, dict):
            for x in value.values():
                finite(x)
        elif isinstance(value, (list, tuple)):
            for x in value:
                finite(x)
        elif isinstance(value, float):
            assert math.isfinite(value), value

    try:
        log("start", commit=source_commit())
        service = tinker.ServiceClient(
            base_url=url,
            api_key=os.environ["TINKER_API_KEY"],
            default_headers={
                "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
                "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
            },
        )
        log("creating_training_client")
        training = (
            spindle.create_full_training_client(service, case)
            if case == "fft4b"
            else service.create_lora_training_client(base_model=case, rank=16)
        )
        log("training_client_ready")
        tokenizer = training.get_tokenizer()
        log("publishing_initial_weights")
        sampler = training.save_weights_and_get_sampling_client()
        log("sampler_ready")
        probe = None
        before = None
        for step, loss in enumerate(
            ["score_centered_mis"] * 3 + ["score_centered"] * 3 + ["mis"] * 3 + ["ppo"]
        ):
            start = time.monotonic()
            log("sampling", step=step, loss=loss)

            def rollout(i):
                a, b, c = 17 + step * 3 + i, 7 + i, 11 + 2 * i
                expected = a * b + c
                text = f"Compute {a} * {b} + {c}. Reply with only the final integer."
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                params = types.SamplingParams(
                    max_tokens=64,
                    temperature=1.0,
                    top_k=64 if loss != "ppo" else -1,
                    top_p=0.95 if loss != "ppo" else 1.0,
                )
                if loss != "ppo":
                    response = spindle.sample_with_replay(
                        sampler,
                        types.ModelInput.from_ints(prompt),
                        sampling_params=params,
                        num_samples=2,
                        return_sampling_probabilities=True,
                    ).result(timeout=3600)
                else:
                    response = sampler.sample(
                        prompt=types.ModelInput.from_ints(prompt),
                        sampling_params=params,
                        num_samples=2,
                    ).result(timeout=3600)
                return prompt, expected, response.sequences

            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(rollout, range(8)))
            samples = []
            data = []
            total_tokens = sum(len(s.tokens) for _, _, seqs in rows for s in seqs)
            assert total_tokens > 0
            for prompt, expected, sequences in rows:
                for seq in sequences:
                    assert seq.tokens
                    text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                    numbers = re.findall(r"-?\d+", text)
                    reward = float(bool(numbers) and int(numbers[-1]) == expected)
                    tokens = prompt + list(seq.tokens)
                    n_prompt = len(prompt) - 1
                    inputs = {
                        "target_tokens": tokens[1:],
                        "advantages": [0.0] * n_prompt
                        + [(reward - 0.5) / total_tokens] * len(seq.tokens),
                    }
                    if loss != "ppo":
                        assert seq.replay.sampling_mask_logprobs is not None
                        finite(seq.replay.sampling_mask_logprobs)
                        inputs.update(seq.replay.training_inputs(len(seq.tokens)))
                    else:
                        assert seq.logprobs is not None
                        finite(seq.logprobs)
                        inputs["logprobs"] = [0.0] * n_prompt + list(seq.logprobs)
                    if len(data) % 2:
                        inputs["weights"] = [0.0] * n_prompt + [1.0] * len(seq.tokens)
                    data.append(
                        types.Datum(
                            model_input=types.ModelInput.from_ints(tokens[:-1]),
                            loss_fn_inputs=inputs,
                        )
                    )
                    samples.append(
                        dict(
                            response=text,
                            expected=expected,
                            reward=reward,
                            tokens=len(seq.tokens),
                        )
                    )
                    if probe is None:
                        probe = tokens
            if before is None:
                before = sampler.compute_logprobs(
                    types.ModelInput.from_ints(probe)
                ).result(timeout=3600)
                log("initial_probe", logprobs=before)
            log(
                "forward_backward",
                step=step,
                loss=loss,
                samples=samples,
                response_tokens=total_tokens,
            )
            config = {} if loss == "ppo" else {"sampling_temperature": 1.0}
            fb = training.forward_backward(data, loss, loss_fn_config=config).result(
                timeout=3600
            )
            finite(fb.metrics)
            optim = training.optim_step(types.AdamParams(learning_rate=1e-6)).result(
                timeout=3600
            )
            finite(optim.metrics)
            assert optim.metrics["update_successful:mean"] == 1.0
            assert optim.metrics["grad_norm:mean"] > 0.0
            log(
                "optimizer_complete",
                step=step,
                loss=loss,
                metrics=fb.metrics,
                optimizer_metrics=optim.metrics,
                seconds=time.monotonic() - start,
            )
            sampler = training.save_weights_and_get_sampling_client()
            log("weights_published", step=step)
        after = sampler.compute_logprobs(types.ModelInput.from_ints(probe)).result(
            timeout=3600
        )
        deltas = [
            abs(x - y) for x, y in zip(before, after) if x is not None and y is not None
        ]
        finite(deltas)
        assert deltas and max(deltas) > 0, (
            "Published model did not change probe probabilities"
        )
        log(
            "probe_changed",
            max_abs_logprob_delta=max(deltas),
            mean_abs_logprob_delta=sum(deltas) / len(deltas),
        )
        checkpoint = training.save_state("validation-final").result(timeout=3600).path
        log("checkpoint_saved", path=checkpoint)
        training.load_state_with_optimizer(checkpoint).result(timeout=3600)
        log("checkpoint_loaded")
        reloaded = training.save_weights_and_get_sampling_client()
        response = reloaded.sample(
            prompt=types.ModelInput.from_ints(probe[:16]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=8, temperature=1.0),
        ).result(timeout=3600)
        assert response.sequences[0].tokens
        log("passed", updates=10, checkpoint=checkpoint)
        return {"case": case, "passed": True, "updates": 10}
    except Exception:
        log("failed", traceback=traceback.format_exc())
        raise


@app.local_entrypoint()
def main(case: str, url: str):
    if case not in {"fft4b", "m111", "m112cp2"}:
        raise ValueError("case must be fft4b, m111, or m112cp2")
    print(validate.remote(case, url))
