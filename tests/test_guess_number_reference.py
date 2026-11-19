"""Guess-number grading, conversation history and catalog completion."""

import importlib
import sys
import tomllib
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tinker
from pydantic import ValidationError

ROOT = Path(__file__).parents[1] / "reference_runs"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
workload = importlib.import_module("guess_number")
launcher = importlib.import_module("run")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Guess: 42", ("Correct", 1.0)),
        ("Guess: 41", ("Too low", 0.0)),
        ("Guess: 43", ("Too high", 0.0)),
        ("Guess: many", (workload.FORMAT_FEEDBACK, -1.0)),
        ("42", (workload.FORMAT_FEEDBACK, -1.0)),
    ],
)
def test_guess_feedback(response: str, expected: tuple[str, float]) -> None:
    assert workload.feedback(response, 42) == expected


def future[T](value: T) -> Future[T]:
    result = Future()
    result.set_result(value)
    return result


def sequence(token: int, *, valid: bool = True) -> tinker.types.SampledSequence:
    return tinker.types.SampledSequence(
        tokens_np=np.array([token], dtype=np.int32),
        logprobs_np=np.array([-1.0 if valid else float("nan")], dtype=np.float32),
        stop_reason="stop",
        sequence_id=str(token),
    )


@pytest.mark.parametrize("invalid_scores", [False, True])
def test_multiturn_history_and_format_penalty(invalid_scores: bool) -> None:
    histories = []
    responses = iter([sequence(i, valid=not invalid_scores) for i in range(4)])

    def prompt(messages: list[dict], **kwargs: object) -> list[int]:
        histories.append(list(messages))
        return [10, 11]

    tokenizer = SimpleNamespace(
        apply_chat_template=prompt,
        eos_token_id=0,
        decode=lambda tokens: ["bad", "Guess: 30", "Guess: 50", "Guess: 42"][tokens[0]],
    )
    sampler = SimpleNamespace(
        sample=lambda *args: future(SimpleNamespace(sequences=[next(responses)]))
    )
    config = workload.Config("model", "a" * 40)
    if invalid_scores:
        with pytest.raises(ValueError, match="invalid logprobs"):
            workload.play(sampler, tokenizer, 42, 0, config)
        return
    episode = workload.play(sampler, tokenizer, 42, 0, config)
    assert episode.solved and episode.reward == 0.0
    assert len(episode.turns) == 4
    assert [m["role"] for m in histories[-1]] == ["system"] + ["assistant", "user"] * 3
    assert [m["content"] for m in histories[-1][2::2]] == [
        workload.FORMAT_FEEDBACK,
        "Too low",
        "Too high",
    ]
    assert (
        workload.response_text(
            "<|channel|>analysis<|message|>Guess: 9<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Guess: 42<|return|>"
        )
        == "Guess: 42"
    )


@pytest.mark.parametrize(
    "mode", ["three", "six", "restore", "broken_restore", "rejected"]
)
def test_launcher_checkpoint_lifecycle(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = "guess_number_gpt_oss_20b" + (
        "_restore" if "restore" in mode else "_6steps" if mode == "six" else ""
    )
    parameters = tomllib.loads((ROOT / "configs" / f"{definition}.toml").read_text())[
        "workload"
    ]
    total = parameters["total_steps"]
    output = ROOT / "outputs" / f"test-guess-number-{mode}.jsonl"
    index = tmp_path / "index.toml"
    monkeypatch.setattr(
        launcher,
        "server_metadata",
        lambda: {"spindle_commit": "a" * 40, "spindle_version": "test"},
    )
    monkeypatch.setattr(
        workload.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **kw: None
    )

    def sample(*args: object) -> list[list[workload.Episode]]:
        return [
            [
                workload.Episode([([10, 11], sequence(12))], 1.0, True),
                workload.Episode([([10, 11], sequence(13))], 0.0, False),
            ]
        ]

    monkeypatch.setattr(workload, "sample", sample)

    class Training:
        def __init__(self) -> None:
            self.weight = self.moment = 0.0
            self.calls = 0
            self.saved: dict[str, tuple[float, float]] = {}
            self.publications: list[str] = []
            self.published_weights: dict[str, float] = {}

        def save_weights_for_sampler(self, name: str) -> Future[Any]:
            self.publications.append(name)
            self.published_weights[name] = self.weight
            return future(SimpleNamespace(path=name))

        def save_state(self, name: str) -> Future[Any]:
            path = f"test://{name}"
            self.saved[path] = self.weight, self.moment
            return future(SimpleNamespace(path=path))

        def load_state_with_optimizer(self, path: str) -> Future[None]:
            if mode != "broken_restore":
                self.weight, self.moment = self.saved[path]
            return future(None)

        def forward_backward(self, batch: list[tinker.Datum], loss: str) -> Future[Any]:
            assert loss == "importance_sampling"
            assert [row.loss_fn_inputs["advantages"].data for row in batch] == [
                [0.0, 0.5],
                [0.0, -0.5],
            ]
            return future(
                SimpleNamespace(
                    metrics={},
                    loss_fn_outputs=[
                        {
                            "logprobs": tinker.types.TensorData(
                                data=[100.0, -0.5], dtype="float32", shape=[2]
                            )
                        }
                        for _ in batch
                    ],
                )
            )

        def optim_step(self, adam: tinker.types.AdamParams) -> Future[Any]:
            self.calls += 1
            self.moment = 0.9 * self.moment + 1
            self.weight += 0.01 * self.moment
            return future(
                SimpleNamespace(
                    metrics={"update_successful:mean": float(mode != "rejected")}
                )
            )

    training = Training()

    class Sampler:
        def __init__(self, model_path: str) -> None:
            self.weight = training.published_weights[model_path]

        def compute_logprobs(
            self, prompt: tinker.types.ModelInput
        ) -> Future[list[float | None]]:
            return future([None, -1.0 + self.weight])

    service = SimpleNamespace(
        create_lora_training_client=lambda *args: training,
        create_sampling_client=lambda model_path: Sampler(model_path),
    )
    monkeypatch.setattr(tinker, "ServiceClient", lambda: service)

    def launch(*args: object) -> None:
        argv = [
            "guess_number.py",
            "--base-model",
            "gpt-oss-20b-lora-32k",
            "--output",
            str(output),
            "--tokenizer",
            "openai/gpt-oss-20b",
            "a" * 40,
        ]
        for field in ("total_steps", "save_step", "restore_step"):
            if field in parameters:
                argv.extend(["--" + field.replace("_", "-"), str(parameters[field])])
        monkeypatch.setattr(sys, "argv", argv)
        workload.main()

    monkeypatch.setattr(launcher, "launch", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            str(ROOT / "configs" / f"{definition}.toml"),
            "--output",
            str(output),
            "--index",
            str(index),
        ],
    )
    try:
        if mode in ("rejected", "broken_restore"):
            error = ValueError if mode == "rejected" else AssertionError
            with pytest.raises(
                error, match="was rejected" if mode == "rejected" else "Not equal"
            ):
                launcher.main()
            assert not index.exists()
            return
        launcher.main()
        catalog = launcher.ReferenceRun.model_validate(
            tomllib.loads(index.read_text())["runs"][0]
        )
        assert catalog.validation == definition
        assert catalog.trainer_image == "miles-v0.1.1-52a7a99"
        assert "0-1 x 1 H200" in catalog.hardware
        assert catalog.spindle_commit == "a" * 40
        records = importlib.import_module("catalog").read_output(output)
        assert records[0]["adam"]["eps"] == 1e-8
        updates = [r for r in records if r["event"] == "update"]
        retained = [r for r in updates if not r["discarded"]]
        assert [r["step"] for r in retained] == list(range(1, total + 1))
        assert [r["attempt"] for r in updates] == list(range(1, len(updates) + 1))
        assert all(r["kl_sample_train_v2"] == 0.125 for r in updates)
        assert [r["step"] for r in records if r["event"] == "evaluation"] == [0, total]
        assert records[-1]["event"] == "checkpoint"
        assert len(set(training.publications)) == len(training.publications)
        weight = moment = 0.0
        for _ in range(total):
            moment = 0.9 * moment + 1
            weight += 0.01 * moment
        assert (training.weight, training.moment) == pytest.approx((weight, moment))
        if mode == "restore":
            assert [r["step"] for r in updates] == [1, 2, 3, 4, 3, 4, 5, 6]
            assert [r["step"] for r in updates if r["discarded"]] == [3, 4]
            assert "seconds_since_previous_update" not in retained[2]
            restore = next(r for r in records if r["event"] == "restore_validation")
            assert (
                restore["step"],
                restore["from_step"],
                restore["max_abs_error"],
            ) == (2, 4, 0)
            assert restore["discarded_score_change"] > 0
            assert training.calls == 8
            plot = importlib.import_module("plot")
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "plot.py",
                    catalog.id,
                    "--index",
                    str(index),
                    "--output",
                    str(tmp_path / "plot.png"),
                ],
            )
            plot.main()
            axes = plot.pyplot.gcf().axes
            assert list(axes[1].lines[0].get_xdata()) == list(range(1, 7))
            assert np.isnan(axes[3].lines[0].get_ydata()[2])
            plot.pyplot.close("all")
    finally:
        output.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "parameters",
    [
        {"total_steps": "6"},
        {"total_steps": 0},
        {"total_steps": True},
        {"total_steps": 6, "save_step": 2},
        {"total_steps": 6, "save_step": 4, "restore_step": 2},
        {"total_steps": 6, "save_step": 2, "restore_step": 6},
    ],
)
def test_invalid_workload_parameters(parameters: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        launcher.WorkloadConfig.model_validate(
            {"tokenizer_model": "model", "tokenizer_revision": "a" * 40, **parameters}
        )
