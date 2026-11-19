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


@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("updates", [3, 6])
def test_launcher_indexes_only_completed_guess_number(
    accepted: bool, updates: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = ROOT / "outputs" / f"test-guess-number-{updates}-{accepted}.jsonl"
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

    def forward(batch: list[tinker.Datum], loss: str) -> Future[Any]:
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

    training = SimpleNamespace(
        save_weights_for_sampler=lambda name: future(SimpleNamespace(path=name)),
        forward_backward=forward,
        optim_step=lambda adam: future(
            SimpleNamespace(metrics={"update_successful:mean": float(accepted)})
        ),
        save_state=lambda name: future(SimpleNamespace(path="test://final")),
    )
    service = SimpleNamespace(
        create_lora_training_client=lambda *args: training,
        create_sampling_client=lambda **kwargs: None,
    )
    monkeypatch.setattr(tinker, "ServiceClient", lambda: service)
    definition = "guess_number_gpt_oss_20b" + ("_6steps" if updates == 6 else "")
    argv = [
        "run.py",
        str(ROOT / "configs" / f"{definition}.toml"),
        "--output",
        str(output),
        "--index",
        str(index),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    def launch(*args: object) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "guess_number.py",
                "--base-model",
                "gpt-oss-20b-lora-32k",
                "--output",
                str(output),
                "--tokenizer",
                "openai/gpt-oss-20b",
                "a" * 40,
                "--updates",
                str(updates),
            ],
        )
        workload.main()

    monkeypatch.setattr(launcher, "launch", launch)
    try:
        if not accepted:
            with pytest.raises(ValueError, match="was rejected"):
                launcher.main()
            assert not index.exists()
            return
        launcher.main()
        catalog = launcher.ReferenceRun.model_validate(
            tomllib.loads(index.read_text())["runs"][0]
        )
        assert catalog.validation == definition
        assert catalog.trainer_image == "miles-v0.1.1-52a7a99"
        assert "1 x 1 H200" in catalog.hardware
        assert "0-1 x 1 H200" in catalog.hardware
        assert catalog.spindle_commit == "a" * 40
        records = importlib.import_module("catalog").read_output(output)
        assert records[0]["adam"]["eps"] == 1e-8
        measured = [row for row in records if row["event"] == "update"]
        assert [row["step"] for row in measured] == list(range(1, updates + 1))
        assert all(row["kl_sample_train_v2"] == 0.125 for row in measured)
        assert "seconds_since_previous_update" in measured[-1]
        assert [row["step"] for row in records if row["event"] == "evaluation"] == [
            0,
            updates,
        ]
        assert records[-1]["event"] == "checkpoint"
    finally:
        output.unlink(missing_ok=True)
