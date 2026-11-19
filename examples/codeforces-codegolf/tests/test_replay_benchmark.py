import dataclasses
import json

import pytest

from benchmark_streaming_replay import load_cases, probe_difference, summarize
from codegolf.config import BatchConfig


@pytest.fixture
def replay(tmp_path):
    cfg = dataclasses.replace(BatchConfig(), prompts_per_step=2, group_size=2)
    (tmp_path / "spec.json").write_text(json.dumps({"config": dataclasses.asdict(cfg)}))
    for step in (3, 4, 5):
        folder = tmp_path / f"rollouts/step-{step:04d}"
        folder.mkdir(parents=True)
        for index in range(2):
            record = {
                "problem_id": str(index),
                "prompt": [10, 11],
                "sampling_seconds": index + 1.0,
                "judge_seconds": 0.1,
                "rows": [
                    {
                        "tokens": [20] * length,
                        "logprobs": [-0.2] * length,
                        "reward": float(reward),
                    }
                    for length, reward in ((2, 0), (3, 1))
                ],
            }
            (folder / f"{index}.json").write_text(json.dumps(record))
    return tmp_path


@pytest.mark.parametrize("field", ["tokens", "sampling_seconds"])
def test_replay_fingerprints_training_inputs_and_readiness_independently(replay, field):
    _, before = load_cases(replay)
    path = replay / "rollouts/step-0003/0.json"
    record = json.loads(path.read_text())
    if field == "tokens":
        record["rows"][0]["tokens"][0] += 1
    else:
        record[field] += 0.5
    path.write_text(json.dumps(record))
    _, after = load_cases(replay)
    changed = "datums_sha256" if field == "tokens" else "schedule_sha256"
    unchanged = "schedule_sha256" if field == "tokens" else "datums_sha256"
    assert before[0][changed] != after[0][changed]
    assert before[0][unchanged] == after[0][unchanged]
    assert before[1:] == after[1:]


@pytest.mark.parametrize(
    "failure", ["missing_group", "partial_group", "unaligned", "nan", "duplicate"]
)
def test_replay_rejects_incomplete_or_invalid_work(replay, failure):
    path = replay / "rollouts/step-0003/0.json"
    record = json.loads(path.read_text())
    if failure == "missing_group":
        path.unlink()
    else:
        if failure == "partial_group":
            record["rows"].pop()
        elif failure == "unaligned":
            record["rows"][0]["logprobs"].pop()
        elif failure == "nan":
            record["sampling_seconds"] = float("nan")
        else:
            record["problem_id"] = "1"
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_cases(replay)


def paired_rows():
    return [
        {
            "pair_id": f"{case}:{profile}:{repeat}",
            "case": case,
            "profile": profile,
            "mode": mode,
            "seconds": 10.0 + repeat - (2.0 if mode == "stream" else 0),
            "grad_norm": 0.5,
            "datums_sha256": case,
            "schedule_sha256": profile,
        }
        for case in ("a", "b", "c")
        for profile in ("all_ready", "recorded")
        for repeat in range(2)
        for mode in ("batch", "stream")
    ]


def test_paired_summary_retains_every_case_and_reports_signed_seconds():
    summary = summarize(paired_rows(), 2, ["a", "b", "c"])
    for result in summary.values():
        assert result["pairs"] == 6
        assert result["median_batch_seconds"] == 10.5
        assert result["median_stream_seconds"] == 8.5
        assert result["mean_paired_saving_seconds"] == 2.0
        assert result["paired_saving_ci95_seconds"] == [2.0, 2.0]
        assert result["per_case_mean_saving_seconds"] == {"a": 2.0, "b": 2.0, "c": 2.0}


def test_timing_only_summary_still_requires_identical_inputs():
    rows = paired_rows()
    rows[0]["grad_norm"] *= 2
    summary = summarize(rows, 2, ["a", "b", "c"], require_equivalence=False)
    assert summary["recorded"]["pairs"] == summary["all_ready"]["pairs"] == 6
    rows[0]["datums_sha256"] = "different"
    with pytest.raises(ValueError, match="inputs or readiness schedules differ"):
        summarize(rows, 2, ["a", "b", "c"], require_equivalence=False)


@pytest.mark.parametrize(
    "failure",
    ["duplicate", "missing_mode", "missing_case", "inputs", "schedule", "gradient"],
)
def test_summary_rejects_unmatched_or_numerically_different_pairs(failure):
    rows = paired_rows()
    if failure == "duplicate":
        rows.append(rows[0])
    elif failure == "missing_mode":
        rows.pop(0)
    elif failure == "missing_case":
        rows = [row for row in rows if row["case"] != "c"]
    elif failure == "inputs":
        rows[0]["datums_sha256"] = "different"
    elif failure == "schedule":
        rows[0]["schedule_sha256"] = "different"
    else:
        rows[0]["grad_norm"] *= 2
    with pytest.raises(ValueError):
        summarize(rows, 2, ["a", "b", "c"])


@pytest.mark.parametrize("second", [[], [[1.0]], [[1.0, float("nan")]]])
def test_probe_rejects_missing_or_invalid_outputs(second):
    with pytest.raises(ValueError):
        probe_difference([[1.0, 2.0]], second)
