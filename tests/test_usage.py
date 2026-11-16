import asyncio
from collections import defaultdict
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from tinker import types

from spindle.engine import Engine, FutureStatus
from spindle.engine import server as engine_server
from spindle.usage import (
    COUNTERS,
    DAY_S,
    PRUNE_AFTER_S,
    DirectorySink,
    ContainerUsage,
    UsageMeter,
    UsageRow,
    labels_from,
    meter_sample,
    poll_busy,
)
from spindle.usage.rows import HOUR_S, hour_start, read_rows
from spindle.usage.store import ROLLUP_DELAY_S
from tests.engine.test_server import forward, forward_backward
from tests.support import EchoExecutor

HOUR = 1_793_973_600.0  # 2026-11-06T14:00:00Z
SESSIONS = {
    "session-a": {"run_id": "A", "team_id": "t-1"},
    "session-b": {"run_id": "B", "team_id": "t-2"},
}
TRAINER, SAMPLER = "shared-trainer-qwen3-8b", "shared-lora-qwen3-8b"


def row(app_name, *, timestamp=HOUR, uid=None, **values):
    return UsageRow(
        timestamp=timestamp,
        model_id="Qwen/Qwen3-8B",
        request_uuid=uid or f"{app_name}-{timestamp}-{sorted(values.items())}",
        app_name=app_name,
        **values,
    )


def shared_pool_rows():
    rows = [
        row(TRAINER, session_id="session-a", train_tokens=25_000_000),
        row(TRAINER, session_id="session-b", train_tokens=50_000_000),
        row(
            SAMPLER,
            session_id="session-a",
            prompt_tokens=35_000_000,
            cached_prompt_tokens=25_000_000,
            completion_tokens=4_000_000,
        ),
        row(
            SAMPLER,
            session_id="session-b",
            prompt_tokens=65_000_000,
            cached_prompt_tokens=50_000_000,
            completion_tokens=20_000_000,
        ),
    ]
    return rows


def test_trained_tokens_include_image_chunks() -> None:
    added = []
    meter = SimpleNamespace(add=lambda **values: added.append(values))
    usage = ContainerUsage(meter, app_name=TRAINER, base_model="m", started_at=0)  # ty: ignore[invalid-argument-type]
    image = types.ImageChunk(data=b"x", format="png", expected_tokens=10)
    chunks = [types.EncodedTextChunk(tokens=[1, 2, 3]), image]
    datum = types.Datum(model_input=types.ModelInput(chunks=chunks), loss_fn_inputs={})

    usage.trained(
        "m-1", types.ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")
    )

    assert added[-1]["train_tokens"] == 13


def test_meter_writes_closed_minutes_once_even_when_a_write_is_retried(
    tmp_path,
) -> None:
    now = [HOUR + 5]
    failures = [1]

    def commit() -> None:
        if failures[0]:
            failures[0] -= 1
            raise OSError("volume commit failed")

    sink = DirectorySink(tmp_path, commit=commit, clock=lambda: now[0])
    meter = UsageMeter(sink, container_id="c1", clock=lambda: now[0])

    def sample(prompt, cached, output):
        meter_sample(
            meter,
            app_name=SAMPLER,
            base_model="Qwen/Qwen3-8B",
            session_id="session-a",
            stats={
                "prefill_tokens": prompt,
                "cached_prompt_tokens": cached,
                "generated_tokens": output,
            },
        )

    sample(100, 60, 10)
    sample(20, 0, 5)
    assert meter.flush() == 0  # The minute is still open.

    now[0] = HOUR + 61
    sample(7, 0, 1)  # A new minute; it stays in memory.
    with pytest.raises(OSError):
        meter.flush()
    # A file whose commit failed is removed, so it can't land after its hour's rollup.
    assert not list(tmp_path.rglob("*.jsonl"))
    assert meter.flush() == 1

    [first] = sink.read(HOUR, HOUR + 3600)
    assert (
        first.prompt_tokens,
        first.cached_prompt_tokens,
        first.completion_tokens,
    ) == (
        120,
        60,
        15,
    )

    meter.close()
    assert sorted(r.prompt_tokens for r in sink.read(HOUR, HOUR + 3600)) == [7, 120]


def test_volume_files_are_never_rewritten(tmp_path) -> None:
    sink = DirectorySink(tmp_path)
    sink.put_session("session-a", {"run_id": "A"})
    sink.put_model("model-a", "session-a")
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    sink.put_session("session-a", {"run_id": "B"})
    sink.put_model("model-a", "session-b")

    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == files


@pytest.mark.parametrize(
    ("put", "args", "written"),
    [
        ("put_model", ("model-a", "session-a"), "models/model-a.json"),
        ("put_session", ("session-a", {"run_id": "A"}), "sessions/session-a.json"),
    ],
)
def test_a_record_whose_commit_failed_is_committed_on_retry(
    tmp_path, put, args, written
) -> None:
    commits: list[bool] = []

    def commit() -> None:
        commits.append(len(commits) > 0)
        if not commits[-1]:
            raise OSError("volume commit failed")

    sink = DirectorySink(tmp_path, commit=commit)
    with pytest.raises(OSError):
        getattr(sink, put)(*args)
    assert not (tmp_path / written).exists()

    getattr(sink, put)(*args)

    assert commits == [False, True]
    assert (tmp_path / written).exists()


class FailingExecutor(EchoExecutor):
    async def execute(self, model_id, kind, payload):
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("executor", "operation", "trained"),
    [
        (EchoExecutor, "forward_backward", [3]),
        (EchoExecutor, "forward", [1]),
        (FailingExecutor, "forward_backward", []),
    ],
)
def test_engine_meters_tokens_of_completed_training_commands_only(
    tmp_path, executor, operation, trained
) -> None:
    sink = DirectorySink(tmp_path)
    meter = UsageMeter(sink, container_id="c1")

    async def run() -> FutureStatus:
        usage = ContainerUsage(
            meter, app_name=TRAINER, base_model="Qwen/Qwen3-8B", started_at=0
        )
        server = Engine(executor(), usage=usage)
        await server.accept_model("model-a", {})
        if operation == "forward":
            await forward(server, 1)
        else:
            await forward_backward(server, 1, data=[1, 2, 3])
        state = await server.retrieve_future("model-a:1", timeout=1.0)
        await server.close()
        return state.status

    asyncio.run(run())
    meter.close()

    hour = hour_start(time.time())
    rows = sink.read(hour - HOUR_S, hour + HOUR_S)
    assert [r.train_tokens for r in rows if r.train_tokens] == trained
    assert [r.training_model_id for r in rows if r.cold_start_seconds] == [None]
    assert sum(r.busy_seconds for r in rows) > 0
    assert sum(r.idle_seconds for r in rows) > 0


NEXT_DAY = HOUR + 10 * HOUR_S  # 2026-11-07T00:00:00Z
ROLLED_UP = NEXT_DAY + HOUR_S + ROLLUP_DELAY_S  # Both hours and 2026-11-06 are closed.
ON_TIME = row(SAMPLER, session_id="session-b", prompt_tokens=50, completion_tokens=5)


def stored_usage(tmp_path):
    """Labeled usage in an hour on each side of a day boundary, written on time."""
    clock = [HOUR + 180]
    sink = DirectorySink(tmp_path, clock=lambda: clock[0])
    sink.put_model("model-a", "session-a")
    sink.put_session("session-a", SESSIONS["session-a"])
    sink.put_session("session-b", SESSIONS["session-b"])
    sink.write(
        [row(TRAINER, training_model_id="model-a", train_tokens=100, busy_seconds=60)]
    )
    sink.write([ON_TIME])
    sink.write([ON_TIME])  # A retry of the same minute.
    sink.write(
        [row(SAMPLER, timestamp=HOUR + 60, session_id="session-a", prompt_tokens=7)]
    )
    clock[0] = NEXT_DAY + 120
    sink.write(
        [row(TRAINER, timestamp=NEXT_DAY, cold_start_seconds=30, idle_seconds=20)]
    )
    return sink, clock


def answers(sink) -> dict[tuple, float]:
    """Every counter's total per app and session."""
    totals: dict[tuple, float] = defaultdict(float)
    for r in sink.read(HOUR, NEXT_DAY + HOUR_S):
        for name in COUNTERS:
            totals[(r.app_name, r.session_id, name)] += getattr(r, name)
    return dict(totals)


def token_total(sink) -> int:
    return sum(
        r.prompt_tokens + r.completion_tokens + r.train_tokens
        for r in sink.read(HOUR, NEXT_DAY + HOUR_S)
    )


def test_compaction_rolls_up_closed_hours_and_days_without_changing_answers(
    tmp_path,
) -> None:
    sink, _ = stored_usage(tmp_path)
    before = answers(sink)

    assert sink.compact(now=ROLLED_UP) == 3  # Two hour rollups and one day rollup.
    assert [p.name for p in (tmp_path / "days").iterdir()] == ["2026-11-06.jsonl"]
    assert sorted(p.name for p in (tmp_path / "hours").iterdir()) == [
        "2026-11-06T14.jsonl",
        "2026-11-07T00.jsonl",
    ]
    assert answers(sink) == pytest.approx(before)
    assert sink.compact(now=ROLLED_UP) == 0


def test_late_rows_are_counted_once(tmp_path) -> None:
    sink, clock = stored_usage(tmp_path)
    sink.compact(now=ROLLED_UP)
    before = token_total(sink)

    # A container that couldn't reach the Volume for hours.
    clock[0] = ROLLED_UP
    sink.write(
        [row(SAMPLER, timestamp=HOUR + 120, session_id="session-a", prompt_tokens=1)]
    )
    assert (tmp_path / "late").is_dir()
    assert token_total(sink) == before + 1

    # A late retry of a row already in the hour's rollup.
    sink.write([ON_TIME])
    assert token_total(sink) == before + 1


def test_pruning_keeps_answers_and_only_uncounted_late_rows(tmp_path) -> None:
    sink, clock = stored_usage(tmp_path)
    sink.compact(now=ROLLED_UP)
    clock[0] = ROLLED_UP
    late = row(SAMPLER, timestamp=HOUR + 120, session_id="session-a", prompt_tokens=1)
    sink.write([ON_TIME])  # Only a duplicate.
    sink.write([ON_TIME, late])  # A duplicate and an uncounted row.
    before = answers(sink)

    later = NEXT_DAY + DAY_S + PRUNE_AFTER_S + HOUR_S
    clock[0] = later
    sink.compact(now=later)

    assert not any((tmp_path / "rows").iterdir())
    kept = [r for path in (tmp_path / "late").rglob("*.jsonl") for r in read_rows(path)]
    assert kept == [late]
    assert answers(sink) == pytest.approx(before)


def test_engine_meters_idle_time_while_it_waits(monkeypatch) -> None:
    monkeypatch.setattr(engine_server, "WINDOW_S", 0.01)

    class Recorder:
        idle_s = 0.0

        def trained(self, model_id, payload) -> None:
            pass

        def busy(self, seconds) -> None:
            pass

        def idle(self, seconds) -> None:
            self.idle_s += seconds

    async def run() -> float:
        usage = Recorder()
        server = Engine(EchoExecutor(), usage=usage)
        await server.accept_model("model-a", {})
        await asyncio.sleep(0.3)
        metered = usage.idle_s
        await server.close()
        return metered

    assert asyncio.run(run()) > 0.05


def test_labels_keep_only_allowlisted_values_and_warn_on_dropped_ones(caplog) -> None:
    metadata = {"run_id": "r1", "team_id": "x" * 257, "api_token": "secret"}

    labels = labels_from(metadata, ("run_id", "team_id"))

    assert labels == {"run_id": "r1"}
    assert "Dropped usage labels ['team_id']" in caplog.text


def test_final_flush_runs_while_another_thread_is_still_busy(tmp_path) -> None:
    # A trainer's serving thread is still running when its container is told to stop.
    script = f"""
import threading, time
from pathlib import Path
from spindle.usage import DirectorySink, UsageMeter

root = Path({str(tmp_path)!r})
meter = UsageMeter(DirectorySink(root)).start()
meter.add(app_name="t", model_id="m", train_tokens=5)

def busy():
    deadline = time.time() + 10
    while time.time() < deadline and not list(root.rglob("*.jsonl")):
        time.sleep(0.05)
    print("flushed" if list(root.rglob("*.jsonl")) else "not flushed", flush=True)

threading.Thread(target=busy).start()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )

    assert result.stdout.strip() == "flushed", result.stderr


def test_reads_must_start_and_end_on_an_hour(tmp_path) -> None:
    # Rollups are stamped with their hour, so a mid-hour range would change once rolled up.
    with pytest.raises(ValueError, match="on an hour"):
        DirectorySink(tmp_path).read(HOUR + 1800, HOUR + 3600)


def test_poll_busy_records_busy_and_idle_time_and_skips_unknown_intervals(
    tmp_path,
) -> None:
    sink = DirectorySink(tmp_path)
    meter = UsageMeter(sink, container_id="pool-1")
    usage = ContainerUsage(meter, app_name=SAMPLER, base_model="m", started_at=0)
    answers = iter([True, False, RuntimeError("metrics down"), False])
    done = threading.Event()

    def busy() -> bool:
        answer = next(answers, None)
        if answer is None:
            done.set()
            return False
        if isinstance(answer, Exception):
            raise answer
        return answer

    stop = poll_busy(usage, busy, interval=0.05)
    assert done.wait(5)
    stop.set()
    meter.flush(include_open=True)

    hour = hour_start(time.time())
    rows = sink.read(hour - HOUR_S, hour + HOUR_S)
    total = {name: sum(getattr(r, name) for r in rows) for name in COUNTERS}
    # A loaded machine can only make intervals longer.
    assert total["busy_seconds"] >= 0.04
    assert total["idle_seconds"] >= 0.08
