import asyncio
import json

import pytest
from tinker import types

from spindle.engine import Engine, FutureStatus
from spindle.usage import (
    DirectorySink,
    Rates,
    TrainerUsage,
    UsageMeter,
    UsageRow,
    SessionUsage,
    attribute_cost,
    billing_usage,
    labels_from,
)
from tests.support import EchoExecutor

# The illustrative per-1M-token rates from the cost attribution proposal.
RATES = Rates(train=0.40, prefill=0.20, cached_prefill=0.04, sample=0.50)
HOUR = 1_793_973_600.0  # 2026-11-06T14:00:00Z
SESSIONS = {
    "session-a": SessionUsage("alice", {"run_id": "A", "team_id": "t-ranking"}),
    "session-b": SessionUsage("bob", {"run_id": "B", "team_id": "t-ads"}),
}
MODELS = {"model-a": "session-a", "model-b": "session-b"}


def row(event_type, app_name, *, timestamp=HOUR, **values):
    return UsageRow(
        deployment="dd-prod",
        app_name=app_name,
        app_kind="sampler" if event_type.startswith("sampling") else "trainer",
        container_id="c1",
        base_model="Qwen/Qwen3-8B",
        event_type=event_type,
        timestamp=timestamp,
        event_uuid=f"{event_type}-{app_name}-{timestamp}-{json.dumps(values)}",
        **values,
    )


def shared_pool_rows(*, idle_s=0.0):
    trainer, sampler = "dd-prod-trainer-qwen3-8b", "dd-prod-inference-qwen3-8b"
    rows = [
        row("training", trainer, model_id="model-a", token_count=25_000_000),
        row("training", trainer, model_id="model-b", token_count=50_000_000),
        row(
            "sampling_prefill", sampler, session_id="session-a", token_count=10_000_000
        ),
        row(
            "sampling_prefill",
            sampler,
            session_id="session-a",
            cached=True,
            token_count=25_000_000,
        ),
        row("sampling_sample", sampler, session_id="session-a", token_count=4_000_000),
        row(
            "sampling_prefill", sampler, session_id="session-b", token_count=15_000_000
        ),
        row(
            "sampling_prefill",
            sampler,
            session_id="session-b",
            cached=True,
            token_count=50_000_000,
        ),
        row("sampling_sample", sampler, session_id="session-b", token_count=20_000_000),
    ]
    if idle_s:
        rows += [
            row("busy", trainer, seconds=3600 - idle_s),
            row("idle", trainer, seconds=idle_s),
        ]
    return rows, {trainer: 10.0, sampler: 4.0}


@pytest.mark.parametrize(
    ("group_by", "a", "b"),
    [("run_id", "A", "B"), ("user", "alice", "bob"), ("team_id", "t-ranking", "t-ads")],
)
@pytest.mark.parametrize(
    ("idle_s", "expected"),
    [
        # Trainer $10 splits 1:2 and sampler $4 splits $5:$15 of token price.
        (0.0, {"A": 10 / 3 + 1.0, "B": 20 / 3 + 3.0}),
        # A sixth of the trainer hour was idle: $1.67 is reported, not charged.
        (600.0, {"A": 25 / 9 + 1.0, "B": 50 / 9 + 3.0, "idle": 10 / 6}),
    ],
)
def test_shared_pool_cost_splits_by_token_price_and_sums_to_the_bill(
    group_by, a, b, idle_s, expected
) -> None:
    rows, app_costs = shared_pool_rows(idle_s=idle_s)
    expected = {{"A": a, "B": b}.get(name, name): usd for name, usd in expected.items()}

    costs = attribute_cost(
        rows, app_costs, RATES, sessions=SESSIONS, models=MODELS, group_by=group_by
    )

    assert costs.keys() == expected.keys()
    for group, usd in expected.items():
        assert costs[group] == pytest.approx(usd)
    assert sum(costs.values()) == pytest.approx(14.0)


def test_same_work_costs_the_same_across_the_window() -> None:
    trainer = "trainer"
    sessions = {f"s{name}": SessionUsage(labels={"run_id": name}) for name in "ABC"}
    rows = [
        row("training", trainer, session_id="sA", token_count=1_000_000),
        row("training", trainer, session_id="sB", token_count=1_000_000),
        row("training", trainer, session_id="sC", token_count=500_000),
        row(
            "training",
            trainer,
            session_id="sC",
            token_count=500_000,
            timestamp=HOUR + 3600,
        ),
    ]

    # $8 in hour 1 and $4 in hour 2, reported as one window.
    costs = attribute_cost(rows, {trainer: 12.0}, RATES, sessions=sessions, models={})

    assert costs == pytest.approx({"A": 4.0, "B": 4.0, "C": 4.0})


def test_cost_without_usage_is_unattributed() -> None:
    costs = attribute_cost([], {"trainer": 3.0}, RATES, sessions={}, models={})
    assert costs == {"unattributed": 3.0}


def test_cold_start_is_charged_to_the_model_that_started_the_trainer() -> None:
    rows = [
        row("cold_start", "trainer", model_id="model-a", seconds=300),
        row("busy", "trainer", seconds=600),
        row("training", "trainer", model_id="model-a", token_count=1_000_000),
        row("training", "trainer", model_id="model-b", token_count=1_000_000),
    ]

    costs = attribute_cost(
        rows, {"trainer": 9.0}, RATES, sessions=SESSIONS, models=MODELS
    )

    # $3 of start-up goes to A; the $6 of work splits evenly.
    assert costs == pytest.approx({"A": 6.0, "B": 3.0})


class FlakySink(DirectorySink):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.failures = 1

    def write(self, rows) -> None:
        rows = list(rows)
        super().write(rows)  # Lands on disk, then the commit fails.
        if self.failures:
            self.failures -= 1
            raise OSError("volume commit failed")


def test_meter_writes_closed_minutes_once_even_when_a_write_is_retried(
    tmp_path,
) -> None:
    now = [HOUR + 5]
    sink = FlakySink(tmp_path)
    meter = UsageMeter(
        sink, deployment="dd-prod", container_id="c1", clock=lambda: now[0]
    )

    def train(tokens):
        meter.add(
            "training",
            app_name="trainer",
            app_kind="trainer",
            base_model="Qwen/Qwen3-8B",
            model_id="model-a",
            tokens=tokens,
        )

    train(100)
    train(20)
    assert meter.flush() == 0  # The minute is still open.

    now[0] = HOUR + 61
    train(7)  # A new minute; it stays in memory.
    with pytest.raises(OSError):
        meter.flush()
    assert meter.flush() == 1

    rows = sink.read(HOUR, HOUR + 3600)
    assert [(r.timestamp, r.token_count) for r in rows] == [(HOUR, 120)]

    meter.close()
    assert sorted(r.token_count for r in sink.read(HOUR, HOUR + 3600)) == [7, 120]


def test_billing_usage_is_a_valid_tinker_response(tmp_path) -> None:
    sink = DirectorySink(tmp_path)
    rows, _ = shared_pool_rows()
    rows.append(
        row(
            "forward",
            "dd-prod-trainer-qwen3-8b",
            model_id="model-a",
            token_count=5_000_000,
        )
    )
    sink.write(rows)
    sink.put_model("model-a", "session-a")
    sink.put_model("model-b", "session-b")
    for session_id, session in SESSIONS.items():
        labels = labels_from(
            {**session.labels, "api_token": "x"}, ("run_id", "team_id")
        )
        sink.put_session(session_id, SessionUsage(session.user_id, labels), HOUR)

    usage = types.BillingUsageResponse.model_validate(
        billing_usage(sink, "2026-11-06T14:00:00Z", "2026-11-06T15:00:00Z")
    )

    totals = {
        (
            event.session_id,
            event.event_info.type,
            getattr(event.event_info, "cached", None),
        ): (event.event_info.token_count)
        for event in usage.data
    }
    # Forward-only tokens are reported as training.
    assert totals[("session-a", "training", None)] == 30_000_000
    assert totals[("session-b", "sampling_prefill", True)] == 50_000_000
    assert usage.sessions["session-a"].user_metadata == SESSIONS["session-a"].labels
    assert {event.session_id: event.user_id for event in usage.data} == {
        "session-a": "alice",
        "session-b": "bob",
    }
    assert [event.event_info.token_count for event in usage.data][:2] == [
        50_000_000,
        50_000_000,
    ]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2026-11-06T16:30:00Z", "2026-11-06T17:00:00Z", "UTC hour"),
        ("2026-11-06T17:00:00Z", "2026-11-06T16:00:00Z", "after"),
        ("2026-11-01T00:00:00Z", "2026-11-16T00:00:00Z", "14 days"),
    ],
)
def test_billing_usage_rejects_windows_tinker_rejects(
    tmp_path, start, end, message
) -> None:
    with pytest.raises(ValueError, match=message):
        billing_usage(DirectorySink(tmp_path), start, end)


class FailingExecutor(EchoExecutor):
    async def execute(self, model_id, kind, payload):
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("executor", "operation", "trained", "forwarded"),
    [
        (EchoExecutor, "forward_backward", [3], []),
        (EchoExecutor, "forward", [], [1]),
        (FailingExecutor, "forward_backward", [], []),
    ],
)
def test_engine_meters_tokens_of_completed_training_commands_only(
    tmp_path, executor, operation, trained, forwarded
) -> None:
    from tests.engine.test_server import forward, forward_backward

    sink = DirectorySink(tmp_path)
    meter = UsageMeter(sink, deployment="dd-prod", container_id="c1")

    async def run() -> FutureStatus:
        usage = TrainerUsage(
            meter, app_name="trainer", base_model="Qwen/Qwen3-8B", started_at=0
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

    rows = sink.read(0, 2**32)
    assert [r.token_count for r in rows if r.event_type == "training"] == trained
    assert [r.token_count for r in rows if r.event_type == "forward"] == forwarded
    assert [r.model_id for r in rows if r.event_type == "cold_start"] == ["model-a"]
    assert {r.event_type for r in rows} >= {"busy", "idle"}
