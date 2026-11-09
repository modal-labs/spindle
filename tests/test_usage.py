import asyncio
from dataclasses import fields

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
    meter_sample,
)
from tests.support import EchoExecutor

# The illustrative per-1M-token rates from the cost attribution proposal.
RATES = Rates(train=0.40, prefill=0.20, cached_prefill=0.04, sample=0.50)
HOUR = 1_793_973_600.0  # 2026-11-06T14:00:00Z
SESSIONS = {
    "session-a": SessionUsage("alice", {"run_id": "A", "team_id": "t-ranking"}),
    "session-b": SessionUsage("bob", {"run_id": "B", "team_id": "t-ads"}),
}
TRAINER, SAMPLER = "dd-prod-trainer-qwen3-8b", "dd-prod-inference-qwen3-8b"


def row(app_name, *, timestamp=HOUR, uid=None, **values):
    return UsageRow(
        workspace_id="ws-1",
        environment_id="main",
        timestamp=timestamp,
        model_id="Qwen/Qwen3-8B",
        request_uuid=uid or f"{app_name}-{timestamp}-{sorted(values.items())}",
        deployment="dd-prod",
        app_name=app_name,
        app_kind="trainer" if app_name == TRAINER else "sampler",
        container_id="c1",
        **values,
    )


def shared_pool_rows(*, idle_s=0.0):
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
    if idle_s:
        rows.append(row(TRAINER, busy_seconds=3600 - idle_s, idle_seconds=idle_s))
    return rows, {TRAINER: 10.0, SAMPLER: 4.0}


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

    costs = attribute_cost(rows, app_costs, RATES, sessions=SESSIONS, group_by=group_by)

    assert costs.keys() == expected.keys()
    for group, usd in expected.items():
        assert costs[group] == pytest.approx(usd)
    assert sum(costs.values()) == pytest.approx(14.0)


def test_same_work_costs_the_same_across_the_window() -> None:
    sessions = {f"s{name}": SessionUsage(labels={"run_id": name}) for name in "ABC"}
    rows = [
        row(TRAINER, session_id="sA", train_tokens=1_000_000),
        row(TRAINER, session_id="sB", train_tokens=1_000_000),
        row(TRAINER, session_id="sC", train_tokens=500_000),
        row(TRAINER, session_id="sC", train_tokens=500_000, timestamp=HOUR + 3600),
    ]

    # $8 in hour 1 and $4 in hour 2, reported as one window.
    costs = attribute_cost(rows, {TRAINER: 12.0}, RATES, sessions=sessions)

    assert costs == pytest.approx({"A": 4.0, "B": 4.0, "C": 4.0})


def test_cost_without_usage_is_unattributed() -> None:
    costs = attribute_cost([], {TRAINER: 3.0}, RATES, sessions={})
    assert costs == {"unattributed": 3.0}


def test_cold_start_is_charged_to_the_session_that_started_the_trainer() -> None:
    rows = [
        row(TRAINER, session_id="session-a", cold_start_seconds=300),
        row(TRAINER, busy_seconds=600),
        row(TRAINER, session_id="session-a", train_tokens=1_000_000),
        row(TRAINER, session_id="session-b", train_tokens=1_000_000),
    ]

    costs = attribute_cost(rows, {TRAINER: 9.0}, RATES, sessions=SESSIONS)

    # $3 of start-up goes to A; the $6 of work splits evenly.
    assert costs == pytest.approx({"A": 6.0, "B": 3.0})


def test_rows_keep_shared_endpoint_column_names_and_types() -> None:
    # Overlap with EndpointTokenMetricRow, so a ClickHouse writer can reuse its shape.
    shared = {
        "workspace_id": "str",
        "environment_id": "str",
        "timestamp": "float",
        "model_id": "str",
        "request_uuid": "str",
        "prompt_tokens": "int",
        "completion_tokens": "int",
        "cached_prompt_tokens": "int",
    }
    columns = {field.name: field.type for field in fields(UsageRow)}
    assert {name: columns.get(name) for name in shared} == shared


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


def test_billing_usage_is_a_valid_tinker_response(tmp_path) -> None:
    sink = DirectorySink(tmp_path)
    rows, _ = shared_pool_rows()
    # Trainers know the training model, not the session; the reader resolves it.
    rows[0] = row(
        TRAINER,
        training_model_id="model-a",
        train_tokens=25_000_000,
    )
    sink.write(rows)
    sink.put_model("model-a", "session-a")
    for session_id, session in SESSIONS.items():
        labels = labels_from(
            {**session.labels, "api_token": "x"}, ("run_id", "team_id")
        )
        sink.put_session(session_id, SessionUsage(session.user_id, labels), HOUR)

    response = billing_usage(sink, "2026-11-06T14:00:00Z", "2026-11-06T15:00:00Z")
    usage = types.BillingUsageResponse.model_validate(response)

    # Nothing beyond Tinker's own fields, at any level.
    assert response.keys() <= types.BillingUsageResponse.model_fields.keys()
    for event in response["data"]:
        assert event.keys() <= types.BillingUsageEvent.model_fields.keys()
    for session in response["sessions"].values():
        assert session.keys() <= types.BillingUsageSession.model_fields.keys()

    totals = {
        (
            event.session_id,
            event.event_info.type,
            getattr(event.event_info, "cached", None),
        ): event.event_info.token_count
        for event in usage.data
    }
    assert totals == {
        ("session-a", "training", None): 25_000_000,
        ("session-b", "training", None): 50_000_000,
        ("session-a", "sampling_prefill", False): 10_000_000,
        ("session-a", "sampling_prefill", True): 25_000_000,
        ("session-a", "sampling_sample", None): 4_000_000,
        ("session-b", "sampling_prefill", False): 15_000_000,
        ("session-b", "sampling_prefill", True): 50_000_000,
        ("session-b", "sampling_sample", None): 20_000_000,
    }
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
        ("2999-01-01T00:00:00Z", "2999-01-01T01:00:00Z", "future"),
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
    from tests.engine.test_server import forward, forward_backward

    sink = DirectorySink(tmp_path)
    meter = UsageMeter(sink, deployment="dd-prod", container_id="c1")

    async def run() -> FutureStatus:
        usage = TrainerUsage(
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

    rows = sink.read(0, 2**32)
    assert [r.train_tokens for r in rows if r.train_tokens] == trained
    assert [r.training_model_id for r in rows if r.cold_start_seconds] == ["model-a"]
    assert sum(r.busy_seconds for r in rows) > 0
    assert sum(r.idle_seconds for r in rows) > 0


def test_cold_start_goes_to_the_model_that_triggered_it_not_the_first_accepted(
    tmp_path,
) -> None:
    sink = DirectorySink(tmp_path)
    meter = UsageMeter(sink, deployment="dd-prod", container_id="c1")
    usage = TrainerUsage(
        meter,
        app_name=TRAINER,
        base_model="Qwen/Qwen3-8B",
        started_at=0,
        cold_start_model="model-b",
    )

    usage.accepted("model-a")
    meter.close()

    rows = sink.read(0, 2**32)
    assert [r.training_model_id for r in rows if r.cold_start_seconds] == ["model-b"]
