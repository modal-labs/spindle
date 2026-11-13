import pytest

from spindle.usage import attribute_cost
from tests.test_usage import HOUR, SAMPLER, SESSIONS, TRAINER, row, shared_pool_rows

APP_COSTS = {TRAINER: 10.0, SAMPLER: 4.0}


@pytest.mark.parametrize(
    ("idle_s", "expected"),
    [
        # Trainer $10 splits 25M:50M train tokens; sampler $4 splits 14M:35M uncached tokens.
        (0.0, {"A": 10 / 3 + 4 * 14 / 49, "B": 20 / 3 + 4 * 35 / 49}),
        # A sixth of the trainer hour was idle: $1.67 is reported, not charged.
        (
            600.0,
            {"A": 25 / 9 + 4 * 14 / 49, "B": 50 / 9 + 4 * 35 / 49, "idle": 10 / 6},
        ),
    ],
)
def test_shared_pool_cost_splits_by_tokens_and_sums_to_the_bill(
    idle_s, expected
) -> None:
    rows = shared_pool_rows()
    if idle_s:
        rows.append(row(TRAINER, busy_seconds=3600 - idle_s, idle_seconds=idle_s))

    costs = attribute_cost(rows, APP_COSTS, sessions=SESSIONS)

    assert costs.keys() == expected.keys()
    for group, usd in expected.items():
        assert costs[group] == pytest.approx(usd)
    assert sum(costs.values()) == pytest.approx(14.0)


def test_same_work_costs_the_same_across_the_window() -> None:
    sessions = {f"s{name}": {"run_id": name} for name in "ABC"}
    rows = [
        row(TRAINER, session_id="sA", train_tokens=1_000_000),
        row(TRAINER, session_id="sB", train_tokens=1_000_000),
        row(TRAINER, session_id="sC", train_tokens=500_000),
        row(TRAINER, session_id="sC", train_tokens=500_000, timestamp=HOUR + 3600),
    ]

    # $8 in hour 1 and $4 in hour 2, reported as one window.
    costs = attribute_cost(rows, {TRAINER: 12.0}, sessions=sessions)

    assert costs == pytest.approx({"A": 4.0, "B": 4.0, "C": 4.0})


def test_cost_nobody_can_be_charged_for_is_its_own_line() -> None:
    rows = [
        # A session with no stored labels, and a model whose owner is unknown.
        row(TRAINER, session_id="session-x", train_tokens=1_000_000),
        row(TRAINER, training_model_id="model-x", train_tokens=1_000_000),
        # Busy with no tokens to split by, such as saving weights.
        row(f"{TRAINER}-2", busy_seconds=60),
        # Every prompt token came from the cache.
        row(SAMPLER, session_id="session-a", prompt_tokens=5, cached_prompt_tokens=5),
    ]
    app_costs = {TRAINER: 2.0, f"{TRAINER}-2": 1.0, SAMPLER: 3.0, "frontend": 0.5}

    costs = attribute_cost(rows, app_costs, sessions={})

    assert costs == {"unknown": 2.0, "unattributed": 4.5}


def test_a_label_named_like_a_summary_line_stays_its_own_group() -> None:
    rows = [row(TRAINER, session_id="session-a", train_tokens=1, idle_seconds=10)]
    rows.append(row(TRAINER, busy_seconds=10))

    costs = attribute_cost(
        rows, {TRAINER: 2.0}, sessions={"session-a": {"run_id": "idle"}}
    )

    assert costs == pytest.approx({"idle": 1.0, "idle (run_id)": 1.0})


def test_cold_start_is_reported_apart_from_the_runs_on_the_trainer() -> None:
    rows = [
        row(TRAINER, cold_start_seconds=300),
        row(TRAINER, busy_seconds=600),
        row(TRAINER, session_id="session-a", train_tokens=1_000_000),
        row(TRAINER, session_id="session-b", train_tokens=1_000_000),
    ]

    costs = attribute_cost(rows, {TRAINER: 9.0}, sessions=SESSIONS)
    by_team = attribute_cost(
        rows, {TRAINER: 9.0}, sessions=SESSIONS, group_by="team_id"
    )

    # $3 of start-up is no run's; the $6 of work splits evenly.
    assert costs == pytest.approx({"cold_start": 3.0, "A": 3.0, "B": 3.0})
    assert by_team == pytest.approx({"cold_start": 3.0, "t-1": 3.0, "t-2": 3.0})
