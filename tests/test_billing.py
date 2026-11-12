import pytest
from tinker import types

from spindle.usage import DirectorySink, TooMuchUsage, billing, billing_usage
from tests.test_usage import SESSIONS, TRAINER, row, shared_pool_rows


def test_billing_usage_is_a_valid_tinker_response(tmp_path) -> None:
    sink = DirectorySink(tmp_path)
    rows = shared_pool_rows()
    # Trainers know the training model, not the session; the reader resolves it.
    rows[0] = row(
        TRAINER,
        training_model_id="model-a",
        train_tokens=25_000_000,
    )
    sink.write(rows)
    sink.put_model("model-a", "session-a")
    for session_id, session in SESSIONS.items():
        sink.put_session(session_id, session)

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
    assert usage.sessions["session-a"].user_metadata == SESSIONS["session-a"]
    order = [(e.bucket_start, -e.event_info.token_count) for e in usage.data]
    assert order == sorted(order)


def test_a_window_with_too_many_events_asks_for_a_shorter_one(
    tmp_path, monkeypatch
) -> None:
    sink = DirectorySink(tmp_path)
    sink.write(shared_pool_rows())
    monkeypatch.setattr(billing, "MAX_EVENTS", 3)

    with pytest.raises(TooMuchUsage, match="ask for a shorter window"):
        billing_usage(sink, "2026-11-06T14:00:00Z", "2026-11-06T15:00:00Z")


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
