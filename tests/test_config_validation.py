from datetime import datetime, timezone

import pytest

import synthetic_monitoring.config_validation as cv
from synthetic_monitoring.chart import RunPoint


@pytest.fixture
def history(monkeypatch):
    store: dict = {}
    monkeypatch.setattr(cv, "_history_dict", lambda: store)
    return store


def _point(hour, day=12, status="success", startup=100):
    return RunPoint(
        ts=datetime(2026, 11, day, hour, tzinfo=timezone.utc).timestamp(),
        timings={"startup": startup},
        total_duration_s=1.0,
        status=status,
    )


def test_record_keeps_latest_success_per_night(history):
    cv._record("cfg", _point(10))
    cv._record("cfg", _point(12))
    assert [p["ts"] for p in history["cfg"]] == [_point(12).ts]


def test_record_failure_does_not_drop_success(history):
    cv._record("cfg", _point(10))
    cv._record("cfg", _point(12, status="failed"))
    assert [p["status"] for p in history["cfg"]] == ["success", "failed"]


def test_record_keeps_other_nights(history):
    cv._record("cfg", _point(10, day=11))
    cv._record("cfg", _point(10, day=12))
    assert [p["ts"] for p in history["cfg"]] == [
        _point(10, day=11).ts,
        _point(10, day=12).ts,
    ]


def test_record_dedupes_same_la_date_across_utc_midnight(history):
    cv._record("cfg", _point(23, day=11))  # 15:00 PT Nov 11
    cv._record("cfg", _point(5, day=12))  # 21:00 PT Nov 11
    assert [p["ts"] for p in history["cfg"]] == [_point(5, day=12).ts]


def test_record_keeps_different_la_dates_within_one_utc_day(history):
    cv._record("cfg", _point(7, day=12))  # 23:00 PT Nov 11
    cv._record("cfg", _point(8, day=12))  # 00:00 PT Nov 12
    assert [p["ts"] for p in history["cfg"]] == [
        _point(7, day=12).ts,
        _point(8, day=12).ts,
    ]


def test_row_delta_compares_previous_night(history):
    cv._record("cfg", _point(10, day=11, startup=100))
    cv._record("cfg", _point(10, day=12, startup=120))
    latest = _point(12, day=12, startup=130)
    row = cv._row("cfg", latest, cv._record("cfg", latest))
    assert row["delta"] == "startup: +30s"
