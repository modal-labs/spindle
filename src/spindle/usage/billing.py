"""`GET /api/v1/billing/usage/events`: usage in Tinker's `BillingUsageResponse` format."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from .rows import HOUR_S, UsageRow, hour_start
from .store import DirectorySink

MAX_QUERY_WINDOW = timedelta(days=14)
# About 25 MB of JSON; a longer response is built in memory and sent in one piece.
MAX_EVENTS = 100_000


class TooMuchUsage(ValueError):
    """The window has more events than one response returns."""


_EVENTS: tuple[tuple[dict[str, Any], Callable[[UsageRow], int]], ...] = (
    ({"type": "training"}, lambda row: row.train_tokens),
    (
        {"type": "sampling_prefill", "cached": False},
        lambda row: row.prompt_tokens - row.cached_prompt_tokens,
    ),
    (
        {"type": "sampling_prefill", "cached": True},
        lambda row: row.cached_prompt_tokens,
    ),
    ({"type": "sampling_sample"}, lambda row: row.completion_tokens),
)


def billing_usage(
    sink: DirectorySink, starting_on: str | datetime, ending_before: str | datetime
) -> dict[str, object]:
    """Hourly token usage per session, shaped like Tinker's `BillingUsageResponse`."""
    start, end = usage_window(starting_on, ending_before)
    totals: dict[tuple, int] = defaultdict(int)
    for row in sink.read(start.timestamp(), end.timestamp()):
        for event, (_, tokens) in enumerate(_EVENTS):
            key = (hour_start(row.timestamp), row.model_id, row.session_id, event)
            totals[key] += tokens(row)
    data: list[dict[str, Any]] = [
        {
            "bucket_start": _iso(hour),
            "bucket_end": _iso(hour + HOUR_S),
            "base_model": model_id,
            "session_id": session_id,
            "event_info": {**_EVENTS[event][0], "token_count": count},
        }
        for (hour, model_id, session_id, event), count in totals.items()
        if count > 0
    ]
    if len(data) > MAX_EVENTS:
        raise TooMuchUsage(
            f"this window has {len(data)} usage events, more than the {MAX_EVENTS} "
            "one response returns; ask for a shorter window"
        )
    data.sort(
        key=lambda item: (item["bucket_start"], -item["event_info"]["token_count"])
    )
    session_ids = dict.fromkeys(
        item["session_id"] for item in data if item["session_id"]
    )
    labels = sink.sessions(session_ids)
    return {
        "data": data,
        "sessions": {
            session_id: {"user_metadata": labels.get(session_id) or None}
            for session_id in session_ids
        },
    }


def usage_window(
    starting_on: str | datetime, ending_before: str | datetime
) -> tuple[datetime, datetime]:
    """The window's start and end in UTC, with the limits Tinker applies."""
    start, end = _datetime(starting_on), _datetime(ending_before)
    for value in (start, end):
        if value.minute or value.second or value.microsecond:
            raise ValueError("billing usage windows must align to a UTC hour")
    if start > datetime.now(UTC):
        raise ValueError("starting_on must not be in the future")
    if end <= start:
        raise ValueError("ending_before must be after starting_on")
    if end - start > MAX_QUERY_WINDOW:
        raise ValueError("billing usage windows are at most 14 days")
    return start, end


def _datetime(value: str | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")
