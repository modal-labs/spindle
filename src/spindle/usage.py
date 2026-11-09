"""Per-session token usage, for splitting shared trainer and sampler cost by run."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from spindle.telemetry.trainer import workload

log = logging.getLogger(__name__)

DEFAULT_LABEL_KEYS = ("run_id", "attempt_id")
WINDOW_S = 60
MAX_QUERY_WINDOW = timedelta(days=14)
_ROW_NAMESPACE = uuid.UUID("5d0a7c52-6f1e-4b0f-9a57-3f1d6c0b8e21")


@dataclass(frozen=True)
class UsageRow:
    """One minute of one session's usage on one container: a `training_api_token_metrics` row.

    Fields shared with shared endpoints' `EndpointTokenMetricRow` keep its names
    and meaning: `model_id` is the billable base model, `prompt_tokens` counts
    all prefill tokens including `cached_prompt_tokens`, and `request_uuid` is
    the row's identity for deduplication.
    """

    workspace_id: str
    environment_id: str
    timestamp: float
    model_id: str
    request_uuid: str
    deployment: str
    app_name: str
    app_kind: str
    container_id: str
    session_id: str | None = None
    training_model_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    train_tokens: int = 0
    forward_tokens: int = 0
    busy_seconds: float = 0.0
    idle_seconds: float = 0.0
    cold_start_seconds: float = 0.0
    org_id: str = ""


COUNTERS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "reasoning_tokens",
    "train_tokens",
    "forward_tokens",
    "busy_seconds",
    "idle_seconds",
    "cold_start_seconds",
)


@dataclass(frozen=True)
class HourlyUsage:
    """One session's tokens on one base model in one UTC hour."""

    bucket_start: float
    model_id: str
    session_id: str | None
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    train_tokens: int = 0
    forward_tokens: int = 0


@dataclass(frozen=True)
class SessionUsage:
    """Who a session's usage is billed to.

    `user_id` comes from the API key the session was created with, never from
    the client. `labels` are the allowlisted `user_metadata` values, stored as
    given; use IDs such as `team_id` and map them to names on your side.
    """

    user_id: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def group(self, key: str) -> str | None:
        return self.user_id if key == "user" else self.labels.get(key)


class UsageWriter(Protocol):
    """Where meters and the control plane send usage: a Volume now, ClickHouse later."""

    def write(self, rows: Iterable[UsageRow]) -> None: ...

    def put_session(
        self, session_id: str, session: SessionUsage, created_at: float
    ) -> None: ...

    def put_model(self, training_model_id: str, session_id: str) -> None: ...


class UsageReader(Protocol):
    """What `GET /api/v1/billing/usage/events` reads, whichever store holds the rows."""

    def hourly_usage(self, start: float, end: float) -> list[HourlyUsage]: ...

    def sessions(self, session_ids: Iterable[str]) -> dict[str, SessionUsage]: ...


def labels_from(
    user_metadata: Mapping[str, object] | None,
    keys: Iterable[str] = DEFAULT_LABEL_KEYS,
) -> dict[str, str]:
    if not user_metadata:
        return {}
    return {
        key: value
        for key in keys
        if isinstance(value := user_metadata.get(key), str) and 0 < len(value) <= 256
    }


def label_keys_from_env() -> tuple[str, ...]:
    value = os.environ.get("SPINDLE_USAGE_LABEL_KEYS")
    if value is None:
        return DEFAULT_LABEL_KEYS
    return tuple(key.strip() for key in value.split(",") if key.strip())


class DirectorySink:
    """Store usage under one directory, such as a mounted Modal Volume.

    Each write is a new file, so containers never write to the same path, and
    readers drop rows they have already seen by `request_uuid`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        commit: Callable[[], None] | None = None,
        reload: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self._commit = commit
        self._reload = reload

    def write(self, rows: Iterable[UsageRow]) -> None:
        by_hour: dict[str, list[UsageRow]] = defaultdict(list)
        for row in rows:
            by_hour[_hour_dir(row.timestamp)].append(row)
        if not by_hour:
            return
        for hour, hour_rows in by_hour.items():
            self._write_file(
                self.root / "rows" / hour / f"{uuid.uuid4().hex}.jsonl",
                "".join(json.dumps(asdict(row)) + "\n" for row in hour_rows),
            )
        self._committed()

    def put_session(
        self, session_id: str, session: SessionUsage, created_at: float
    ) -> None:
        self._write_file(
            self.root / "sessions" / f"{session_id}.json",
            json.dumps(
                {
                    "user_id": session.user_id,
                    "labels": dict(session.labels),
                    "created_at": created_at,
                }
            ),
        )
        self._committed()

    def put_model(self, training_model_id: str, session_id: str) -> None:
        self._write_file(
            self.root / "models" / f"{training_model_id}.json",
            json.dumps({"session_id": session_id}),
        )
        self._committed()

    def read(self, start: float, end: float) -> list[UsageRow]:
        """Raw rows with training rows resolved to their session."""
        if self._reload is not None:
            self._reload()
        rows: dict[str, UsageRow] = {}
        hour = math.floor(start / 3600) * 3600
        while hour < end:
            directory = self.root / "rows" / _hour_dir(hour)
            for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else ():
                for line in path.read_text().splitlines():
                    row = UsageRow(**json.loads(line))
                    if start <= row.timestamp < end:
                        rows.setdefault(row.request_uuid, row)
            hour += 3600
        owners: dict[str, str | None] = {}
        resolved = []
        for row in rows.values():
            if row.session_id is None and row.training_model_id is not None:
                if row.training_model_id not in owners:
                    owners[row.training_model_id] = self._owner(row.training_model_id)
                row = replace(row, session_id=owners[row.training_model_id])
            resolved.append(row)
        return resolved

    def hourly_usage(self, start: float, end: float) -> list[HourlyUsage]:
        totals: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
        for row in self.read(start, end):
            hour = math.floor(row.timestamp / 3600) * 3600
            entry = totals[(hour, row.model_id, row.session_id)]
            entry[0] += row.prompt_tokens
            entry[1] += row.cached_prompt_tokens
            entry[2] += row.completion_tokens
            entry[3] += row.train_tokens
            entry[4] += row.forward_tokens
        return [
            HourlyUsage(hour, model_id, session_id, *counts)
            for (hour, model_id, session_id), counts in totals.items()
            if any(counts)
        ]

    def sessions(self, session_ids: Iterable[str]) -> dict[str, SessionUsage]:
        sessions = {}
        for session_id in session_ids:
            path = self.root / "sessions" / f"{session_id}.json"
            if path.is_file():
                record = json.loads(path.read_text())
                sessions[session_id] = SessionUsage(record["user_id"], record["labels"])
        return sessions

    def _owner(self, training_model_id: str) -> str | None:
        path = self.root / "models" / f"{training_model_id}.json"
        return json.loads(path.read_text())["session_id"] if path.is_file() else None

    def _write_file(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        partial.write_text(text)
        partial.replace(path)

    def _committed(self) -> None:
        if self._commit is not None:
            self._commit()


class UsageMeter:
    """Sum usage per minute and session in memory; write closed minutes to a writer."""

    def __init__(
        self,
        writer: UsageWriter,
        *,
        deployment: str,
        container_id: str | None = None,
        workspace_id: str | None = None,
        environment_id: str | None = None,
        org_id: str = "",
        clock: Callable[[], float] = time.time,
        window_s: int = WINDOW_S,
    ) -> None:
        self.writer = writer
        self.deployment = deployment
        self.container_id = (
            container_id or os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
        )
        self.workspace_id = workspace_id or os.environ.get("MODAL_WORKSPACE_ID", "")
        self.environment_id = environment_id or os.environ.get("MODAL_ENVIRONMENT", "")
        self.org_id = org_id
        self.clock = clock
        self.window_s = window_s
        self._counts: dict[tuple, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(
        self,
        *,
        app_name: str,
        app_kind: str,
        model_id: str,
        session_id: str | None = None,
        training_model_id: str | None = None,
        **counters: float,
    ) -> None:
        unknown = counters.keys() - set(COUNTERS)
        if unknown:
            raise ValueError(f"unknown usage counters {sorted(unknown)}")
        if not any(value > 0 for value in counters.values()):
            return
        window = math.floor(self.clock() / self.window_s) * self.window_s
        key = (window, app_name, app_kind, model_id, session_id, training_model_id)
        with self._lock:
            _merge(self._counts.setdefault(key, {}), counters)

    def flush(self, *, include_open: bool = False) -> int:
        """Write every closed minute; on shutdown, the open minute too."""
        cutoff = math.inf if include_open else self.clock() - self.window_s
        with self._lock:
            ready = {key: v for key, v in self._counts.items() if key[0] <= cutoff}
            for key in ready:
                del self._counts[key]
        if not ready:
            return 0
        rows = [self._row(key, counters) for key, counters in ready.items()]
        try:
            self.writer.write(rows)
        except Exception:
            with self._lock:
                for key, counters in ready.items():
                    _merge(self._counts.setdefault(key, {}), counters)
            raise
        return len(rows)

    def start(self, interval_s: float = WINDOW_S) -> UsageMeter:
        """Flush from a background thread, and once more when the process exits."""

        def loop() -> None:
            while not self._stop.wait(interval_s):
                try:
                    self.flush()
                except Exception:  # noqa: BLE001 - usage must not stop training
                    log.warning("Usage flush failed; retrying next interval")

        self._thread = threading.Thread(target=loop, name="usage-flush", daemon=True)
        self._thread.start()
        atexit.register(self.close)
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        try:
            self.flush(include_open=True)
        except Exception:  # noqa: BLE001 - usage must not stop shutdown
            log.warning("Final usage flush failed")

    def _row(self, key: tuple, counters: Mapping[str, float]) -> UsageRow:
        window, app_name, app_kind, model_id, session_id, training_model_id = key
        identity = json.dumps([self.container_id, *key])
        return UsageRow(
            workspace_id=self.workspace_id,
            environment_id=self.environment_id,
            timestamp=window,
            model_id=model_id,
            request_uuid=str(uuid.uuid5(_ROW_NAMESPACE, identity)),
            deployment=self.deployment,
            app_name=app_name,
            app_kind=app_kind,
            container_id=self.container_id,
            session_id=session_id,
            training_model_id=training_model_id,
            org_id=self.org_id,
            **{name: _typed(name, value) for name, value in counters.items()},
        )


class TrainerUsage:
    """What one trainer container meters: trained tokens, busy and idle time, cold start."""

    def __init__(
        self,
        meter: UsageMeter,
        *,
        app_name: str,
        base_model: str,
        started_at: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.meter = meter
        self.app_name = app_name
        self.base_model = base_model
        self._cold_start_s = max(0.0, clock() - started_at)

    def trained(self, model_id: str, payload: object) -> None:
        tokens = workload(payload).get("spindle.input_tokens")
        if tokens:
            self._add(training_model_id=model_id, train_tokens=tokens)

    def forwarded(self, model_id: str, payload: object) -> None:
        tokens = workload(payload).get("spindle.input_tokens")
        if tokens:
            self._add(training_model_id=model_id, forward_tokens=tokens)

    def accepted(self, model_id: str) -> None:
        """Charge the container's start-up to the first model it accepted."""
        if self._cold_start_s:
            self._add(training_model_id=model_id, cold_start_seconds=self._cold_start_s)
            self._cold_start_s = 0.0

    def busy(self, seconds: float) -> None:
        self._add(busy_seconds=seconds)

    def idle(self, seconds: float) -> None:
        self._add(idle_seconds=seconds)

    def _add(self, **values) -> None:
        self.meter.add(
            app_name=self.app_name,
            app_kind="trainer",
            model_id=self.base_model,
            **values,
        )


def meter_sample(
    meter: UsageMeter,
    *,
    app_name: str,
    base_model: str,
    session_id: str,
    stats: Mapping[str, object],
) -> None:
    """Record one finished sample request from the stats `sample_task` fills in."""
    prefill = int(stats.get("prefill_tokens") or 0)
    meter.add(
        app_name=app_name,
        app_kind="sampler",
        model_id=base_model,
        session_id=session_id,
        prompt_tokens=prefill,
        cached_prompt_tokens=min(int(stats.get("cached_prompt_tokens") or 0), prefill),
        completion_tokens=int(stats.get("generated_tokens") or 0),
    )


def billing_usage(
    reader: UsageReader, starting_on: str | datetime, ending_before: str | datetime
) -> dict[str, object]:
    """Hourly token usage per session, shaped like Tinker's `BillingUsageResponse`."""
    start, end = usage_window(starting_on, ending_before)
    data = []
    for usage in reader.hourly_usage(start.timestamp(), end.timestamp()):
        for info in (
            # Tinker has no forward event, so forward-only tokens count as training.
            {
                "type": "training",
                "token_count": usage.train_tokens + usage.forward_tokens,
            },
            {
                "type": "sampling_prefill",
                "cached": False,
                "token_count": usage.prompt_tokens - usage.cached_prompt_tokens,
            },
            {
                "type": "sampling_prefill",
                "cached": True,
                "token_count": usage.cached_prompt_tokens,
            },
            {"type": "sampling_sample", "token_count": usage.completion_tokens},
        ):
            if info["token_count"] > 0:
                data.append(
                    {
                        "bucket_start": _iso(usage.bucket_start),
                        "bucket_end": _iso(usage.bucket_start + 3600),
                        "base_model": usage.model_id,
                        "session_id": usage.session_id,
                        "event_info": info,
                    }
                )
    data.sort(
        key=lambda event: (event["bucket_start"], -event["event_info"]["token_count"])
    )
    session_ids = dict.fromkeys(
        event["session_id"] for event in data if event["session_id"]
    )
    sessions = reader.sessions(session_ids)
    for event in data:
        user_id = sessions.get(event["session_id"], SessionUsage()).user_id
        event["user_id"] = event["user_name"] = user_id
    return {
        "data": data,
        "sessions": {
            session_id: {
                "user_metadata": dict(sessions.get(session_id, SessionUsage()).labels)
                or None
            }
            for session_id in session_ids
        },
    }


@dataclass(frozen=True)
class Rates:
    """Token prices in USD per million tokens, used only to weigh sessions against each other."""

    train: float
    prefill: float
    cached_prefill: float
    sample: float
    forward: float | None = None

    @property
    def forward_rate(self) -> float:
        """Forward-only passes default to the prefill rate."""
        return self.prefill if self.forward is None else self.forward

    def price(self, row: UsageRow) -> float:
        uncached = row.prompt_tokens - row.cached_prompt_tokens
        return (
            self.train * row.train_tokens
            + self.forward_rate * row.forward_tokens
            + self.prefill * uncached
            + self.cached_prefill * row.cached_prompt_tokens
            + self.sample * row.completion_tokens
        ) / 1e6


def attribute_cost(
    rows: Iterable[UsageRow],
    app_costs: Mapping[str, float],
    rates: Rates,
    *,
    sessions: Mapping[str, SessionUsage],
    group_by: str = "run_id",
) -> dict[str, float]:
    """Split each app's Modal cost for the window across groups by token price.

    `group_by` is `user` or any label key.

    Rows must already carry their `session_id` (`DirectorySink.read` resolves
    training rows). Idle and unowned cold start time are reported as their own
    groups, and an app with cost but no usage rows is `unattributed`, so groups
    sum to the bill.
    """
    by_app: dict[str, list[UsageRow]] = defaultdict(list)
    for row in rows:
        by_app[row.app_name].append(row)

    def group(session_id: str | None) -> str:
        session = sessions.get(session_id or "", SessionUsage())
        return session.group(group_by) or "unknown"

    costs: dict[str, float] = defaultdict(float)
    for app_name, cost in app_costs.items():
        app_rows = by_app.get(app_name, [])
        container_s = sum(
            row.busy_seconds + row.idle_seconds + row.cold_start_seconds
            for row in app_rows
        )
        per_second = cost / container_s if container_s else 0.0
        remaining = cost
        for row in app_rows:
            if row.cold_start_seconds:
                cold = per_second * row.cold_start_seconds
                costs[group(row.session_id) if row.session_id else "cold_start"] += cold
                remaining -= cold
        if idle := per_second * sum(row.idle_seconds for row in app_rows):
            costs["idle"] += idle
            remaining -= idle

        prices: dict[str | None, float] = defaultdict(float)
        for row in app_rows:
            if price := rates.price(row):
                prices[row.session_id] += price
        total = sum(prices.values())
        if not total:
            costs["idle" if app_rows else "unattributed"] += remaining
            continue
        for session_id, price in prices.items():
            costs[group(session_id)] += remaining * price / total
    return dict(costs)


def usage_window(starting_on: str | datetime, ending_before: str | datetime):
    start, end = _datetime(starting_on), _datetime(ending_before)
    for value in (start, end):
        if value.minute or value.second or value.microsecond:
            raise ValueError("billing usage windows must align to a UTC hour")
    if end <= start:
        raise ValueError("ending_before must be after starting_on")
    if end - start > MAX_QUERY_WINDOW:
        raise ValueError("billing usage windows are at most 14 days")
    return start, end


_INTEGER_COUNTERS = {
    item.name
    for item in fields(UsageRow)
    if item.name in COUNTERS and item.type == "int"
}


def _typed(name: str, value: float) -> float | int:
    return int(value) if name in _INTEGER_COUNTERS else float(value)


def _merge(totals: dict[str, float], counters: Mapping[str, float]) -> None:
    for name, value in counters.items():
        totals[name] = totals.get(name, 0) + value


def _datetime(value: str | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def _hour_dir(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H")
