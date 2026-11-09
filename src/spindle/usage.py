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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from spindle.telemetry.trainer import workload

log = logging.getLogger(__name__)

TOKEN_EVENTS = ("training", "forward", "sampling_prefill", "sampling_sample")
TIME_EVENTS = ("busy", "idle", "cold_start")
EVENT_TYPES = TOKEN_EVENTS + TIME_EVENTS
DEFAULT_LABEL_KEYS = ("run_id", "attempt_id")
WINDOW_S = 60
MAX_QUERY_WINDOW = timedelta(days=14)
_ROW_NAMESPACE = uuid.UUID("5d0a7c52-6f1e-4b0f-9a57-3f1d6c0b8e21")


@dataclass(frozen=True)
class UsageRow:
    """One minute of one session's usage on one container: a `training_api_token_metrics` row."""

    deployment: str
    app_name: str
    app_kind: str
    container_id: str
    base_model: str
    event_type: str
    timestamp: float
    event_uuid: str
    session_id: str | None = None
    model_id: str | None = None
    cached: bool = False
    token_count: int = 0
    seconds: float = 0.0
    org_id: str = ""


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
    readers drop rows they have already seen by `event_uuid`.
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

    def put_model(self, model_id: str, session_id: str) -> None:
        self._write_file(
            self.root / "models" / f"{model_id}.json",
            json.dumps({"session_id": session_id}),
        )
        self._committed()

    def read(self, start: float, end: float) -> list[UsageRow]:
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
                        rows.setdefault(row.event_uuid, row)
            hour += 3600
        return list(rows.values())

    def sessions(self) -> dict[str, SessionUsage]:
        sessions = {}
        for path in _json_files(self.root / "sessions"):
            record = json.loads(path.read_text())
            sessions[path.stem] = SessionUsage(record["user_id"], record["labels"])
        return sessions

    def models(self) -> dict[str, str]:
        return {
            path.stem: json.loads(path.read_text())["session_id"]
            for path in _json_files(self.root / "models")
        }

    def _write_file(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        partial.write_text(text)
        partial.replace(path)

    def _committed(self) -> None:
        if self._commit is not None:
            self._commit()


class UsageMeter:
    """Sum usage per minute and session in memory; write closed minutes to a sink."""

    def __init__(
        self,
        sink: DirectorySink,
        *,
        deployment: str,
        container_id: str | None = None,
        org_id: str = "",
        clock: Callable[[], float] = time.time,
        window_s: int = WINDOW_S,
    ) -> None:
        self.sink = sink
        self.deployment = deployment
        self.container_id = (
            container_id or os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
        )
        self.org_id = org_id
        self.clock = clock
        self.window_s = window_s
        self._counts: dict[tuple, list] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(
        self,
        event_type: str,
        *,
        app_name: str,
        app_kind: str,
        base_model: str,
        session_id: str | None = None,
        model_id: str | None = None,
        cached: bool = False,
        tokens: int = 0,
        seconds: float = 0.0,
    ) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown usage event {event_type!r}")
        if tokens <= 0 and seconds <= 0:
            return
        window = math.floor(self.clock() / self.window_s) * self.window_s
        key = (
            window,
            app_name,
            app_kind,
            base_model,
            session_id,
            model_id,
            event_type,
            cached,
        )
        with self._lock:
            totals = self._counts.setdefault(key, [0, 0.0])
            totals[0] += tokens
            totals[1] += seconds

    def flush(self, *, include_open: bool = False) -> int:
        """Write every closed minute; on shutdown, the open minute too."""
        cutoff = math.inf if include_open else self.clock() - self.window_s
        with self._lock:
            ready = {key: v for key, v in self._counts.items() if key[0] <= cutoff}
            for key in ready:
                del self._counts[key]
        if not ready:
            return 0
        rows = [self._row(key, *totals) for key, totals in ready.items()]
        try:
            self.sink.write(rows)
        except Exception:
            with self._lock:
                for key, (tokens, seconds) in ready.items():
                    totals = self._counts.setdefault(key, [0, 0.0])
                    totals[0] += tokens
                    totals[1] += seconds
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

    def _row(self, key: tuple, tokens: int, seconds: float) -> UsageRow:
        window, app_name, app_kind, base_model, session_id, model_id, event, cached = (
            key
        )
        identity = json.dumps([self.container_id, *key])
        return UsageRow(
            deployment=self.deployment,
            app_name=app_name,
            app_kind=app_kind,
            container_id=self.container_id,
            base_model=base_model,
            event_type=event,
            timestamp=window,
            event_uuid=str(uuid.uuid5(_ROW_NAMESPACE, identity)),
            session_id=session_id,
            model_id=model_id,
            cached=cached,
            token_count=tokens,
            seconds=seconds,
            org_id=self.org_id,
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
            self._add("training", model_id=model_id, tokens=tokens)

    def forwarded(self, model_id: str, payload: object) -> None:
        tokens = workload(payload).get("spindle.input_tokens")
        if tokens:
            self._add("forward", model_id=model_id, tokens=tokens)

    def accepted(self, model_id: str) -> None:
        """Charge the container's start-up to the first model it accepted."""
        if self._cold_start_s:
            self._add("cold_start", model_id=model_id, seconds=self._cold_start_s)
            self._cold_start_s = 0.0

    def busy(self, seconds: float) -> None:
        self._add("busy", seconds=seconds)

    def idle(self, seconds: float) -> None:
        self._add("idle", seconds=seconds)

    def _add(self, event_type: str, **values) -> None:
        self.meter.add(
            event_type,
            app_name=self.app_name,
            app_kind="trainer",
            base_model=self.base_model,
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
    cached = min(int(stats.get("cached_prompt_tokens") or 0), prefill)
    for event_type, is_cached, tokens in (
        ("sampling_prefill", False, prefill - cached),
        ("sampling_prefill", True, cached),
        ("sampling_sample", False, int(stats.get("generated_tokens") or 0)),
    ):
        meter.add(
            event_type,
            app_name=app_name,
            app_kind="sampler",
            base_model=base_model,
            session_id=session_id,
            cached=is_cached,
            tokens=tokens,
        )


def billing_usage(
    sink: DirectorySink, starting_on: str | datetime, ending_before: str | datetime
) -> dict[str, object]:
    """Hourly token usage per session, shaped like Tinker's `BillingUsageResponse`."""
    start, end = usage_window(starting_on, ending_before)
    rows = sink.read(start.timestamp(), end.timestamp())
    session_of = sink.models()
    buckets: dict[tuple, int] = defaultdict(int)
    for row in rows:
        if row.event_type not in TOKEN_EVENTS:
            continue
        session_id = row.session_id or session_of.get(row.model_id or "")
        hour = math.floor(row.timestamp / 3600) * 3600
        # Tinker has no forward event, so forward-only tokens count as training.
        event_type = "training" if row.event_type == "forward" else row.event_type
        buckets[(hour, row.base_model, session_id, event_type, row.cached)] += (
            row.token_count
        )
    sessions = sink.sessions()
    data = []
    for (hour, base_model, session_id, event_type, cached), tokens in sorted(
        buckets.items(), key=lambda item: (item[0][0], -item[1])
    ):
        info: dict[str, object] = {"type": event_type, "token_count": tokens}
        if event_type == "sampling_prefill":
            info["cached"] = cached
        user_id = sessions.get(session_id, SessionUsage()).user_id
        data.append(
            {
                "bucket_start": _iso(hour),
                "bucket_end": _iso(hour + 3600),
                "base_model": base_model,
                "user_id": user_id,
                "user_name": user_id,
                "session_id": session_id,
                "event_info": info,
            }
        )
    return {
        "data": data,
        "sessions": {
            session_id: {
                "user_metadata": dict(sessions.get(session_id, SessionUsage()).labels)
                or None
            }
            for session_id in dict.fromkeys(
                event["session_id"] for event in data if event["session_id"]
            )
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

    def price(self, row: UsageRow) -> float:
        rate = {
            "training": self.train,
            # Forward-only passes default to the prefill rate.
            "forward": self.prefill if self.forward is None else self.forward,
            "sampling_prefill": self.cached_prefill if row.cached else self.prefill,
            "sampling_sample": self.sample,
        }[row.event_type]
        return rate * row.token_count / 1e6


def attribute_cost(
    rows: Iterable[UsageRow],
    app_costs: Mapping[str, float],
    rates: Rates,
    *,
    sessions: Mapping[str, SessionUsage],
    models: Mapping[str, str],
    group_by: str = "run_id",
) -> dict[str, float]:
    """Split each app's Modal cost for the window across groups by token price.

    `group_by` is `user` or any label key.

    Idle and unowned cold start time are reported as their own groups, and an
    app with cost but no usage rows is `unattributed`, so groups sum to the bill.
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
        seconds = defaultdict(float)
        for row in app_rows:
            if row.event_type in TIME_EVENTS:
                seconds[row.event_type] += row.seconds
        container_s = sum(seconds.values())
        per_second = cost / container_s if container_s else 0.0
        remaining = cost
        for row in app_rows:
            if row.event_type == "cold_start":
                owner = row.session_id or models.get(row.model_id or "")
                costs[group(owner) if owner else "cold_start"] += (
                    per_second * row.seconds
                )
                remaining -= per_second * row.seconds
        if idle := per_second * seconds["idle"]:
            costs["idle"] += idle
            remaining -= idle

        prices: dict[str | None, float] = defaultdict(float)
        for row in app_rows:
            if row.event_type in TOKEN_EVENTS:
                prices[row.session_id or models.get(row.model_id or "")] += rates.price(
                    row
                )
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


def _json_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json")) if directory.is_dir() else []
