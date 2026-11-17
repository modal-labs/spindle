"""Storing usage as write-once files under one directory, such as a mounted Modal Volume.

rows/<hour>/<id>.jsonl   one file per container flush: per-minute rows
late/<hour>/<id>.jsonl   rows written after their hour's raw files were frozen
hours/<hour>.jsonl       an hour's raw rows summed per session and app
days/<day>.jsonl         a day's hour rollups in one file
sessions/<id>.json       a session's labels
models/<id>.json         the session that created a training model
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path

from .rows import (
    COUNTERS,
    DAY_S,
    HOUR_S,
    ROW_NAMESPACE,
    UsageRow,
    add_counters,
    hour_name,
    hour_start,
    parse_hour_name,
    read_rows,
    to_jsonl,
)

log = logging.getLogger(__name__)

# An hour's rows/ is frozen ROLLUP_DELAY_S after the hour ends, and writers switch to
# late/ LATE_AFTER_S after it ends. The gap is how long a write may stall, or a
# container's clock lag, before its rows would be missed.
LATE_AFTER_S = 10 * 60
ROLLUP_DELAY_S = 60 * 60
PRUNE_AFTER_S = 14 * DAY_S
READ_THREADS = 16


class DirectorySink:
    def __init__(
        self,
        root: str | Path,
        *,
        commit: Callable[[], None] | None = None,
        reload: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.clock = clock
        self._commit = commit
        self._reload = reload
        # A Modal Volume can't reload while this process has files open.
        self._lock = threading.Lock()

    def write(self, rows: Iterable[UsageRow]) -> None:
        by_hour: dict[float, list[UsageRow]] = defaultdict(list)
        for row in rows:
            by_hour[hour_start(row.timestamp)].append(row)
        if not by_hour:
            return
        late_before = self.clock() - HOUR_S - LATE_AFTER_S
        with self._lock:
            paths = []
            for hour, hour_rows in by_hour.items():
                kind = "late" if hour < late_before else "rows"
                paths.append(
                    self.root / kind / hour_name(hour) / f"{uuid.uuid4().hex}.jsonl"
                )
                self._write_file(paths[-1], to_jsonl(hour_rows))
            self._commit_or_remove(paths)

    def put_session(self, session_id: str, labels: Mapping[str, str]) -> None:
        if not labels:
            return
        path = self.root / "sessions" / f"{session_id}.json"
        with self._lock:
            if self._write_file(path, json.dumps({"labels": dict(labels)})):
                self._commit_or_remove([path])

    def put_model(self, training_model_id: str, session_id: str) -> None:
        path = self.root / "models" / f"{training_model_id}.json"
        with self._lock:
            if self._write_file(path, json.dumps({"session_id": session_id})):
                self._commit_or_remove([path])

    def read(self, start: float, end: float) -> list[UsageRow]:
        """Rows in [start, end), with training rows given their model's session."""
        if start % HOUR_S or end % HOUR_S:
            raise ValueError("usage reads must start and end on an hour")
        with self._lock:
            if self._reload is not None:
                self._reload()
            rows = self._rows_between(start, end)
            owners = self._model_owners(rows)
        return [
            replace(row, session_id=owners.get(row.training_model_id))
            if row.session_id is None and row.training_model_id is not None
            else row
            for row in rows
        ]

    def sessions(self, session_ids: Iterable[str]) -> dict[str, dict[str, str]]:
        with self._lock:
            records = self._read_records("sessions", session_ids)
        return {session_id: record["labels"] for session_id, record in records.items()}

    def compact(self, now: float | None = None) -> int:
        """Roll up closed hours and days, and prune old raw files; returns files written."""
        with self._lock:
            if self._reload is not None:
                self._reload()
            cutoff = (self.clock() if now is None else now) - ROLLUP_DELAY_S
            by_day = self._closed_hours_by_day(cutoff)
            written = sum(
                self._roll_up_day(day, hours, cutoff)
                for day, hours in sorted(by_day.items())
            )
            self._prune_before(by_day, cutoff - PRUNE_AFTER_S)
            return written

    def _closed_hours_by_day(self, cutoff: float) -> dict[float, list[float]]:
        raw = self.root / "rows"
        by_day: dict[float, list[float]] = defaultdict(list)
        for directory in raw.iterdir() if raw.is_dir() else ():
            hour = parse_hour_name(directory.name)
            if hour is not None and hour + HOUR_S <= cutoff:
                by_day[math.floor(hour / DAY_S) * DAY_S].append(hour)
        return by_day

    def _roll_up_day(self, day: float, hours: list[float], cutoff: float) -> int:
        day_path = self._day_path(day)
        if day_path.exists():
            return 0
        written = 0
        day_rows = []
        for hour in sorted(hours):
            hour_path = self._hour_path(hour)
            hour_rows = read_rows(hour_path)
            if hour_rows is None:
                hour_rows = _rollup(self._raw_rows(hour))
                self._write_file(hour_path, to_jsonl(hour_rows))
                written += 1
            day_rows += hour_rows
        if day + DAY_S <= cutoff:
            self._write_file(day_path, to_jsonl(day_rows))
            written += 1
        if written:
            # Commit each day, so a run cut short keeps what it rolled up.
            self._committed()
        return written

    def _prune_before(self, by_day: dict[float, list[float]], before: float) -> None:
        pruned = [
            hour
            for day, hours in by_day.items()
            if self._day_path(day).exists()
            for hour in hours
            if hour + HOUR_S <= before
        ]
        for hour in pruned:
            self._prune(hour)
        if pruned:
            self._committed()

    def _prune(self, hour: float) -> None:
        """Delete an hour's raw files, keeping only late rows its rollup doesn't count."""
        counted = {row.request_uuid for row in self._raw_rows(hour)}
        late = self.root / "late" / hour_name(hour)
        for path in late.glob("*.jsonl") if late.is_dir() else ():
            rows = read_rows(path) or []
            uncounted = [row for row in rows if row.request_uuid not in counted]
            if len(uncounted) < len(rows):
                if uncounted:
                    self._write_file(
                        path.with_name(f"kept-{path.name}"), to_jsonl(uncounted)
                    )
                path.unlink()
        shutil.rmtree(self.root / "rows" / hour_name(hour))

    def _rows_between(self, start: float, end: float) -> list[UsageRow]:
        rows: dict[str, UsageRow] = {}
        # Day rollups read so far, as rows by hour, or None when that day has none yet.
        days: dict[float, dict[float, list[UsageRow]] | None] = {}
        hour = hour_start(start)
        while hour < end:
            for row in [*self._hour_rows(hour, days), *self._late_rows(hour)]:
                if start <= row.timestamp < end:
                    rows.setdefault(row.request_uuid, row)
            hour += HOUR_S
        return list(rows.values())

    def _hour_rows(
        self, hour: float, days: dict[float, dict[float, list[UsageRow]] | None]
    ) -> list[UsageRow]:
        day = math.floor(hour / DAY_S) * DAY_S
        if day not in days:
            day_rows = read_rows(self._day_path(day))
            days[day] = None
            if day_rows is not None:
                days[day] = defaultdict(list)
                for row in day_rows:
                    days[day][row.timestamp].append(row)
        if days[day] is not None:
            return days[day].get(hour, [])
        hour_rows = read_rows(self._hour_path(hour))
        return self._raw_rows(hour) if hour_rows is None else hour_rows

    def _late_rows(self, hour: float) -> list[UsageRow]:
        late = self._raw_rows(hour, "late")
        if not late:
            return []
        # A retry can land here after its first write was rolled up.
        counted = {row.request_uuid for row in self._raw_rows(hour)}
        return [row for row in late if row.request_uuid not in counted]

    def _model_owners(self, rows: list[UsageRow]) -> dict[str | None, str]:
        model_ids = {
            row.training_model_id
            for row in rows
            if row.session_id is None and row.training_model_id is not None
        }
        records = self._read_records("models", model_ids)
        return {model_id: record["session_id"] for model_id, record in records.items()}

    def _raw_rows(self, hour: float, kind: str = "rows") -> list[UsageRow]:
        directory = self.root / kind / hour_name(hour)
        if not directory.is_dir():
            return []
        paths = sorted(directory.glob("*.jsonl"))
        # Each open on a Volume waits on the network, so read files concurrently.
        with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
            return [row for rows in pool.map(read_rows, paths) for row in rows or ()]

    def _read_records(self, kind: str, ids: Iterable[str]) -> dict[str, dict]:
        paths = {name: self.root / kind / f"{name}.json" for name in ids}
        with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
            records = dict(zip(paths, pool.map(_read_json, paths.values())))
        return {name: record for name, record in records.items() if record is not None}

    def _hour_path(self, hour: float) -> Path:
        return self.root / "hours" / f"{hour_name(hour)}.jsonl"

    def _day_path(self, day: float) -> Path:
        return self.root / "days" / f"{datetime.fromtimestamp(day, UTC):%Y-%m-%d}.jsonl"

    def _write_file(self, path: Path, text: str) -> bool:
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        partial.write_text(text)
        partial.replace(path)
        return True

    def _committed(self) -> None:
        if self._commit is not None:
            self._commit()

    def _commit_or_remove(self, paths: list[Path]) -> None:
        try:
            self._committed()
        except Exception:
            # The caller retries, and a retry that found these files would skip the commit.
            for path in paths:
                path.unlink(missing_ok=True)
            raise


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except ValueError:
        log.warning("Skipping unreadable %s", path)
        return None


_ROLLUP_KEY = tuple(
    item.name
    for item in fields(UsageRow)
    if item.name not in COUNTERS and item.name != "request_uuid"
)


def _rollup(rows: Iterable[UsageRow]) -> list[UsageRow]:
    """Sum rows per hour, app, model and session, counting each row id once."""
    unique = {row.request_uuid: row for row in rows}
    totals: dict[tuple, dict[str, float]] = {}
    for row in unique.values():
        row = replace(row, timestamp=hour_start(row.timestamp))
        key = tuple(getattr(row, name) for name in _ROLLUP_KEY)
        add_counters(
            totals.setdefault(key, {}), {name: getattr(row, name) for name in COUNTERS}
        )
    return [
        UsageRow(
            **dict(zip(_ROLLUP_KEY, key)),
            request_uuid=str(uuid.uuid5(ROW_NAMESPACE, json.dumps(["rollup", *key]))),
            **counters,
        )
        for key, counters in totals.items()
    ]
