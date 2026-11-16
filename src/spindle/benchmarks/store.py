from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .schema import numeric_leaves


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Store:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "benchmarks.sqlite3"
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, manifest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, time REAL NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS records_run ON records(run_id, time);
                CREATE TABLE IF NOT EXISTS metrics (
                    record_id TEXT NOT NULL, run_id TEXT NOT NULL, time REAL NOT NULL,
                    name TEXT NOT NULL, value REAL NOT NULL, labels TEXT NOT NULL,
                    unit TEXT NOT NULL, kind TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS metrics_lookup ON metrics(run_id, name, time);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def register(self, manifest):
        with self.connection() as db:
            old = db.execute(
                "SELECT manifest FROM runs WHERE id=?", (manifest["run_id"],)
            ).fetchone()
            merged = {**(json.loads(old[0]) if old else {}), **manifest}
            db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?)",
                (manifest["run_id"], json.dumps(merged, allow_nan=False)),
            )

    def append(self, run_id, record, *, connection=None):
        record = json_safe(record)
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        identity = hashlib.sha256((run_id + payload).encode()).hexdigest()
        stamp = record.get("time", time.time())
        kind = record.get("kind", "event")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("record time must be finite")
        labels = record.get("labels", {})
        if not isinstance(labels, dict):
            raise ValueError("record labels must be an object")
        values = record.get("metrics", {})
        rows = [
            (
                identity,
                run_id,
                stamp,
                name,
                value,
                json.dumps(labels, sort_keys=True),
                record.get("unit", ""),
                kind,
            )
            for name, value in numeric_leaves(values)
        ]
        with self.connection() if connection is None else nullcontext(connection) as db:
            inserted = db.execute(
                "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)",
                (identity, run_id, stamp, kind, payload),
            ).rowcount
            if inserted:
                db.executemany("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?)", rows)

    def append_many(self, records):
        with self.connection() as db:
            for run_id, record in records:
                self.append(run_id, record, connection=db)

    def runs(self):
        with self.connection() as db:
            return [
                json.loads(r[0])
                for r in db.execute("SELECT manifest FROM runs ORDER BY rowid DESC")
            ]

    def catalog(self, run_ids):
        if not run_ids:
            return []
        placeholders = ",".join("?" for _ in run_ids)
        with self.connection() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    f"SELECT name, unit, kind, count(*) AS samples FROM metrics WHERE run_id IN ({placeholders}) GROUP BY name,unit,kind ORDER BY name",
                    run_ids,
                )
            ]

        derived = self.weight_sync_series(run_ids)
        if derived:
            rows.append(
                {
                    "name": "weight_sync.capture_to_ready_s",
                    "unit": "s",
                    "kind": "derived",
                    "samples": len(derived),
                }
            )
        return sorted(rows, key=lambda row: row["name"])

    def weight_sync_series(self, run_ids):
        starts = {}
        for row in self.series(run_ids, "weight_sync.capture_started_unix_s"):
            labels = row["labels"]
            key = (
                row["run_id"],
                labels.get("spindle.model_id"),
                labels.get("publish_version"),
            )
            starts[key] = min(starts.get(key, float("inf")), row["value"])
        ready = {}
        for row in self.series(run_ids, "weight_sync.ready_unix_s"):
            labels = row["labels"]
            publication = (
                row["run_id"],
                labels.get("spindle.model_id"),
                labels.get("publish_version"),
            )
            if publication[1] is None or publication[2] is None:
                continue
            start = starts.get(publication)
            if start is None or row["value"] < start:
                continue
            key = (*publication, labels.get("spindle.worker"))
            if key not in ready or row["value"] < ready[key]["time"]:
                ready[key] = {
                    **row,
                    "name": "weight_sync.capture_to_ready_s",
                    "unit": "s",
                    "kind": "derived",
                    "time": row["value"],
                    "value": row["value"] - start,
                }
        return sorted(ready.values(), key=lambda row: row["time"])

    def series(self, run_ids, name):
        if name == "weight_sync.capture_to_ready_s":
            return self.weight_sync_series(run_ids)
        if not run_ids:
            return []
        placeholders = ",".join("?" for _ in run_ids)
        with self.connection() as db:
            return [
                dict(r) | {"labels": json.loads(r["labels"])}
                for r in db.execute(
                    f"SELECT run_id,time,name,value,labels,unit,kind FROM metrics WHERE run_id IN ({placeholders}) AND name=? ORDER BY time",
                    [*run_ids, name],
                )
            ]

    def bundle(self, run_id):
        with self.connection() as db:
            run = db.execute(
                "SELECT manifest FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            return {
                "schema_version": 1,
                "manifest": json.loads(run[0]),
                "records": [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT payload FROM records WHERE run_id=? ORDER BY time",
                        (run_id,),
                    )
                ],
            }

    def ingest(self, bundle):
        if (
            not isinstance(bundle, dict)
            or bundle.get("schema_version") != 1
            or not isinstance(bundle.get("records"), list)
        ):
            raise ValueError("expected a version 1 benchmark bundle")
        manifest = bundle.get("manifest", {})
        if not isinstance(manifest, dict) or any(
            not isinstance(record, dict) for record in bundle["records"]
        ):
            raise ValueError("manifest and records must be objects")
        if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
            raise ValueError("missing run_id")
        self.register(manifest)
        self.append_many((manifest["run_id"], record) for record in bundle["records"])
