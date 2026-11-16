"""Portable fine-grained synmon results, independent of timing history."""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

import httpx

from spindle.benchmarks.auth import modal_proxy_headers
from spindle.benchmarks.events import EventReader
from spindle.benchmarks.runner import Recorder, safe_config
from spindle.benchmarks.store import Store
from spindle.version import source_commit


class SynmonMetrics:
    def __init__(
        self, name: str, config: dict[str, object], *, directory: Path | None = None
    ) -> None:
        self.run_id = "synmon-" + uuid.uuid4().hex
        self.directory = directory or Path(tempfile.mkdtemp(prefix="synmon-metrics-"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.roots: set[Path] = set()
        self.recorder = Recorder(
            Store(self.directory / "store"),
            {
                "run_id": self.run_id,
                "name": name,
                "suite": "async-rl",
                "status": "running",
                "created_at": time.time(),
                "revision_verified": False,
                "harness_revision": source_commit(),
                "config": safe_config(config),
                "collection_scope": "synmon",
                "server_telemetry": "requires instrumented server and matching run attribution",
            },
            self.directory,
        )
        self.reader = EventReader(self.recorder)

    def environment(self, output: Path) -> dict[str, str]:
        self.roots.add(output)
        return {
            "SPINDLE_BENCHMARK_RUN_ID": self.run_id,
            "SPINDLE_BENCHMARK_LOG_DIR": str(output / "sdk"),
        }

    def ingest(self, *, final: bool = False) -> None:
        for root in self.roots:
            try:
                self.reader.read(root, final=final)
            except Exception as exc:
                self.recorder.collection_error("recipe ingestion", type(exc).__name__)

    def finish(
        self,
        status: str,
        timings: dict[str, float],
        destination: Path,
        *,
        collector_url: str | None = None,
    ) -> tuple[Path, dict[str, float]]:
        self.ingest(final=True)
        self.recorder.record({"synmon": timings}, {"phase": "summary"})
        self.recorder.event(
            "coverage",
            {
                "client": "all emitted recipe fields and observed SDK metrics",
                "server": "only independently exported, run-attributed server telemetry; shared device metrics belong to their deployment stream",
            },
        )
        self.recorder.finish(status)
        bundle = self.recorder.store.bundle(self.run_id)
        uploaded = 0.0
        if collector_url:
            try:
                headers = modal_proxy_headers(collector_url)
                with httpx.Client(
                    base_url=collector_url.rstrip("/"), headers=headers, timeout=60
                ) as client:
                    response = client.post("/api/upload", json=bundle)
                    response.raise_for_status()
                    uploaded = 1.0
                    response = client.get(f"/api/runs/{self.run_id}/bundle")
                    response.raise_for_status()
                    combined = response.json()
                    if combined.get("manifest", {}).get("run_id") != self.run_id:
                        raise ValueError("Collector returned a different run")
                    self.recorder.store.ingest(combined)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                self.recorder.collection_error("collector", type(exc).__name__)
        records = self.recorder.store.bundle(self.run_id)["records"]
        server_records = sum(
            row.get("kind", "").startswith(("otlp.", "span")) for row in records
        )
        health = {
            "telemetry.records": float(len(records)),
            "telemetry.server_records": float(server_records),
            "telemetry.collector_upload_ok": uploaded,
            "telemetry.collection_errors": float(
                sum(row.get("kind") == "collection-error" for row in records)
            ),
        }
        self.recorder.record(health, {"phase": "summary"})
        path = self.recorder.finish(status)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{self.run_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(path.read_bytes())
        temporary.replace(target)
        return target, health


def collector_url() -> str | None:
    return os.getenv("SPINDLE_SYNMON_COLLECTOR_URL") or None
