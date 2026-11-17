"""Build sidecar routing views from hook and KV event snapshots."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import httpx

from .kv_events import KVEventSubscriber

RoutingSource = Literal["hook", "kv_events", "compare"]
ROUTING_SOURCES: tuple[RoutingSource, ...] = ("hook", "kv_events", "compare")
ROUTING_CATALOG_PATH = Path("/tmp/spindle-routing-work.json")
ROUTING_CATALOG_REFRESH_SECONDS = 0.5
ROUTING_SUBSCRIBER_MAX_AGE_SECONDS = 2.5
FETCH_METRICS_TIMEOUT_SECONDS = 2.0


class RequestWork(TypedDict):
    rid: str
    adapter: str
    prefill_tokens: int
    decode_tokens: int


class HookCatalog(TypedDict):
    timestamp: float
    requests: list[RequestWork]
    prefixes: list[str]
    cpu_adapters: list[str]
    gpu_adapters: list[str]
    snapshot_seconds: float
    cache_kind: str
    running_requests: int
    waiting_requests: int
    kv_used_tokens: NotRequired[int]
    kv_capacity_tokens: NotRequired[int]


class LoadSnapshot(TypedDict):
    timestamp: float
    num_running_reqs: int
    num_waiting_reqs: int
    num_used_tokens: int
    max_total_num_tokens: int


class WorkView(TypedDict):
    age_seconds: float
    prefixes: list[str]
    cpu_adapters: list[str]
    gpu_adapters: list[str]
    request_ids: list[str]
    pending_adapters: list[str]
    running_requests: int
    waiting_requests: int
    kv_used_tokens: int
    kv_capacity_tokens: int
    prefill_tokens: NotRequired[int]
    decode_tokens: NotRequired[int]
    snapshot_seconds: NotRequired[float]
    cache_kind: NotRequired[str]


class RoutingBody(TypedDict):
    replica: str
    active: int
    adapters: dict[str, int]
    request_ids: list[str]
    pending_adapters: list[str]
    cpu_adapters: list[str]
    age_seconds: float | None
    work: WorkView | None


class CompareView(TypedDict):
    prefix_count: int
    load: dict[str, int | None] | None
    age_seconds: float | None


class CompareSamples(TypedDict):
    hook_only: list[str]
    kv_events_only: list[str]


class CompareBody(TypedDict):
    hook: CompareView
    kv_events: CompareView
    intersection: int
    hook_only: int
    kv_events_only: int
    samples: CompareSamples
    subscriber_ready: bool


class RoutingState:
    """Own cached routing snapshots and build the sidecar's wire responses."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        source: RoutingSource,
        subscriber: KVEventSubscriber | None,
        routing_requests: dict[str, str],
        request_work: dict[str, RequestWork],
        registered_adapters: dict[str, str],
    ) -> None:
        self.client = client
        self.source = source
        self.subscriber = (
            (subscriber or KVEventSubscriber()) if source != "hook" else None
        )
        self.routing_requests = routing_requests
        self.request_work = request_work
        self.registered_adapters = registered_adapters
        self.catalog: HookCatalog | None = None
        self.read_at = 0.0
        self.loads: LoadSnapshot | None = None
        self.loads_at = 0.0
        self.loads_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.subscriber:
            self.subscriber.start_thread()

    async def stop(self) -> None:
        if self.subscriber:
            await asyncio.to_thread(self.subscriber.stop_thread)
        if self.loads_task:
            self.loads_task.cancel()
            await asyncio.gather(self.loads_task, return_exceptions=True)

    async def refresh_loads(self) -> None:
        try:
            response = await self.client.get(
                "/v1/loads?include=core", timeout=FETCH_METRICS_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            loads = response.json()["loads"]
            snapshot: LoadSnapshot | None = None
            if len(loads) == 1:
                row = loads[0]
                snapshot = {
                    "timestamp": row["timestamp"],
                    "num_running_reqs": row["num_running_reqs"],
                    "num_waiting_reqs": row["num_waiting_reqs"],
                    "num_used_tokens": row["num_used_tokens"],
                    "max_total_num_tokens": row["max_total_num_tokens"],
                }
            self.loads = snapshot
            self.loads_at = time.monotonic()
        except (httpx.HTTPError, ValueError, LookupError, TypeError):
            pass
        finally:
            self.loads_task = None

    def refresh_loads_if_needed(self) -> None:
        now = time.monotonic()
        if (
            self.loads_task is None
            and now - self.loads_at > ROUTING_CATALOG_REFRESH_SECONDS
        ):
            self.loads_task = asyncio.create_task(self.refresh_loads())

    async def hook_view(self) -> WorkView | None:
        now = time.monotonic()
        if now - self.read_at > ROUTING_CATALOG_REFRESH_SECONDS:
            self.read_at = now
            try:
                self.catalog = json.loads(
                    await asyncio.to_thread(ROUTING_CATALOG_PATH.read_text)
                )
            except (OSError, ValueError):
                self.catalog = None
        snapshot = self.catalog
        if not snapshot:
            return None
        rows = {
            row["rid"]: row
            for row in snapshot["requests"]
            if row["rid"] in self.routing_requests
        }
        rows.update(
            (rid, row) for rid, row in self.request_work.items() if rid not in rows
        )
        return dict(
            age_seconds=time.time() - snapshot["timestamp"],
            prefixes=snapshot["prefixes"],
            cpu_adapters=snapshot["cpu_adapters"],
            gpu_adapters=snapshot["gpu_adapters"],
            request_ids=list(rows),
            pending_adapters=sorted(set(self.routing_requests.values())),
            prefill_tokens=sum(row["prefill_tokens"] for row in rows.values()),
            decode_tokens=sum(row["decode_tokens"] for row in rows.values()),
            snapshot_seconds=snapshot["snapshot_seconds"],
            cache_kind=snapshot["cache_kind"],
            running_requests=snapshot["running_requests"],
            waiting_requests=snapshot["waiting_requests"],
            kv_used_tokens=snapshot.get("kv_used_tokens", 0),
            kv_capacity_tokens=snapshot.get("kv_capacity_tokens", 0),
        )

    def kv_view(self) -> WorkView | None:
        self.refresh_loads_if_needed()
        subscriber = self.subscriber
        load = self.loads
        if load is None or subscriber is None:
            return None
        snapshot = subscriber.snapshot
        return dict(
            age_seconds=max(0.0, time.time() - load["timestamp"]),
            prefixes=(
                list(snapshot.prefixes)
                if snapshot.ready
                and time.monotonic() - snapshot.checked_at
                < ROUTING_SUBSCRIBER_MAX_AGE_SECONDS
                else []
            ),
            cpu_adapters=list(self.registered_adapters),
            gpu_adapters=[],
            request_ids=list(self.routing_requests),
            pending_adapters=sorted(set(self.routing_requests.values())),
            running_requests=load["num_running_reqs"],
            waiting_requests=load["num_waiting_reqs"],
            kv_used_tokens=load["num_used_tokens"],
            kv_capacity_tokens=load["max_total_num_tokens"],
        )

    def routing_body(self, work: WorkView | None) -> RoutingBody:
        return {
            "replica": os.environ.get("MODAL_TASK_ID", "local"),
            "active": len(self.routing_requests),
            "adapters": dict(Counter(self.routing_requests.values())),
            "request_ids": list(self.routing_requests),
            "pending_adapters": sorted(set(self.routing_requests.values())),
            "cpu_adapters": work["cpu_adapters"] if work else [],
            "age_seconds": (
                time.time() - self.catalog["timestamp"]
                if self.source == "hook" and self.catalog
                else work["age_seconds"]
                if work
                else None
            ),
            "work": work,
        }

    async def body(self) -> RoutingBody:
        work = self.kv_view() if self.source == "kv_events" else await self.hook_view()
        return self.routing_body(work)

    @staticmethod
    def _compare_view(view: WorkView | None) -> CompareView:
        if view is None:
            return {"prefix_count": 0, "load": None, "age_seconds": None}
        fields = (
            "running_requests",
            "waiting_requests",
            "kv_used_tokens",
            "kv_capacity_tokens",
        )
        return {
            "prefix_count": len(view["prefixes"]),
            "load": {key: view.get(key) for key in fields},
            "age_seconds": view.get("age_seconds"),
        }

    @classmethod
    def _compare_routing_views(
        cls,
        hook: WorkView | None,
        kv: WorkView | None,
        subscriber: KVEventSubscriber | None,
    ) -> CompareBody:
        hook_prefixes = set((hook or {}).get("prefixes", []))
        kv_prefixes = set((kv or {}).get("prefixes", []))
        return {
            "hook": cls._compare_view(hook),
            "kv_events": cls._compare_view(kv),
            "intersection": len(hook_prefixes & kv_prefixes),
            "hook_only": len(hook_prefixes - kv_prefixes),
            "kv_events_only": len(kv_prefixes - hook_prefixes),
            "samples": {
                "hook_only": sorted(hook_prefixes - kv_prefixes)[:5],
                "kv_events_only": sorted(kv_prefixes - hook_prefixes)[:5],
            },
            "subscriber_ready": bool(subscriber and subscriber.snapshot.ready),
        }

    async def compare_body(self) -> CompareBody:
        hook = await self.hook_view()
        kv = self.kv_view()
        return self._compare_routing_views(hook, kv, self.subscriber)
