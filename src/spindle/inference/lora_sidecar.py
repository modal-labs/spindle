from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import modal
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from stitch.types import VersionConstraint, VersionRef

from spindle.startup_timing import timeline

from .bulletin import SnapshotBulletin, SnapshotNotFound
from .request_replay import RequestReplay
from .kv_events import KVEventSubscriber, cache_salt


ABORT_TIMEOUT_SECONDS = 2.0
ROUTING_CATALOG_PATH = Path("/tmp/spindle-routing-work.json")
ROUTING_CATALOG_REFRESH_SECONDS = 0.5


def _routing_adapter(payload: dict) -> str:
    """Exact pins are known before load; other requests use the resolved path."""
    constraint = VersionConstraint.from_payload(payload)
    run_id = payload.get("weight_run_id")
    if run_id is not None and constraint.exact_version is not None:
        return VersionRef(str(run_id), constraint.exact_version).identity
    return payload.get("lora_path", "base")


FETCH_METRICS_TIMEOUT_SECONDS = 2.0


async def refresh_loads(app):
    try:
        response = await app.state.client.get("/v1/loads?include=core", timeout=0.25)
        response.raise_for_status()
        loads = response.json()["loads"]
        app.state.loads = loads[0] if len(loads) == 1 else None
        app.state.loads_at = time.monotonic()
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    finally:
        app.state.loads_task = None


def hook_view(app):
    snapshot = app.state.routing_catalog
    if not snapshot:
        return None
    rows = {
        row["rid"]: row
        for row in snapshot["requests"]
        if row["rid"] in app.state.routing_requests
    }
    rows.update(
        (rid, row) for rid, row in app.state.request_work.items() if rid not in rows
    )
    return dict(
        age_seconds=time.time() - snapshot["timestamp"],
        prefixes=snapshot["prefixes"],
        cpu_adapters=snapshot["cpu_adapters"],
        gpu_adapters=snapshot["gpu_adapters"],
        request_ids=list(rows),
        pending_adapters=sorted(set(app.state.routing_requests.values())),
        prefill_tokens=sum(row["prefill_tokens"] for row in rows.values()),
        decode_tokens=sum(row["decode_tokens"] for row in rows.values()),
        snapshot_seconds=snapshot["snapshot_seconds"],
        cache_kind=snapshot["cache_kind"],
        running_requests=snapshot["running_requests"],
        waiting_requests=snapshot["waiting_requests"],
        kv_used_tokens=snapshot.get("kv_used_tokens", 0),
        kv_capacity_tokens=snapshot.get("kv_capacity_tokens", 0),
    )


def _compare_view(view):
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


def routing_body(app, work, snapshot=None):
    return {
        "replica": os.environ.get("MODAL_TASK_ID", "local"),
        "active": len(app.state.routing_requests),
        "adapters": dict(Counter(app.state.routing_requests.values())),
        "request_ids": list(app.state.routing_requests),
        "pending_adapters": sorted(set(app.state.routing_requests.values())),
        "cpu_adapters": (work or {}).get(
            "cpu_adapters",
            (snapshot or {}).get("cpu_adapters", list(app.state.registered_adapters)),
        ),
        "age_seconds": work.get("age_seconds") if work else None,
        "work": work,
    }


def create_app(
    bulletin: SnapshotBulletin,
    upstream_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    routing_source: str = "hook",
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=upstream_url.rstrip("/"),
            timeout=3000,
            trust_env=False,
            transport=transport,
            # SGLang owns admission. Long generations must not exhaust the
            # HTTP pool before its configured concurrency or block control calls.
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
        )
        app.state.adapter_lock = asyncio.Lock()
        app.state.registered_adapters = {}
        app.state.adapter_resolutions = {}
        app.state.routing_requests = {}
        app.state.request_work = {}
        app.state.routing_catalog = {}
        app.state.routing_read_at = 0.0
        app.state.request_replay = RequestReplay()
        app.state.kv_events = KVEventSubscriber() if routing_source != "hook" else None
        app.state.kv_task = (
            asyncio.create_task(app.state.kv_events.run())
            if app.state.kv_events
            else None
        )
        app.state.loads = None
        app.state.loads_at = 0.0
        app.state.loads_task = None
        try:
            yield
        finally:
            if app.state.kv_task:
                app.state.kv_task.cancel()
                await asyncio.gather(app.state.kv_task, return_exceptions=True)
            if app.state.loads_task:
                app.state.loads_task.cancel()
                await asyncio.gather(app.state.loads_task, return_exceptions=True)
            await app.state.request_replay.aclose()
            pending = list(app.state.adapter_resolutions.values())
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/metrics")
    async def metrics() -> Response:
        try:
            response = await app.state.client.get(
                "/metrics", timeout=FETCH_METRICS_TIMEOUT_SECONDS
            )
        except httpx.RequestError:
            return Response(status_code=503)
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.get("/routing-state")
    async def routing_state():
        now = time.monotonic()
        if now - app.state.routing_read_at > ROUTING_CATALOG_REFRESH_SECONDS:
            app.state.routing_read_at = now
            try:
                app.state.routing_catalog = json.loads(
                    await asyncio.to_thread(ROUTING_CATALOG_PATH.read_text)
                )
            except (OSError, ValueError):
                app.state.routing_catalog = {}
        snapshot = app.state.routing_catalog
        if routing_source == "kv_events":
            if (
                app.state.loads_task is None
                and now - app.state.loads_at > ROUTING_CATALOG_REFRESH_SECONDS
            ):
                app.state.loads_task = asyncio.create_task(refresh_loads(app))
            subscriber = app.state.kv_events
            load = app.state.loads
            work = None
            if load is not None:
                work = dict(
                    age_seconds=max(0.0, time.time() - load["timestamp"]),
                    prefixes=list(subscriber.cache.prefixes)
                    if subscriber.ready
                    and time.monotonic() - subscriber.checked_at < 10
                    else [],
                    cpu_adapters=list(app.state.registered_adapters),
                    gpu_adapters=[],
                    request_ids=list(app.state.routing_requests),
                    pending_adapters=sorted(set(app.state.routing_requests.values())),
                    prefill_tokens=0,
                    decode_tokens=0,
                    snapshot_seconds=0.0,
                    cache_kind="kv_events",
                    running_requests=load["num_running_reqs"],
                    waiting_requests=load["num_waiting_reqs"],
                    kv_used_tokens=load["num_used_tokens"],
                    kv_capacity_tokens=load["max_total_num_tokens"],
                )
            return routing_body(app, work)
        pending_adapters = sorted(set(app.state.routing_requests.values()))
        work = None
        if snapshot:
            rows = {
                row["rid"]: row
                for row in snapshot["requests"]
                if row["rid"] in app.state.routing_requests
            }
            rows.update(
                (rid, row)
                for rid, row in app.state.request_work.items()
                if rid not in rows
            )
            work = dict(
                age_seconds=time.time() - snapshot["timestamp"],
                prefixes=snapshot["prefixes"],
                cpu_adapters=snapshot["cpu_adapters"],
                gpu_adapters=snapshot["gpu_adapters"],
                request_ids=list(rows),
                pending_adapters=pending_adapters,
                prefill_tokens=sum(row["prefill_tokens"] for row in rows.values()),
                decode_tokens=sum(row["decode_tokens"] for row in rows.values()),
                snapshot_seconds=snapshot["snapshot_seconds"],
                cache_kind=snapshot["cache_kind"],
                running_requests=snapshot["running_requests"],
                waiting_requests=snapshot["waiting_requests"],
                kv_used_tokens=snapshot.get("kv_used_tokens", 0),
                kv_capacity_tokens=snapshot.get("kv_capacity_tokens", 0),
            )
        return routing_body(app, work, snapshot)

    @app.get("/routing-state/compare")
    async def routing_compare():
        await routing_state()
        now = time.monotonic()
        if (
            app.state.loads_task is None
            and now - app.state.loads_at > ROUTING_CATALOG_REFRESH_SECONDS
        ):
            app.state.loads_task = asyncio.create_task(refresh_loads(app))
        hook = hook_view(app)
        load = app.state.loads
        subscriber = app.state.kv_events
        kv = (
            None
            if load is None or subscriber is None
            else {
                "prefixes": list(subscriber.cache.prefixes)
                if subscriber.ready and time.monotonic() - subscriber.checked_at < 10
                else [],
                "age_seconds": max(0.0, time.time() - load["timestamp"]),
                "running_requests": load["num_running_reqs"],
                "waiting_requests": load["num_waiting_reqs"],
                "kv_used_tokens": load["num_used_tokens"],
                "kv_capacity_tokens": load["max_total_num_tokens"],
            }
        )
        hook_prefixes = (hook or {}).get("prefixes", [])
        kv_prefixes = (kv or {}).get("prefixes", [])
        left = set(hook_prefixes) if isinstance(hook_prefixes, list) else set()
        right = set(kv_prefixes) if isinstance(kv_prefixes, list) else set()
        return {
            "hook": _compare_view(hook),
            "kv_events": _compare_view(kv),
            "intersection": len(left & right),
            "hook_only": len(left - right),
            "kv_events_only": len(right - left),
            "samples": {
                "hook_only": sorted(left - right)[:5],
                "kv_events_only": sorted(right - left)[:5],
            },
            "subscriber_ready": bool(subscriber and subscriber.ready),
            "adapter_attribution_supported": bool(
                subscriber and subscriber.cache.adapter_names_seen
            ),
        }

    @app.get("/health")
    async def health() -> Response:
        response = await app.state.client.get("/health")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        payload = await request.json()
        rid = payload.setdefault("rid", f"spindle-{uuid.uuid4().hex}")
        # Deduplicate scalar attempts. Preserve the existing batch-ID path;
        # its separate engine request IDs do not identify one shared attempt.
        replay_id = rid if isinstance(rid, str) else object()
        response = await app.state.request_replay.run(
            replay_id,
            await request.body(),
            lambda: generate_once(payload),
            request.is_disconnected,
        )
        if response is None:
            response = Response(status_code=499)
        response.headers["x-spindle-replica"] = os.environ.get("MODAL_TASK_ID", "local")
        return response

    def remember_request(payload: dict, adapter: str | None = None) -> None:
        rid = payload.get("rid")
        if not isinstance(rid, str):
            return
        if adapter is None:
            adapter = _routing_adapter(payload)
        app.state.routing_requests[rid] = adapter
        work = app.state.request_work.get(rid)
        if work is None:
            sampling = payload.get("sampling_params") or {}
            app.state.request_work[rid] = dict(
                rid=rid,
                adapter=adapter,
                prefill_tokens=len(payload.get("input_ids", ())),
                decode_tokens=sampling.get("max_new_tokens", 1),
            )
        else:
            work["adapter"] = adapter

    async def generate_once(payload: dict) -> Response:
        rid = payload["rid"]
        remember_request(payload)
        try:
            return await generate_response(payload)
        finally:
            if isinstance(rid, str):
                app.state.routing_requests.pop(rid, None)
                app.state.request_work.pop(rid, None)

    async def generate_response(payload: dict) -> Response:
        run_id = payload.pop("weight_run_id", None)
        constraint = VersionConstraint.from_payload(payload)
        payload.pop("weight_version", None)
        served_version = None
        if run_id is not None:
            try:
                ref, registration_error = await _prepare_adapter(
                    app, bulletin, str(run_id), constraint
                )
                if registration_error is not None:
                    return Response(
                        content=registration_error.content,
                        status_code=registration_error.status_code,
                        media_type=registration_error.headers.get("content-type"),
                    )
                payload["lora_path"] = ref.identity
                served_version = ref.version
            except (SnapshotNotFound, ValueError) as exc:
                return JSONResponse(
                    {"error": f"adapter version is not ready: {exc}"},
                    status_code=409,
                )

        adapter = payload.get("lora_path", "base")
        if routing_source == "kv_events":
            payload["cache_salt"] = cache_salt(adapter, payload.get("cache_salt"))
        remember_request(payload, adapter)
        response = await _post_generate(app.state.client, payload)
        try:
            body = response.json()
        except ValueError:
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )
        if served_version is not None and isinstance(body, dict):
            metadata = body.setdefault("meta_info", {})
            metadata["weight_version_start"] = served_version
            metadata["weight_version_end"] = served_version
        return JSONResponse(body, status_code=response.status_code)

    return app


async def _prepare_adapter(
    app: FastAPI,
    bulletin: SnapshotBulletin,
    run_id: str,
    constraint: VersionConstraint,
) -> tuple[VersionRef, httpx.Response | None]:
    ref = (
        VersionRef(run_id, constraint.exact_version)
        if constraint.exact_version is not None
        else None
    )
    if ref is not None and ref.identity in app.state.registered_adapters:
        return ref, None
    key = (run_id, constraint.exact_version, constraint.min_version)
    pending = app.state.adapter_resolutions
    task = pending.get(key)
    if task is None:

        async def prepare():
            # Reload must not hide paths while SGLang reads a new adapter.
            async with app.state.adapter_lock:
                if ref is not None and ref.identity in app.state.registered_adapters:
                    return ref, None
                if ref is not None:
                    resolved = ref
                    try:
                        path = await asyncio.to_thread(bulletin.resolve, resolved)
                    except SnapshotNotFound:
                        # Exact versions are immutable: reload only if this mount
                        # does not yet contain the complete requested snapshot.
                        await bulletin.refresh()
                        path = await asyncio.to_thread(bulletin.resolve, resolved)
                else:
                    # Latest/minimum requests must see newly published policies.
                    await bulletin.refresh()
                    resolved = bulletin.read_latest(run_id)
                    if resolved is None:
                        raise SnapshotNotFound(run_id)
                    if (
                        constraint.min_version is not None
                        and resolved.version < constraint.min_version
                    ):
                        raise SnapshotNotFound(
                            f"{run_id} latest={resolved.version} required={constraint.min_version}"
                        )
                    if resolved.identity in app.state.registered_adapters:
                        return resolved, None
                    path = await asyncio.to_thread(bulletin.resolve, resolved)
                response = await app.state.client.post(
                    "/load_lora_adapter",
                    json={
                        "lora_name": resolved.identity,
                        "lora_path": str(path),
                        "pinned": False,
                    },
                )
                if response.is_error and not await _adapter_is_registered(
                    app.state.client, resolved.identity, str(path)
                ):
                    return resolved, response
                if not app.state.registered_adapters:
                    timeline("sampler-sidecar").mark(
                        "first_adapter_loaded", adapter=resolved.identity
                    )
                app.state.registered_adapters[resolved.identity] = str(path)
                return resolved, None

        task = asyncio.create_task(prepare())
        pending[key] = task

        def finished(completed):
            if pending.get(key) is completed:
                del pending[key]
            if not completed.cancelled():
                # A disconnected caller may leave no waiter for this failure.
                completed.exception()

        task.add_done_callback(finished)
    # Share only in-flight work. Later requests still resolve the latest version.
    # One disconnected request must not cancel registration for other callers.
    return await asyncio.shield(task)


async def _adapter_is_registered(
    client: httpx.AsyncClient, name: str, path: str
) -> bool:
    try:
        response = await client.get("/v1/models", timeout=10.0)
        response.raise_for_status()
        models = response.json().get("data", [])
        return any(
            model.get("id") == name and model.get("root") == path for model in models
        )
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        return False


async def _post_generate(
    client: httpx.AsyncClient,
    payload: dict,
) -> httpx.Response:
    try:
        return await client.post("/generate", json=payload)
    except asyncio.CancelledError:
        await _abort(client, payload["rid"])
        raise


async def _abort(client: httpx.AsyncClient, rid) -> None:
    rids = rid if isinstance(rid, list) else [rid]
    try:
        async with asyncio.timeout(ABORT_TIMEOUT_SECONDS):
            await asyncio.gather(
                *(
                    client.post("/abort_request", json={"rid": str(item)})
                    for item in rids
                ),
                return_exceptions=True,
            )
    except TimeoutError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--bulletin-root", required=True)
    parser.add_argument("--bulletin-volume", default="")
    parser.add_argument(
        "--routing-source", choices=("hook", "kv_events", "compare"), default="hook"
    )
    args = parser.parse_args()

    refresh = None
    if args.bulletin_volume:
        refresh = modal.Volume.from_name(args.bulletin_volume, version=2).reload
    bulletin = SnapshotBulletin(Path(args.bulletin_root), refresh=refresh)
    uvicorn.run(
        create_app(bulletin, args.upstream_url, routing_source=args.routing_source),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
