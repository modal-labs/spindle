from __future__ import annotations

import gzip
import asyncio
import io
import json
import os
import secrets
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from google.protobuf.message import DecodeError

from .otlp import decode_metrics, decode_traces
from .store import Store

MAX_BODY = 32 * 1024 * 1024


def create_app(directory, *, token=None, persist=None, persist_interval=5):
    store = Store(directory)
    access = asyncio.Lock()
    dirty = asyncio.Event()
    stopping = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app):
        async def commit():
            assert persist is not None
            while True:
                await dirty.wait()
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=persist_interval)
                except TimeoutError:
                    pass
                async with access:
                    try:
                        await asyncio.to_thread(persist)
                    except Exception as exc:
                        store.append_many(
                            (
                                run["run_id"],
                                {
                                    "kind": "collection-error",
                                    "source": "persistence",
                                    "error": type(exc).__name__,
                                    "metrics": {"collection.persistence_errors": 1},
                                },
                            )
                            for run in store.runs()
                        )
                    else:
                        dirty.clear()
                if stopping.is_set():
                    return

        task = asyncio.create_task(commit()) if persist else None
        try:
            yield
        finally:
            if task:
                stopping.set()
                dirty.set()
                await task

    app = FastAPI(title="Spindle benchmarks", lifespan=lifespan)
    app.state.store = store

    if persist:

        @app.middleware("http")
        async def serialize_snapshots(request: Request, call_next):
            async with access:
                response = await call_next(request)
                if request.method == "POST":
                    dirty.set()
                return response

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if (
            token
            and request.url.path != "/"
            and not secrets.compare_digest(
                request.headers.get("authorization", ""), "Bearer " + token
            )
        ):
            return JSONResponse(
                {"detail": "A benchmark access token is required"}, status_code=401
            )
        return await call_next(request)

    async def body(request):
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > MAX_BODY:
                raise HTTPException(413, "Maximum request size is 32 MiB")
        data = bytes(chunks)
        if request.headers.get("content-encoding") == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
                data = compressed.read(MAX_BODY + 1)
            if len(data) > MAX_BODY:
                raise HTTPException(413, "Maximum decoded request size is 32 MiB")
        return data

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).with_name("dashboard.html")).read_text()

    @app.get("/api/runs")
    def runs():
        return store.runs()

    @app.get("/api/metrics")
    def metrics(runs: str = ""):
        return store.catalog(runs.split(",") if runs else [])

    @app.get("/api/series")
    def series(metric: str, runs: str = ""):
        return store.series(runs.split(",") if runs else [], metric)

    @app.get("/api/runs/{run_id}/bundle")
    def download(run_id: str):
        try:
            return store.bundle(run_id)
        except KeyError:
            raise HTTPException(404, "Run not found") from None

    @app.post("/api/upload")
    async def upload(request: Request):
        try:
            bundle = json.loads(await body(request))
            store.ingest(bundle)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "run_id": bundle["manifest"]["run_id"],
            "records": len(bundle["records"]),
        }

    async def receive(request, signal):
        data = await body(request)
        try:
            records = list(
                (decode_metrics if signal == "metrics" else decode_traces)(data)
            )
        except (ValueError, DecodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        known = {run["run_id"] for run in store.runs()}
        for run_id, record in records:
            if run_id not in known:
                store.register(
                    {
                        "run_id": run_id,
                        "status": "telemetry-only",
                        "created_at": record["time"],
                    }
                )
                known.add(run_id)
        store.append_many(records)
        endpoint = os.getenv("SPINDLE_BENCHMARK_FORWARD_OTLP_ENDPOINT")
        if endpoint:
            headers = json.loads(os.getenv("SPINDLE_BENCHMARK_FORWARD_HEADERS", "{}"))
            headers["content-type"] = "application/x-protobuf"
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(
                        endpoint.rstrip("/") + "/v1/" + signal,
                        content=data,
                        headers=headers,
                    )
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                for run_id in {run_id for run_id, _ in records}:
                    store.append(
                        run_id,
                        {
                            "kind": "collection-error",
                            "source": "otlp-forward." + signal,
                            "error": type(exc).__name__,
                            "metrics": {"collection.forward_errors": 1},
                        },
                    )
                raise HTTPException(
                    503, "Telemetry saved locally; forwarding failed and may be retried"
                ) from None
        return Response(content=b"", media_type="application/x-protobuf")

    @app.post("/v1/metrics")
    async def otlp_metrics(request: Request):
        return await receive(request, "metrics")

    @app.post("/v1/traces")
    async def otlp_traces(request: Request):
        return await receive(request, "traces")

    return app
