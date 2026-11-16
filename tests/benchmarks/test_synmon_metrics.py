import asyncio
import json
from concurrent.futures import Future
from pathlib import Path

import httpx
import pytest
import tinker
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from tinker import types
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture

from spindle.benchmarks.client_metrics import ClientMetrics, ObservedFuture
from spindle.benchmarks.server import create_app
from synthetic_monitoring.metrics import SynmonMetrics
from tests.control_plane.test_sdk_e2e import API_KEY, BASE_MODEL, base_url as base_url

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")


def test_stock_sdk_metrics_survive_recipe_discarding_results(
    base_url: str, tmp_path: Path
) -> None:
    capture = SynmonMetrics("test", {}, directory=tmp_path / "capture")
    env = capture.environment(tmp_path / "recipe")
    observer = ClientMetrics(env["SPINDLE_BENCHMARK_LOG_DIR"], capture.run_id).install()
    try:
        service = tinker.ServiceClient(base_url=base_url, api_key=API_KEY)
        training = service.create_lora_training_client(BASE_MODEL, rank=32)
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=[2, 3, 4], dtype="int64", shape=[3]
                ),
                "weights": types.TensorData(
                    data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]
                ),
            },
        )
        training.forward_backward([datum], "cross_entropy").result(timeout=30)
        training.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=30)
        sampling = service.create_sampling_client(base_model=BASE_MODEL)
        future = sampling.sample(
            prompt=types.ModelInput.from_ints([1, 2]),
            num_samples=2,
            sampling_params=types.SamplingParams(max_tokens=3),
        )
        assert isinstance(future, Future)
        assert len(future.result(timeout=30).sequences) == 2

        async def check_async() -> None:
            result = await sampling.sample_async(
                prompt=types.ModelInput.from_ints([1, 2]),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=3),
            )
            assert len(result.sequences) == 1
            pending = await training.optim_step_async(
                types.AdamParams(learning_rate=2e-4)
            )
            assert (await pending).metrics == {"lr": 2e-4}

        asyncio.run(check_async())
    finally:
        observer.close()
    path, health = capture.finish("completed", {}, tmp_path / "durable")
    rows = [
        row["raw"]
        for row in json.loads(path.read_text())["records"]
        if row.get("raw", {}).get("event") == "sdk_result"
    ]
    assert len(rows) == 5
    backward = next(row for row in rows if row["operation"] == "forward_backward")
    assert backward["sdk"]["metrics"]["loss:sum"] == 1.25
    assert backward["sdk"]["input_tokens"] == 3
    assert sorted(
        row["sdk"]["output_tokens"] for row in rows if row["operation"] == "sample"
    ) == [3, 6]
    assert all(row["sdk"]["observed_latency_s"] >= 0 for row in rows)
    assert health["telemetry.server_records"] == 0
    assert "target_tokens" not in path.read_text() and API_KEY not in path.read_text()


def test_observed_future_preserves_timeout_retry_and_result_identity() -> None:
    future = Future()
    observed = []
    wrapped = ObservedFuture(AwaitableConcurrentFuture(future), observed.append)
    with pytest.raises(TimeoutError):
        wrapped.result(timeout=0)
    value = object()
    future.set_result(value)
    assert wrapped.result() is value

    async def check() -> None:
        assert await wrapped is value

    asyncio.run(check())
    assert observed == [value]


def test_failed_run_keeps_partial_and_unknown_metrics_importable(
    tmp_path: Path,
) -> None:
    capture = SynmonMetrics("test", {}, directory=tmp_path / "capture")
    root = tmp_path / "recipe"
    root.mkdir()
    capture.environment(root)
    event = {"event": "step_done", "step": 1, "new_backend_stat": {"queue_s": 0.125}}
    path = root / "events.jsonl"
    path.write_text(json.dumps(event)[:30])
    capture.ingest()
    path.write_text(json.dumps(event) + "\ninvalid\n" + json.dumps({"reward": 0.75}))
    capture.ingest()
    # URL validation rejects HTTP before any upload can occur.
    saved, health = capture.finish(
        "failed", {}, tmp_path / "durable", collector_url="http://127.0.0.1:1"
    )
    assert health["telemetry.collector_upload_ok"] == 0
    assert health["telemetry.collection_errors"] == 2
    bundle = json.loads(saved.read_text())
    assert bundle["manifest"]["status"] == "failed"
    assert sum(row.get("raw") == event for row in bundle["records"]) == 1
    assert capture.reader.completed == 1

    async def check() -> None:
        app = create_app(tmp_path / "dashboard")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/api/upload", json=bundle)).status_code == 200
            response = await client.get(
                "/api/series",
                params={
                    "runs": capture.run_id,
                    "metric": "recipe.new_backend_stat.queue_s",
                },
            )
            assert [point["value"] for point in response.json()] == [0.125]
            response = await client.get(
                "/api/series",
                params={"runs": capture.run_id, "metric": "recipe.reward"},
            )
            assert [point["value"] for point in response.json()] == [0.75]

    asyncio.run(check())


def test_shared_server_routes_model_metrics_and_spans_to_their_run(
    tmp_path: Path,
) -> None:
    metrics = ExportMetricsServiceRequest()
    resource = metrics.resource_metrics.add()
    resource.resource.attributes.add(
        key="spindle.run_id"
    ).value.string_value = "deployment"
    scope = resource.scope_metrics.add()
    scope.metrics.add(name="gpu_memory_bytes").gauge.data_points.add(as_int=100)
    owned = scope.metrics.add(name="trainer_step_s").gauge.data_points.add(
        as_double=0.25
    )
    owned.attributes.add(key="spindle.run_id").value.string_value = "synmon-run"
    traces = ExportTraceServiceRequest()
    resource = traces.resource_spans.add()
    resource.resource.attributes.add(
        key="spindle.run_id"
    ).value.string_value = "deployment"
    span = resource.scope_spans.add().spans.add(
        name="optimizer",
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=1_200_000_000,
    )
    span.attributes.add(key="spindle.run_id").value.string_value = "synmon-run"

    async def check() -> None:
        app = create_app(tmp_path / "dashboard")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (
                await client.post("/v1/metrics", content=metrics.SerializeToString())
            ).status_code == 200
            assert (
                await client.post("/v1/traces", content=traces.SerializeToString())
            ).status_code == 200
            run = (await client.get("/api/runs/synmon-run/bundle")).json()
            deployment = (await client.get("/api/runs/deployment/bundle")).json()
            assert len(run["records"]) == 2
            assert {name for row in run["records"] for name in row["metrics"]} == {
                "trainer_step_s",
                "span.duration_s",
            }
            assert deployment["records"][0]["metrics"] == {"gpu_memory_bytes": 100}

    asyncio.run(check())
