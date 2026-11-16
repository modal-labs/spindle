import asyncio
import gzip
import base64
import runpy
from pathlib import Path
import json
import math
import subprocess
from types import SimpleNamespace

import httpx
import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from spindle.benchmarks.__main__ import comparison
from spindle.benchmarks.auth import modal_proxy_headers
from spindle.benchmarks import runner
from spindle.benchmarks.schema import BenchmarkConfig
from spindle.benchmarks.server import create_app
from spindle.benchmarks.store import Store
from spindle.benchmarks.workloads import command_suite, recipe_command
from spindle.benchmarks import workloads
from spindle.telemetry.benchmark import (
    BenchmarkMetrics,
    prometheus_samples,
    scrape_once,
)
from spindle.telemetry import trainer as trainer_telemetry
from spindle.proto.responses import encode_forward_backward_exact


@pytest.mark.parametrize("encoded", [False, True])
def test_trainer_result_metrics_survive_json_and_binary_transports(
    monkeypatch, encoded
):
    metrics = {"loss:mean": 2.0, "new_metric:sum": 3.0}
    result = {
        "loss_fn_output_type": "cross_entropy",
        "loss_fn_outputs": [],
        "metrics": metrics,
    }
    if encoded:
        result = encode_forward_backward_exact(result)
    records = []
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setattr(trainer_telemetry, "benchmark_enabled", lambda: True)
    monkeypatch.setattr(
        trainer_telemetry,
        "record_benchmark_metrics",
        lambda values, labels, prefix: records.append((values, labels)),
    )
    observer = trainer_telemetry.TrainerTelemetry("instance", "definition", "boot")
    observer.finish(
        SimpleNamespace(
            request_id="request",
            model_id="model",
            seq_id=7,
            kind=SimpleNamespace(value="forward_backward"),
        ),
        SimpleNamespace(result=result),
    )
    assert records[0][0] == metrics and records[0][1]["spindle.model_id"] == "model"


@pytest.mark.parametrize(
    "url",
    [
        "https://collector.example.com",
        "https://app.modal.run.evil.test",
        "http://app.modal.run",
        "https://user:password@app.modal.run",
    ],
)
def test_proxy_credentials_are_restricted_to_https_modal_endpoints(url, monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "test-secret")
    with pytest.raises(ValueError, match="HTTPS Modal"):
        modal_proxy_headers(url)


def metrics_payload(run_id="test-run"):
    request = ExportMetricsServiceRequest()
    resource = request.resource_metrics.add()
    resource.resource.attributes.add(key="spindle.run_id").value.string_value = run_id
    scope = resource.scope_metrics.add()
    metric = scope.metrics.add(name="sglang:token_usage", unit="1")
    point = metric.gauge.data_points.add(time_unix_nano=2_000_000_000, as_double=0.75)
    point.attributes.add(key="tp_rank").value.int_value = 1
    histogram = scope.metrics.add(name="queue_seconds", unit="s")
    histogram.histogram.data_points.add(
        time_unix_nano=2_000_000_000,
        count=3,
        sum=4.0,
        bucket_counts=[1, 2],
        explicit_bounds=[1.0],
    )
    return request.SerializeToString()


def test_otlp_upload_roundtrip_preserves_unknown_metrics_histograms_and_traces(
    tmp_path,
):
    async def check():
        app = create_app(tmp_path / "first", token="test-token")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/api/runs")).status_code == 401
            client.headers["Authorization"] = "Bearer test-token"
            for _ in range(2):
                response = await client.post(
                    "/v1/metrics",
                    content=gzip.compress(metrics_payload()),
                    headers={
                        "Content-Type": "application/x-protobuf",
                        "Content-Encoding": "gzip",
                    },
                )
                assert response.status_code == 200
            request = ExportTraceServiceRequest()
            resource = request.resource_spans.add()
            resource.resource.attributes.add(
                key="spindle.run_id"
            ).value.string_value = "test-run"
            span = resource.scope_spans.add().spans.add(
                name="forward_backward",
                trace_id=b"1" * 16,
                span_id=b"2" * 8,
                start_time_unix_nano=1_000_000_000,
                end_time_unix_nano=3_000_000_000,
            )
            span.attributes.add(key="spindle.loss_tokens").value.int_value = 64
            assert (
                await client.post("/v1/traces", content=request.SerializeToString())
            ).status_code == 200
            catalog = (
                await client.get("/api/metrics", params={"runs": "test-run"})
            ).json()
            assert {m["name"] for m in catalog} >= {
                "sglang:token_usage",
                "queue_seconds.bucket_counts.1",
                "span.duration_s",
                "spindle.loss_tokens",
            }
            points = (
                await client.get(
                    "/api/series",
                    params={"runs": "test-run", "metric": "sglang:token_usage"},
                )
            ).json()
            assert len(points) == 1
            assert points[0]["value"] == 0.75 and points[0]["labels"]["tp_rank"] == 1
            bundle = (await client.get("/api/runs/test-run/bundle")).json()
            histogram = next(
                r for r in bundle["records"] if r["kind"] == "otlp.histogram"
            )
            assert histogram["raw"]["point"]["bucket_counts"] == ["1", "2"]
            target = create_app(tmp_path / "second")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=target), base_url="http://test"
            ) as other:
                assert (await other.post("/api/upload", json=bundle)).status_code == 200
                assert (await other.get("/api/runs/test-run/bundle")).json() == bundle

    asyncio.run(check())


def test_invalid_otlp_and_nonfinite_samples_are_not_fabricated_zeroes(tmp_path):
    async def check():
        app = create_app(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (
                await client.post("/v1/metrics", content=b"invalid protobuf")
            ).status_code == 400
            missing_id = ExportMetricsServiceRequest()
            missing_id.resource_metrics.add()
            assert (
                await client.post("/v1/metrics", content=missing_id.SerializeToString())
            ).status_code == 400
            request = ExportMetricsServiceRequest.FromString(metrics_payload())
            request.resource_metrics[0].scope_metrics[0].metrics[0].gauge.data_points[
                0
            ].as_double = math.nan
            assert (
                await client.post("/v1/metrics", content=request.SerializeToString())
            ).status_code == 200
            rows = app.state.store.series(["test-run"], "sglang:token_usage")
            assert rows == []
            raw = app.state.store.bundle("test-run")["records"]
            assert any(
                r.get("metrics", {}).get("sglang:token_usage") == "nan" for r in raw
            )

    asyncio.run(check())


def test_persistence_batches_concurrent_uploads_and_flushes_on_shutdown(tmp_path):
    counts = []

    def persist():
        counts.append(len(app.state.store.bundle("persisted")["records"]))

    app = create_app(tmp_path, persist=persist, persist_interval=60)

    async def check():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                responses = await asyncio.gather(
                    *(
                        client.post(
                            "/api/upload",
                            json={
                                "schema_version": 1,
                                "manifest": {"run_id": "persisted"},
                                "records": [
                                    {"time": index, "metrics": {"value": index}}
                                ],
                            },
                        )
                        for index in range(8)
                    )
                )
                assert all(response.status_code == 200 for response in responses)
                assert counts == []

    asyncio.run(check())
    assert counts == [8]


def test_native_prometheus_samples_keep_bucket_labels_and_report_failure(monkeypatch):
    text = '# TYPE sglang:latency histogram\nsglang:latency_bucket{le="+Inf",model="a\\"b"} 3\nsglang:latency_sum 4.5\n# TYPE unknown_total counter\nunknown_total 9\n'
    samples = list(prometheus_samples(text))
    assert samples[0] == (
        "sglang:latency_bucket",
        3.0,
        {
            "le": "+Inf",
            "model": 'a"b',
            "prometheus.type": "histogram",
            "prometheus.name": "sglang:latency_bucket",
        },
    )
    observed = []
    monkeypatch.setattr(
        "spindle.telemetry.benchmark.record",
        lambda metrics, labels=None: observed.append((metrics, labels)),
    )

    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr("spindle.telemetry.benchmark.httpx.get", unavailable)
    scrape_once(["http://localhost/metrics"], devices=False)
    assert observed == [
        ({"spindle.collection.scrape_ok": 0}, {"source": "http://localhost/metrics"})
    ]


def test_arbitrary_backend_metric_names_survive_otel_validation():
    reader = InMemoryMetricReader()
    sink = BenchmarkMetrics(metric_reader=reader)
    names = {"loss:mean": 1.5, "long." + "x" * 100: 2.5, "a_b": 3.5, "a:b": 4.5}
    sink.record(names)
    data = reader.get_metrics_data()
    metrics = data.resource_metrics[0].scope_metrics[0].metrics
    assert len({m.name for m in metrics}) == len(names)
    assert {
        m.data.data_points[0].attributes[
            "spindle.metric.original_name"
        ]: m.data.data_points[0].value
        for m in metrics
    } == names
    sink.close()
    sink.close()


def test_comparison_refuses_mismatched_workload_and_unverified_revision(tmp_path):
    store = Store(tmp_path)
    for name, value in (("baseline", 2), ("candidate", 3)):
        store.register(
            {
                "run_id": name,
                "status": "completed",
                "revision_verified": True,
                "workload_fingerprint": "same",
                "configuration_fingerprint": "same",
                "resolved_deployment": {"recipe": {"weights_revision": "pinned"}},
            }
        )
        store.append(
            name,
            {
                "time": 1,
                "kind": "otlp.gauge",
                "metrics": {"spindle.gpu.memory.total": 80000},
                "labels": {
                    "gpu.uuid": name,
                    "gpu.name": "H100",
                    "spindle.component": "trainer",
                },
            },
        )
        for step in range(3):
            store.append(
                name,
                {
                    "kind": "measurement",
                    "time": step + 1,
                    "metrics": {"step_s": value},
                    "labels": {"step": step, "phase": "measurement", "clients": 1},
                },
            )
        store.append(
            name,
            {
                "kind": "measurement",
                "time": 4,
                "metrics": {"step_s": 999},
                "labels": {"phase": "warmup"},
            },
        )
    assert (
        comparison(store, "baseline", "candidate")["metrics"][0]["change_percent"] == 50
    )
    store.register(
        {
            "run_id": "candidate",
            "revision_verified": False,
            "workload_fingerprint": "changed",
        }
    )
    with pytest.raises(ValueError, match="workload_fingerprint"):
        comparison(store, "baseline", "candidate")
    assert (
        comparison(store, "baseline", "candidate", exploratory=True)["metrics"][0][
            "change_percent"
        ]
        is None
    )


def test_failed_run_keeps_measurements_and_restores_environment(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "uv.lock").write_text("locked")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "name": "failure",
                "suite": "trainer",
                "model": "test",
                "clients": [1],
                "context_lengths": [8],
            }
        )
    )

    async def fail(config, url, recorder):
        recorder.record({"retained": 123}, {"phase": "measurement"})
        raise RuntimeError("backend disconnected")

    monkeypatch.setattr(runner, "sdk_suite", fail)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "original")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=root)
    (root / "uv.lock").write_text("uncommitted changes")
    with pytest.raises(RuntimeError, match="disconnected"):
        runner.run(
            config,
            root=root,
            directory=tmp_path / "results",
            revision=sha,
            base_url="http://test",
            source_archive=archive,
        )
    store = Store(tmp_path / "results")
    run = store.runs()[0]
    assert run["status"] == "failed" and run["revision_verified"] is False
    assert run["source_lock_sha256"] == runner.digest("locked")
    assert store.series([run["run_id"]], "retained")[0]["value"] == 123
    assert next((tmp_path / "results/artifacts").glob("*/results.json")).is_file()
    assert runner.os.environ["OTEL_SERVICE_NAME"] == "original"


@pytest.mark.parametrize(
    "suite,fail_optimizer",
    [
        ("inference", False),
        ("trainer", False),
        ("eval", False),
        ("weight-sync", False),
        ("trainer", True),
    ],
)
def test_sdk_suites_retain_warmup_results_and_backend_metrics(
    tmp_path, monkeypatch, suite, fail_optimizer
):
    class Future:
        def __init__(self, metrics):
            self.metrics = metrics

        async def result_async(self):
            return SimpleNamespace(metrics=self.metrics)

    class Trainer:
        model_id = "test-model"

        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda text, **kw: [1, 2, 3, 4])

        async def forward_async(self, data, loss):
            assert suite == "eval"
            assert data[0].loss_fn_inputs["target_tokens"].shape == [16]
            return Future({"loss:mean": 0.5})

        async def forward_backward_async(self, data, loss):
            assert suite != "eval"
            assert data[0].loss_fn_inputs["target_tokens"].shape == [16]
            return Future({"loss:mean": 0.5})

        async def optim_step_async(self, params):
            assert suite != "eval"
            if fail_optimizer:
                raise RuntimeError("optimizer disconnected")
            return Future({"update_successful:mean": 1.0, "timing/internal_s": 0.01})

        async def save_weights_and_get_sampling_client_async(self):
            return self

        async def sample_async(self, prompt, **kwargs):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1, 2])])

    class Service:
        def __init__(self, **kwargs):
            pass

        async def create_lora_training_client_async(self, *args, **kwargs):
            assert kwargs["user_metadata"]["run_id"] == "sdk"
            return Trainer()

    monkeypatch.setattr(workloads.tinker, "ServiceClient", Service)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        workloads.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            **kwargs,
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
        ),
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    store = Store(tmp_path)
    recorder = runner.Recorder(store, {"run_id": "sdk"}, tmp_path)
    config = BenchmarkConfig(
        name="sdk",
        suite=suite,
        model="test",
        clients=[2],
        context_lengths=[16],
        steps=2,
        warmup_steps=1,
    )
    if fail_optimizer:
        with pytest.raises(RuntimeError, match="disconnected"):
            asyncio.run(workloads.sdk_suite(config, "http://test", recorder))
        assert store.series(["sdk"], "forward_backward.loss:mean")[0]["value"] == 0.5
        return
    asyncio.run(workloads.sdk_suite(config, "http://test", recorder))
    name = {
        "inference": "sampling.latency_s",
        "trainer": "training.step_s",
        "eval": "eval.forward_s",
        "weight-sync": "weight_sync.update_to_first_response_s",
    }[suite]
    samples = store.series(["sdk"], name)
    assert (
        len(samples) == 6
        and sum(r["labels"]["phase"] == "warmup" for r in samples) == 2
    )
    if suite not in {"inference", "eval"}:
        assert len(store.series(["sdk"], "optimizer.timing/internal_s")) == 6
    if suite == "eval":
        assert not store.series(["sdk"], "training.optimizer_s")
        assert (
            store.series(["sdk"], "aggregate.eval_input_tokens_per_s")[0]["value"] > 0
        )


def test_command_recipe_retains_partial_metrics_on_failure(tmp_path):
    config = BenchmarkConfig(
        name="failure",
        suite="async-rl",
        recipe="command",
        model="test",
        clients=[1],
        context_lengths=[8],
        command=[
            runner.sys.executable,
            "-c",
            'import pathlib,sys; pathlib.Path("events.jsonl").write_text(\'{"event":"train_done","nested":{"policy_lag":2}}\\n\');sys.exit(1)',
        ],
    )
    store = Store(tmp_path / "store")
    recorder = runner.Recorder(store, {"run_id": "partial"}, tmp_path)
    with pytest.raises(RuntimeError, match="exited"):
        command_suite(config, tmp_path, tmp_path, "http://test", recorder)
    assert store.series(["partial"], "recipe.nested.policy_lag")[0]["value"] == 2


def test_swe_adapter_applies_config_and_keeps_helpers(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text('MODEL="old"\nTRAINING={"context_tokens":100,"turn_tokens":20}\n')
    (tmp_path / "helpers").mkdir()
    (tmp_path / "helpers/identity.py").write_text("version=1")
    monkeypatch.setenv("SWE_GYM_WORKER", str(worker))
    config = BenchmarkConfig(
        name="swe",
        suite="async-rl",
        recipe="swe-gym",
        model="new",
        clients=[2],
        context_lengths=[1024],
        output_tokens=64,
    )
    output = tmp_path / "output"
    output.mkdir()
    command, work = recipe_command(config, tmp_path, output, "http://test")
    namespace = {}
    exec((work / "swe_gym.py").read_text(), namespace)
    assert (
        namespace["MODEL"] == "new" and namespace["TRAINING"]["context_tokens"] == 1024
    )
    assert (work / "helpers/identity.py").is_file()
    assert not (work / "results").exists()
    assert command[-1] == str(work / "results")


@pytest.mark.parametrize("status", [200, 503])
def test_control_plane_load_retains_samples_and_failure_summary(
    tmp_path, monkeypatch, status
):
    original_client = httpx.AsyncClient
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    def respond(request):
        assert request.headers["x-api-key"] == "test-key"
        assert request.url.path == "/api/v1/get_server_capabilities"
        return httpx.Response(status, json={"supported_models": []})

    monkeypatch.setattr(
        workloads.httpx,
        "AsyncClient",
        lambda **kw: original_client(**kw, transport=httpx.MockTransport(respond)),
    )
    store = Store(tmp_path)
    recorder = runner.Recorder(store, {"run_id": "load"}, tmp_path)
    config = BenchmarkConfig(
        name="load",
        suite="control-plane",
        model="test",
        clients=[4],
        warmup_steps=0,
        requests_per_client=5,
    )
    if status == 503:
        with pytest.raises(RuntimeError, match="20 control-plane requests failed"):
            asyncio.run(
                workloads.control_plane_suite(config, "https://test/api/v1", recorder)
            )
    else:
        asyncio.run(
            workloads.control_plane_suite(config, "https://test/api/v1", recorder)
        )
    samples = store.series(["load"], "control_plane.request_s")
    assert len(samples) == 20
    assert store.series(["load"], "control_plane.request_count")[0]["value"] == 20
    assert store.series(["load"], "control_plane.error_count")[0]["value"] == (
        20 if status == 503 else 0
    )
    assert store.series(["load"], "control_plane.p95_s")[0]["value"] > 0


def test_capture_to_ready_correlates_versions_replicas_and_out_of_order_reimports(
    tmp_path,
):
    store = Store(tmp_path / "source")
    store.register({"run_id": "sync"})

    def append(metric, stamp, version=1, worker="sampler-a"):
        store.append(
            "sync",
            {
                "time": stamp + 100,
                "labels": {
                    "spindle.model_id": "model",
                    "publish_version": version,
                    "spindle.worker": worker,
                },
                "metrics": {metric: stamp},
            },
        )

    append("weight_sync.ready_unix_s", 13)
    append("weight_sync.ready_unix_s", 15, worker="sampler-b")
    append("weight_sync.ready_unix_s", 30, version=2)
    append("weight_sync.ready_unix_s", 9, version=3)
    append("weight_sync.capture_started_unix_s", 10, worker="trainer")
    append("weight_sync.capture_started_unix_s", 12, version=3, worker="trainer")
    append("weight_sync.ready_unix_s", 13)
    rows = store.series(["sync"], "weight_sync.capture_to_ready_s")
    assert [row["value"] for row in rows] == [3, 5]
    assert (
        next(
            row
            for row in store.catalog(["sync"])
            if row["name"] == "weight_sync.capture_to_ready_s"
        )["samples"]
        == 2
    )
    target = Store(tmp_path / "restored")
    for _ in range(2):
        target.ingest(store.bundle("sync"))
    assert target.series(["sync"], "weight_sync.capture_to_ready_s") == rows


@pytest.mark.parametrize(
    "router,score", [(False, False), (True, False), (False, True), (True, True)]
)
def test_dapo_replay_payload_is_retained_in_training_data(router, score):
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/dapo-27b-4h200-12b200/client.py")
    )
    replay = {"prompt_tokens": 2, "temperature": 1.0}
    if router:
        replay["routed_experts"] = base64.b64encode(b"\x00" * 12).decode()
    if score:
        replay.update(
            sampling_mask_ids=[7, 8],
            sampling_mask_offsets=[0, 1, 2],
            sampling_logprobs=[-0.1, -0.2],
            sampling_mask_logprobs=[-0.1, -0.2],
        )
    datum = namespace["datum"](
        [1, 2],
        [7, 8],
        [-1.0, -2.0],
        1.0,
        2,
        replay if router or score else None,
        num_layers=1,
        experts_per_token=1,
    )
    inputs = datum.loss_fn_inputs
    assert ("routed_experts" in inputs) is router
    assert ("sampling_mask_logprobs" in inputs) is score
    if router:
        assert inputs["routed_experts"].shape == [3, 1, 1]
    if score:
        assert inputs["sampling_mask_offsets"].tolist() == [0, 0, 1, 2]
        assert inputs["logprobs"].tolist() == pytest.approx([0, -0.1, -0.2])
