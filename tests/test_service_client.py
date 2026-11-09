import asyncio
import inspect

import pytest
import tinker
from starlette.responses import JSONResponse
from tinker import types
from tinker.lib.api_future_impl import QueueState

import spindle
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from spindle.service_client import _Progress
from tests.control_plane.test_sdk_e2e import (
    API_KEY,
    BASE_MODEL,
    DEFINITION,
    DEFINITIONS,
)
from tests.support import TinkerStubExecutor, TinkerStubSampler, serve

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")


@pytest.mark.parametrize("client_kind", ["spindle", "quiet", "stock"])
def test_real_sdk_auth_progress_and_training(client_kind, monkeypatch, caplog):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform(DEFINITION, TinkerStubExecutor),
        sampling_tasks=LocalSamplingTaskPlatform(TinkerStubSampler()),
    )
    app = create_control_plane_app(plane, DEFINITIONS, api_key=API_KEY)
    pending = set()

    @app.middleware("http")
    async def proxy_and_progress(request, call_next):
        if (
            request.headers.get("modal-key") != "wk-test"
            or request.headers.get("modal-secret") != "ws-test"
        ):
            return JSONResponse(
                status_code=403, content={"error": "proxy auth missing"}
            )
        if request.url.path.endswith("/retrieve_future"):
            body = await request.json()
            request_id = body["request_id"]
            if request_id not in pending:
                pending.add(request_id)
                return JSONResponse(
                    status_code=408,
                    content={
                        "type": "try_again",
                        "request_id": request_id,
                        "queue_state": "starting",
                        "queue_state_reason": "starting a trainer (31s so far)",
                    },
                )
        return await call_next(request)

    with serve(app) as url:
        kwargs = dict(base_url=url, api_key=API_KEY)
        if client_kind == "stock":
            header_arg = (
                "default_headers_overrides"
                if "default_headers_overrides"
                in inspect.signature(tinker.ServiceClient.__init__).parameters
                else "default_headers"
            )
            service = tinker.ServiceClient(
                **kwargs,
                **{header_arg: {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"}},
            )
        else:
            service = spindle.ServiceClient(**kwargs, progress=client_kind == "spindle")
        assert service.get_server_capabilities().supported_models
        training = service.create_lora_training_client(base_model=BASE_MODEL, rank=8)
        if not isinstance(training, tinker.TrainingClient) or hasattr(
            training, "_awaited_client"
        ):
            training = training.result(timeout=30)
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2]),
            loss_fn_inputs={"target_tokens": [2, 3], "weights": [1.0, 1.0]},
        )

        async def train():
            return await training.forward_backward([datum], "cross_entropy")

        assert asyncio.run(train()).metrics["loss:sum"] == 1.25
        assert training.optim_step(types.AdamParams(learning_rate=1e-4)).result(
            timeout=30
        ).metrics == {"lr": 1e-4}
        sampling = service.create_sampling_client(base_model=BASE_MODEL)
        if hasattr(sampling, "_awaited_client"):
            sampling = sampling.result(timeout=30)
        sampled = sampling.sample(
            prompt=types.ModelInput.from_ints([1]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result(timeout=30)
        assert list(sampled.sequences[0].tokens) == [0, 1]
        if hasattr(service, "close"):
            service.close(status="success")
    progress = [r.message for r in caplog.records if r.name == "spindle.service_client"]
    assert bool(progress) == (client_kind == "spindle")
    if progress:
        assert any("CreateModel" in m and "starting a trainer" in m for m in progress)
        assert any("ForwardBackward" in m for m in progress)


@pytest.mark.parametrize(
    "reason",
    [
        "starting a trainer ({elapsed} so far)",
        "all trainer slots in use (2nd in line, {elapsed})",
    ],
)
def test_progress_stage_changes_and_heartbeat(monkeypatch, caplog, reason):
    now = [0.0]
    monkeypatch.setattr("spindle.service_client.time.monotonic", lambda: now[0])
    progress = _Progress("request-1", "CreateModel", 60)
    for second in (0, 1, 59, 60):
        now[0] = second
        progress.on_queue_state_change(
            QueueState.UNKNOWN, reason.format(elapsed=f"{second}s")
        )
    progress.on_queue_state_change(QueueState.PAUSED_CAPACITY, "waiting for GPUs")
    assert len(caplog.records) == 3
    assert "60s" in caplog.records[1].message
    assert "waiting for GPUs" in caplog.records[2].message


@pytest.mark.parametrize("explicit", [False, True])
def test_proxy_pair_validation_does_not_expose_credentials(monkeypatch, explicit):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)
    kwargs = {"proxy_token_secret": "private-value"} if explicit else {}
    if not explicit:
        monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "private-value")
    with pytest.raises(ValueError) as error:
        spindle.ServiceClient(**kwargs)
    assert "private-value" not in str(error.value)


def test_explicit_auth_and_case_insensitive_header_override(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "env-key")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "env-secret")
    captured = {}

    def init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tinker.ServiceClient, "__init__", init)
    spindle.ServiceClient(
        proxy_token_id="explicit-key",
        proxy_token_secret="explicit-secret",
        default_headers={"modal-key": "override-key", "X-Custom": "custom"},
    )
    assert captured["default_headers"] == {
        "modal-key": "override-key",
        "modal-secret": "explicit-secret",
        "x-custom": "custom",
    }
