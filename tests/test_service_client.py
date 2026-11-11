import asyncio

import pytest
import tinker
from starlette.responses import JSONResponse
from tinker import types
from tinker.lib import queue_state_logger
from tinker.lib.api_future_impl import QueueState

import spindle
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from tests.control_plane.test_sdk_e2e import (
    API_KEY,
    BASE_MODEL,
    DEFINITION,
    DEFINITIONS,
)
from tests.support import TinkerStubExecutor, TinkerStubSampler, serve

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")


@pytest.fixture(autouse=True)
def restore_sdk_grace_period(monkeypatch):
    if hasattr(queue_state_logger, "PAUSE_GRACE_PERIOD"):
        monkeypatch.setattr(
            queue_state_logger,
            "PAUSE_GRACE_PERIOD",
            queue_state_logger.PAUSE_GRACE_PERIOD,
        )


@pytest.mark.parametrize(
    "auth_kind", ["env", "explicit", "default_headers", "default_headers_overrides"]
)
def test_real_sdk_auth_progress_and_training(auth_kind, monkeypatch, caplog):
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
            or (
                auth_kind.startswith("default_headers")
                and request.headers.get("x-custom") != "custom"
            )
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
        if auth_kind != "env":
            monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "unused-key")
            monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "unused-secret")
            kwargs.update(proxy_token_id="wk-test", proxy_token_secret="ws-test")
        if auth_kind.startswith("default_headers"):
            kwargs.update(
                proxy_token_id="explicit-key", proxy_token_secret="explicit-secret"
            )
            kwargs[auth_kind] = {
                "modal-key": "wk-test",
                "mODAL-sECRET": "ws-test",
                "X-Custom": "custom",
            }
        service = spindle.create_service_client(**kwargs)
        assert type(service) is tinker.ServiceClient
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
    progress = [
        r.message for r in caplog.records if r.name == "tinker.lib.queue_state_logger"
    ]
    assert any("Model creation" in m and "starting a trainer" in m for m in progress)
    assert any("Training" in m for m in progress)


def test_factory_enables_existing_sdk_observer_without_changing_throttle(
    monkeypatch, caplog
):
    if hasattr(queue_state_logger, "PAUSE_GRACE_PERIOD"):
        monkeypatch.setattr(queue_state_logger, "PAUSE_GRACE_PERIOD", 900)
    now = [1000.0]
    monkeypatch.setattr(queue_state_logger.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(queue_state_logger.time, "time", lambda: now[0])
    observer = queue_state_logger.QueueStateLogger("model-1", "Training")
    spindle.create_service_client()
    for second in (0, 1, 59, 60):
        now[0] = 1000.0 + second
        observer.on_queue_state_change(
            QueueState.PAUSED_CAPACITY, f"waiting for GPUs ({second}s so far)"
        )
    assert len(caplog.records) == 2
    assert "0s so far" in caplog.records[0].message
    assert "60s so far" in caplog.records[1].message


@pytest.mark.parametrize("explicit", [False, True])
def test_proxy_pair_validation_does_not_expose_credentials(monkeypatch, explicit):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)
    kwargs = {"proxy_token_secret": "private-value"} if explicit else {}
    if not explicit:
        monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "private-value")
    with pytest.raises(ValueError) as error:
        spindle.create_service_client(**kwargs)
    assert "private-value" not in str(error.value)
