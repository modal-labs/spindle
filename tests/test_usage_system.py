import contextlib
import math
import time
from datetime import UTC, datetime

import pytest
import tinker
from tinker import types

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine import Engine
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.engine.http import HttpEngineClient, create_engine_app
from spindle.providers.local import InMemoryKeyValueStore
from spindle.usage import DirectorySink, TrainerUsage, UsageMeter
from tests.support import SingleEnginePlatform, TinkerStubExecutor, serve
from tests.test_system import API_KEY, BASE_MODEL, DEFINITION, DEFINITIONS, ENGINE_TOKEN

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")


@pytest.fixture
def deployment(tmp_path):
    sink = DirectorySink(tmp_path)
    meter = UsageMeter(sink, deployment="dd-prod", container_id="trainer-1")
    with contextlib.ExitStack() as stack:
        backend_url = stack.enter_context(
            serve(create_backend_app(TinkerStubExecutor()))
        )
        usage = TrainerUsage(
            meter, app_name="dd-prod-trainer", base_model=BASE_MODEL, started_at=0
        )
        engine_url = stack.enter_context(
            serve(
                create_engine_app(
                    Engine(HttpBackendClient(backend_url), usage=usage),
                    token=ENGINE_TOKEN,
                )
            )
        )
        platform = SingleEnginePlatform(
            DEFINITION, HttpEngineClient(engine_url, token=ENGINE_TOKEN)
        )
        plane = ControlPlane(InMemoryKeyValueStore(), platform, usage=sink)
        app = create_control_plane_app(
            plane, DEFINITIONS, api_key=API_KEY, retrieve_window=5.0
        )
        yield stack.enter_context(serve(app)), meter


def test_stock_tinker_sdk_reads_training_usage_per_run(deployment) -> None:
    base_url, meter = deployment
    hour = math.floor(time.time() / 3600) * 3600
    service = tinker.ServiceClient(
        base_url=base_url,
        api_key=API_KEY,
        user_metadata={"run_id": "487852", "team": "ranking", "api_token": "x"},
    )
    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={"target_tokens": [2, 3, 4, 5], "weights": [1.0] * 4},
    )
    training.forward_backward([datum], "cross_entropy").result(timeout=30)
    training.forward_backward([datum, datum], "cross_entropy").result(timeout=30)
    meter.flush(include_open=True)

    usage = (
        service.create_rest_client()
        .get_billing_usage(
            datetime.fromtimestamp(hour, UTC),
            datetime.fromtimestamp(hour + 3600, UTC),
        )
        .result(timeout=30)
    )

    [event] = usage.data
    assert event.event_info == types.TrainingBillingEvent(token_count=12)
    assert event.base_model == BASE_MODEL
    assert usage.sessions[event.session_id].user_metadata == {
        "run_id": "487852",
        "team": "ranking",
    }
