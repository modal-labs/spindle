import contextlib
import math
import time
from datetime import UTC, datetime

import httpx
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
ALICE_KEY = "tml-alice"


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
        plane = ControlPlane(
            InMemoryKeyValueStore(),
            platform,
            usage=sink,
            usage_label_keys=("run_id", "team_id"),
        )
        app = create_control_plane_app(
            plane,
            DEFINITIONS,
            api_key=API_KEY,
            user_keys={ALICE_KEY: "alice"},
            retrieve_window=5.0,
        )
        yield stack.enter_context(serve(app)), meter


@pytest.mark.parametrize(
    ("api_key", "user_id"), [(ALICE_KEY, "alice"), (API_KEY, None)]
)
def test_stock_tinker_sdk_reads_training_usage_per_run(
    deployment, api_key, user_id
) -> None:
    base_url, meter = deployment
    hour = math.floor(time.time() / 3600) * 3600
    service = tinker.ServiceClient(
        base_url=base_url,
        api_key=api_key,
        # Only allowlisted labels are kept; the user comes from the API key.
        user_metadata={"run_id": "487852", "team_id": "t-42", "user": "mallory"},
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
    assert (event.base_model, event.user_id) == (BASE_MODEL, user_id)
    assert usage.sessions[event.session_id].user_metadata == {
        "run_id": "487852",
        "team_id": "t-42",
    }


def test_unknown_api_keys_are_rejected(deployment) -> None:
    base_url, _ = deployment
    response = httpx.get(
        f"{base_url}/api/v1/billing/usage/events",
        params={
            "starting_on": "2026-11-06T14:00:00Z",
            "ending_before": "2026-11-06T15:00:00Z",
        },
        headers={"x-api-key": "tml-unknown"},
    )
    assert response.status_code == 401
