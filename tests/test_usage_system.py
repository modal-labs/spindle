import asyncio
import contextlib
import math
import time
from datetime import UTC, datetime

import httpx
import pytest
import tinker
from tinker import types

from spindle.control_plane import (
    ControlPlane,
    create_control_plane_app,
    parse_user_keys,
)
from spindle.control_plane.usage import SessionUsageWriter
from spindle.control_plane.keys import session_key
from spindle.engine import Engine
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.engine.http import HttpEngineClient, create_engine_app
from spindle.providers.local import InMemoryKeyValueStore
from spindle.usage import DirectorySink, SessionUsage, TrainerUsage, UsageMeter
from tests.support import SingleEnginePlatform, TinkerStubExecutor, serve
from tests.test_system import API_KEY, BASE_MODEL, DEFINITION, DEFINITIONS, ENGINE_TOKEN

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")
ALICE_KEY = "tml-alice"
DATUM = types.Datum(
    model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
    loss_fn_inputs={"target_tokens": [2, 3, 4, 5], "weights": [1.0] * 4},
)


@pytest.fixture
def deployment(tmp_path):
    sink = DirectorySink(tmp_path)
    meter = UsageMeter(sink, container_id="trainer-1")
    with contextlib.ExitStack() as stack:
        backend_url = stack.enter_context(
            serve(create_backend_app(TinkerStubExecutor()))
        )
        usage = TrainerUsage(
            meter, app_name="shared-trainer", base_model=BASE_MODEL, started_at=0
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
            usage=SessionUsageWriter(sink, ("run_id", "team_id")),
        )
        app = create_control_plane_app(
            plane,
            DEFINITIONS,
            api_key=API_KEY,
            user_keys={ALICE_KEY: "alice"},
            retrieve_window=5.0,
            usage_sink=sink,
        )
        yield stack.enter_context(serve(app)), meter, plane, sink


def wait_for_usage_writes(plane: ControlPlane) -> None:
    plane.usage._writer.submit(lambda: None).result(timeout=30)


# A lost model file covers models created before usage was recorded.
@pytest.mark.parametrize("model_file_lost", [False, True])
def test_stock_tinker_sdk_training_usage_is_stored_per_session(
    deployment, model_file_lost
) -> None:
    base_url, meter, plane, sink = deployment
    hour = math.floor(time.time() / 3600) * 3600
    service = tinker.ServiceClient(
        base_url=base_url,
        api_key=API_KEY,
        # Only allowlisted labels are kept.
        user_metadata={"run_id": "487852", "team_id": "t-42", "api_token": "x"},
    )
    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    if model_file_lost:
        model_file = sink.root / "models" / f"{training.model_id}.json"
        deadline = time.monotonic() + 30
        while not model_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        model_file.unlink()
        # As if the model was created by a frontend that didn't record usage.
        plane.usage = SessionUsageWriter(sink, ("run_id", "team_id"))
    training.forward_backward([DATUM], "cross_entropy").result(timeout=30)
    training.forward_backward([DATUM, DATUM], "cross_entropy").result(timeout=30)
    meter.flush(include_open=True)
    wait_for_usage_writes(plane)

    rows = sink.read(hour, hour + 7200)

    [session_id] = {row.session_id for row in rows if row.train_tokens}
    assert sum(row.train_tokens for row in rows) == 12
    assert {row.model_id for row in rows if row.train_tokens} == {BASE_MODEL}
    assert sink.sessions([session_id]) == {
        session_id: SessionUsage(None, {"run_id": "487852", "team_id": "t-42"})
    }


def test_a_failed_model_file_write_is_retried_on_the_next_training_op(
    deployment, monkeypatch
) -> None:
    base_url, meter, plane, sink = deployment
    put_model, failures = sink.put_model, [2]

    def flaky_put_model(*args) -> None:
        if failures[0]:
            failures[0] -= 1
            raise OSError("volume commit failed")
        put_model(*args)

    monkeypatch.setattr(sink, "put_model", flaky_put_model)
    hour = math.floor(time.time() / 3600) * 3600
    client = tinker.ServiceClient(
        base_url=base_url, api_key=API_KEY, user_metadata={"run_id": "487852"}
    )
    training = client.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    for _ in range(2):
        training.forward_backward([DATUM], "cross_entropy").result(timeout=30)
        wait_for_usage_writes(plane)
    meter.flush(include_open=True)

    rows = sink.read(hour, hour + 7200)

    assert failures == [0]
    [session_id] = {row.session_id for row in rows if row.train_tokens}
    assert sink.sessions([session_id]) == {
        session_id: SessionUsage(None, {"run_id": "487852"})
    }


def test_labels_lost_at_session_creation_are_written_before_the_session_is_swept(
    deployment, monkeypatch
) -> None:
    _, _, plane, sink = deployment
    put_session, failures = sink.put_session, [2]

    def flaky_put_session(*args) -> None:
        if failures[0]:
            failures[0] -= 1
            raise OSError("volume commit failed")
        put_session(*args)

    monkeypatch.setattr(sink, "put_session", flaky_put_session)

    async def create_and_sweep() -> tuple[str, dict]:
        session = await plane.create_session(
            user_metadata={"run_id": "487852"}, user_id="u-1"
        )
        wait_for_usage_writes(plane)
        # The sweep's write fails too, so the session is kept for the next sweep.
        await plane.sweep_idle_sessions(0)
        kept = sink.sessions([session.session_id])
        await plane.sweep_idle_sessions(0)
        return session.session_id, kept

    session_id, kept = asyncio.run(create_and_sweep())

    assert kept == {}
    assert failures == [0]
    assert sink.sessions([session_id]) == {
        session_id: SessionUsage("u-1", {"run_id": "487852"})
    }
    assert asyncio.run(plane.kv.get(session_key(session_id))) is None


def test_stock_tinker_sdk_reads_training_usage_per_run(deployment) -> None:
    base_url, meter, plane, _ = deployment
    hour = math.floor(time.time() / 3600) * 3600
    service = tinker.ServiceClient(
        base_url=base_url,
        api_key=ALICE_KEY,
        # The user comes from the API key, not user_metadata.
        user_metadata={"run_id": "487852", "team_id": "t-42", "user": "mallory"},
    )
    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    training.forward_backward([DATUM, DATUM, DATUM], "cross_entropy").result(timeout=30)
    meter.flush(include_open=True)
    wait_for_usage_writes(plane)

    usage = (
        service.create_rest_client()
        .get_billing_usage(
            datetime.fromtimestamp(hour, UTC),
            datetime.fromtimestamp(hour + 7200, UTC),
        )
        .result(timeout=30)
    )

    [event] = usage.data
    assert event.event_info == types.TrainingBillingEvent(token_count=12)
    assert (event.base_model, event.user_id) == (BASE_MODEL, "alice")
    assert usage.sessions[event.session_id].user_metadata == {
        "run_id": "487852",
        "team_id": "t-42",
    }


def test_a_user_key_can_neither_use_nor_read_another_keys_session(deployment) -> None:
    base_url, meter, plane, sink = deployment
    hour = math.floor(time.time() / 3600) * 3600
    shared = tinker.ServiceClient(
        base_url=base_url, api_key=API_KEY, user_metadata={"run_id": "shared"}
    )
    training = shared.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    training.forward_backward([DATUM], "cross_entropy").result(timeout=30)
    meter.flush(include_open=True)
    wait_for_usage_writes(plane)
    [session_id] = {r.session_id for r in sink.read(hour, hour + 3600) if r.session_id}

    created = httpx.post(
        f"{base_url}/api/v1/create_model",
        headers={"x-api-key": ALICE_KEY},
        json={"session_id": session_id, "model_seq_id": 9, "base_model": BASE_MODEL},
    )
    usage = httpx.get(
        f"{base_url}/api/v1/billing/usage/events",
        headers={"x-api-key": ALICE_KEY},
        params={
            "starting_on": datetime.fromtimestamp(hour, UTC).isoformat(),
            "ending_before": datetime.fromtimestamp(hour + 3600, UTC).isoformat(),
        },
    )

    assert created.status_code == 403
    assert usage.json() == {"data": [], "sessions": {}}


def test_a_user_key_can_finish_its_session_after_it_closed(deployment) -> None:
    base_url, *_ = deployment
    headers = {"x-api-key": ALICE_KEY}
    session_id = httpx.post(
        f"{base_url}/api/v1/create_session", headers=headers, json={}
    ).json()["session_id"]
    finish = {"reason": {"type": "success"}, "detail": None}
    url = f"{base_url}/api/v1/sessions/{session_id}/finish"

    statuses = [
        httpx.post(url, headers=headers, json=finish).status_code for _ in range(2)
    ]

    assert statuses == [200, 200]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("not json", "JSON object of API key to user name"),
        ('["tml-a"]', "JSON object of API key to user name"),
        ('{"": "alice"}', "JSON object of API key to user name"),
        ('{"tml-admin": "alice"}', "must not include TINKER_API_KEY"),
    ],
)
def test_malformed_user_keys_name_the_setting(value, error) -> None:
    with pytest.raises(ValueError, match=error):
        parse_user_keys(value, "tml-admin")
