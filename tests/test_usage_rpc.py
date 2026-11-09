import contextlib
import math
import time
from dataclasses import replace
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
from spindle.usage_rpc import Caller, RpcUsageStore, create_usage_ingest_app
from tests.support import SingleEnginePlatform, TinkerStubExecutor, serve
from tests.test_system import API_KEY, BASE_MODEL, DEFINITION, DEFINITIONS, ENGINE_TOKEN

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")
TOKENS = {"tok-dd": Caller("ws-dd", "main"), "tok-other": Caller("ws-other", "main")}


def verify(token: str) -> Caller:
    return TOKENS[token]


@pytest.fixture
def ingest(tmp_path):
    def store_for(caller: Caller, deployment: str | None) -> DirectorySink:
        return DirectorySink(tmp_path / caller.workspace_id)

    with serve(create_usage_ingest_app(store_for, verify)) as url:
        yield url, tmp_path


def rpc(url: str, token: str) -> RpcUsageStore:
    return RpcUsageStore(url, deployment="dd-prod", token=lambda: token)


@pytest.fixture
def deployment(ingest):
    url, root = ingest
    store = rpc(url, "tok-dd")
    meter = UsageMeter(store, deployment="dd-prod", workspace_id="ws-spoofed")
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
            usage=store,
            usage_label_keys=("run_id", "team_id"),
        )
        app = create_control_plane_app(
            plane,
            DEFINITIONS,
            api_key=API_KEY,
            user_keys={"tml-alice": "alice"},
            retrieve_window=5.0,
            usage_reader=store,
        )
        yield stack.enter_context(serve(app)), meter, url, root


def test_stock_tinker_sdk_reads_usage_through_the_ingest_service(deployment) -> None:
    base_url, meter, ingest_url, root = deployment
    hour = math.floor(time.time() / 3600) * 3600
    service = tinker.ServiceClient(
        base_url=base_url,
        api_key="tml-alice",
        user_metadata={"run_id": "487852", "team_id": "t-42"},
    )
    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=8)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={"target_tokens": [2, 3, 4, 5], "weights": [1.0] * 4},
    )
    training.forward_backward([datum, datum], "cross_entropy").result(timeout=30)
    meter.flush(include_open=True)

    usage = (
        service.create_rest_client()
        .get_billing_usage(
            datetime.fromtimestamp(hour, UTC), datetime.fromtimestamp(hour + 3600, UTC)
        )
        .result(timeout=30)
    )

    [event] = usage.data
    assert event.event_info == types.TrainingBillingEvent(token_count=8)
    assert event.user_id == "alice"
    assert usage.sessions[event.session_id].user_metadata == {
        "run_id": "487852",
        "team_id": "t-42",
    }
    # The service, not the meter, decides the workspace.
    stored = DirectorySink(root / "ws-dd").read(hour, hour + 3600)
    assert stored and {row.workspace_id for row in stored} == {"ws-dd"}
    # Another workspace's token sees none of it.
    assert rpc(ingest_url, "tok-other").hourly_usage(hour, hour + 3600) == []


@pytest.mark.parametrize("authorization", [None, "Bearer tok-unknown", "tok-dd"])
def test_ingest_service_rejects_callers_it_cannot_verify(ingest, authorization) -> None:
    url, _ = ingest
    headers = {"authorization": authorization} if authorization else {}
    response = httpx.post(
        f"{url}/v1/training-api/usage/models",
        json={"training_model_id": "m", "session_id": "s"},
        headers=headers,
    )
    assert response.status_code == 401


def test_clickhouse_store_answers_the_reader_queries() -> None:
    chdb = pytest.importorskip("chdb", reason="needs an embedded ClickHouse (chdb)")
    from spindle.usage import SessionUsage, UsageRow
    from spindle.usage_rpc import TABLES, ClickHouseUsageStore

    session = chdb.session.Session()

    def execute(query: str, params: dict[str, str], body: str | None) -> str:
        for name, value in params.items():
            kind = {"start": "Float64", "end": "Float64", "ids": "Array(String)"}
            literal = value if kind.get(name) else "'" + value.replace("'", "\\'") + "'"
            query = query.replace(
                "{" + f"{name}:{kind.get(name, 'String')}" + "}", literal
            )
        if body is not None:
            query = query.replace("FORMAT JSONEachRow", "FORMAT JSONEachRow\n" + body)
        return str(session.query(query))

    for table in TABLES:
        session.query(table)
    store = ClickHouseUsageStore(execute, Caller("ws-dd", "main"), deployment="dd-prod")
    hour = 1_793_973_600.0
    base = UsageRow(
        workspace_id="ws-dd",
        environment_id="main",
        timestamp=hour + 60,
        model_id=BASE_MODEL,
        request_uuid="r1",
        deployment="dd-prod",
        app_name="trainer",
        app_kind="trainer",
        container_id="c1",
    )
    store.write(
        [
            replace(base, training_model_id="model-a", train_tokens=100),
            replace(base, training_model_id="model-a", train_tokens=100),  # retried
            replace(
                base,
                request_uuid="r2",
                session_id="s1",
                prompt_tokens=50,
                cached_prompt_tokens=20,
                completion_tokens=5,
            ),
        ]
    )
    store.put_model("model-a", "s1")
    store.put_session("s1", SessionUsage("alice", {"run_id": "A"}), hour)

    [usage] = store.hourly_usage(hour, hour + 3600)
    assert (usage.session_id, usage.train_tokens, usage.prompt_tokens) == (
        "s1",
        100,
        50,
    )
    assert store.sessions(["s1"]) == {"s1": SessionUsage("alice", {"run_id": "A"})}
