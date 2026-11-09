"""Send usage to a hosted ingest service instead of a Volume, and the reference service.

Spindle runs in the customer's workspace, so it cannot write to a hosted
ClickHouse directly. `RpcUsageStore` implements `UsageWriter` and
`UsageReader` over HTTPS; the ingest service checks the caller's identity
token and scopes every write and read to the caller's workspace, so a row
cannot claim another workspace.

Routes, all under `/v1/training-api/usage` and authenticated with
`Authorization: Bearer <token>`:

- `POST /rows`             {"rows": [UsageRow, ...]}
- `POST /sessions`         {"session_id", "user_id", "labels", "created_at"}
- `POST /models`           {"training_model_id", "session_id"}
- `GET  /hourly`           ?deployment=&start=&end= -> {"usage": [HourlyUsage, ...]}
- `POST /sessions/lookup`  {"session_ids": [...]} -> {"sessions": {id: {"user_id", "labels"}}}
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from typing import Protocol

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from spindle.usage import (
    HourlyUsage,
    SessionUsage,
    UsageReader,
    UsageRow,
    UsageWriter,
)

PREFIX = "/v1/training-api/usage"
ROWS_PER_REQUEST = 5_000
IDENTITY_TOKEN_ENV = "MODAL_IDENTITY_TOKEN"


class RpcUsageStore:
    """`UsageWriter` and `UsageReader` backed by the usage ingest service."""

    def __init__(
        self,
        url: str,
        *,
        deployment: str,
        token: Callable[[], str] = lambda: os.environ[IDENTITY_TOKEN_ENV],
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.deployment = deployment
        self._token = token
        self._client = client or httpx.Client(base_url=url, timeout=timeout)

    def write(self, rows: Iterable[UsageRow]) -> None:
        rows = [asdict(row) for row in rows]
        for start in range(0, len(rows), ROWS_PER_REQUEST):
            self._call(
                "POST", "/rows", json={"rows": rows[start : start + ROWS_PER_REQUEST]}
            )

    def put_session(
        self, session_id: str, session: SessionUsage, created_at: float
    ) -> None:
        self._call(
            "POST",
            "/sessions",
            json={
                "session_id": session_id,
                "user_id": session.user_id,
                "labels": dict(session.labels),
                "created_at": created_at,
            },
        )

    def put_model(self, training_model_id: str, session_id: str) -> None:
        self._call(
            "POST",
            "/models",
            json={"training_model_id": training_model_id, "session_id": session_id},
        )

    def hourly_usage(self, start: float, end: float) -> list[HourlyUsage]:
        body = self._call(
            "GET",
            "/hourly",
            params={"deployment": self.deployment, "start": start, "end": end},
        )
        return [HourlyUsage(**usage) for usage in body["usage"]]

    def sessions(self, session_ids: Iterable[str]) -> dict[str, SessionUsage]:
        body = self._call(
            "POST", "/sessions/lookup", json={"session_ids": list(session_ids)}
        )
        return {
            session_id: SessionUsage(record["user_id"], record["labels"])
            for session_id, record in body["sessions"].items()
        }

    def _call(self, method: str, path: str, **kwargs) -> dict:
        response = self._client.request(
            method,
            PREFIX + path,
            headers={"authorization": f"Bearer {self._token()}"},
            **kwargs,
        )
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class Caller:
    """Who sent a request, from their verified identity token."""

    workspace_id: str
    environment_id: str


class UsageStore(UsageWriter, UsageReader, Protocol):
    """A writer and reader scoped to one caller's workspace and, for reads, one deployment."""


class _RowsBody(BaseModel):
    rows: list[dict]


class _SessionBody(BaseModel):
    session_id: str
    user_id: str | None = None
    labels: dict[str, str] = {}
    created_at: float


class _ModelBody(BaseModel):
    training_model_id: str
    session_id: str


class _LookupBody(BaseModel):
    session_ids: list[str]


def create_usage_ingest_app(
    store_for: Callable[[Caller, str | None], UsageStore],
    verify: Callable[[str], Caller],
) -> FastAPI:
    """The ingest service: verify the caller, then read and write only their workspace."""

    def caller(request: Request) -> Caller:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="missing identity token")
        try:
            return verify(token)
        except Exception as exc:
            raise HTTPException(
                status_code=401, detail="invalid identity token"
            ) from exc

    app = FastAPI()

    @app.post(PREFIX + "/rows")
    def write_rows(body: _RowsBody, who: Caller = Depends(caller)) -> dict:
        rows = [
            replace(
                UsageRow(**row),
                workspace_id=who.workspace_id,
                environment_id=who.environment_id,
            )
            for row in body.rows
        ]
        store_for(who, None).write(rows)
        return {"accepted": len(rows)}

    @app.post(PREFIX + "/sessions")
    def put_session(body: _SessionBody, who: Caller = Depends(caller)) -> dict:
        store_for(who, None).put_session(
            body.session_id, SessionUsage(body.user_id, body.labels), body.created_at
        )
        return {}

    @app.post(PREFIX + "/models")
    def put_model(body: _ModelBody, who: Caller = Depends(caller)) -> dict:
        store_for(who, None).put_model(body.training_model_id, body.session_id)
        return {}

    @app.get(PREFIX + "/hourly")
    def hourly(
        deployment: str, start: float, end: float, who: Caller = Depends(caller)
    ) -> dict:
        usage = store_for(who, deployment).hourly_usage(start, end)
        return {"usage": [asdict(item) for item in usage]}

    @app.post(PREFIX + "/sessions/lookup")
    def lookup(body: _LookupBody, who: Caller = Depends(caller)) -> dict:
        sessions = store_for(who, None).sessions(body.session_ids)
        return {
            "sessions": {
                session_id: {"user_id": session.user_id, "labels": dict(session.labels)}
                for session_id, session in sessions.items()
            }
        }

    return app


def modal_identity_verifier(
    issuer: str = "https://oidc.modal.com",
) -> Callable[[str], Caller]:
    """Verify Modal OIDC identity tokens against the issuer's published keys."""
    # PyJWT is only needed by the hosted ingest service, not by Spindle deployments.
    import jwt

    keys = jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json")

    def verify(token: str) -> Caller:
        claims = jwt.decode(
            token,
            keys.get_signing_key_from_jwt(token).key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False},
        )
        return Caller(claims["workspace_id"], claims["environment_name"])

    return verify


TABLES = (
    """CREATE TABLE IF NOT EXISTS training_api_token_metrics (
    workspace_id String,
    environment_id String,
    timestamp DateTime64(3, 'UTC'),
    model_id LowCardinality(String),
    request_uuid String,
    deployment String,
    app_name String,
    app_kind LowCardinality(String),
    container_id String,
    session_id Nullable(String),
    training_model_id Nullable(String),
    prompt_tokens UInt64,
    completion_tokens UInt64,
    cached_prompt_tokens UInt64,
    reasoning_tokens UInt64,
    train_tokens UInt64,
    forward_tokens UInt64,
    busy_seconds Float64,
    idle_seconds Float64,
    cold_start_seconds Float64,
    org_id String,
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (workspace_id, environment_id, deployment, timestamp, request_uuid)""",
    """CREATE TABLE IF NOT EXISTS training_api_sessions (
    workspace_id String,
    session_id String,
    user_id Nullable(String),
    labels Map(String, String),
    created_at DateTime64(3, 'UTC')
) ENGINE = ReplacingMergeTree() ORDER BY (workspace_id, session_id)""",
    """CREATE TABLE IF NOT EXISTS training_api_models (
    workspace_id String,
    training_model_id String,
    session_id String
) ENGINE = ReplacingMergeTree() ORDER BY (workspace_id, training_model_id)""",
)

_HOURLY = """
SELECT
    toUnixTimestamp(toStartOfHour(r.timestamp)) AS bucket_start,
    r.model_id AS model_id,
    if(isNull(r.session_id), nullIf(m.session_id, ''), r.session_id) AS session_id,
    sum(r.prompt_tokens) AS prompt_tokens,
    sum(r.cached_prompt_tokens) AS cached_prompt_tokens,
    sum(r.completion_tokens) AS completion_tokens,
    sum(r.train_tokens) AS train_tokens,
    sum(r.forward_tokens) AS forward_tokens
FROM training_api_token_metrics AS r FINAL
LEFT JOIN (SELECT * FROM training_api_models FINAL WHERE workspace_id = {workspace:String}) AS m
    ON m.training_model_id = r.training_model_id
WHERE r.workspace_id = {workspace:String}
    AND r.deployment = {deployment:String}
    AND r.timestamp >= toDateTime64({start:Float64}, 3, 'UTC')
    AND r.timestamp < toDateTime64({end:Float64}, 3, 'UTC')
GROUP BY bucket_start, model_id, session_id
HAVING prompt_tokens + completion_tokens + train_tokens + forward_tokens > 0
FORMAT JSONEachRow
"""

_SESSIONS = """
SELECT session_id, user_id, labels FROM training_api_sessions FINAL
WHERE workspace_id = {workspace:String} AND has({ids:Array(String)}, session_id)
FORMAT JSONEachRow
"""


class ClickHouseUsageStore:
    """Usage in ClickHouse for one workspace, through ClickHouse's HTTP interface.

    `execute(query, params, body)` runs one statement and returns its output;
    `clickhouse_http` builds one for a ClickHouse server.
    """

    def __init__(
        self,
        execute: Callable[[str, dict[str, str], str | None], str],
        caller: Caller,
        *,
        deployment: str | None = None,
    ) -> None:
        self._execute = execute
        self.caller = caller
        self.deployment = deployment

    def write(self, rows: Iterable[UsageRow]) -> None:
        body = "".join(json.dumps(asdict(row)) + "\n" for row in rows)
        if body:
            self._insert("training_api_token_metrics", body)

    def put_session(
        self, session_id: str, session: SessionUsage, created_at: float
    ) -> None:
        record = {
            "workspace_id": self.caller.workspace_id,
            "session_id": session_id,
            "user_id": session.user_id,
            "labels": dict(session.labels),
            "created_at": created_at,
        }
        self._insert("training_api_sessions", json.dumps(record) + "\n")

    def put_model(self, training_model_id: str, session_id: str) -> None:
        record = {
            "workspace_id": self.caller.workspace_id,
            "training_model_id": training_model_id,
            "session_id": session_id,
        }
        self._insert("training_api_models", json.dumps(record) + "\n")

    def hourly_usage(self, start: float, end: float) -> list[HourlyUsage]:
        params = {
            "workspace": self.caller.workspace_id,
            "deployment": self.deployment or "",
            "start": repr(start),
            "end": repr(end),
        }
        return [
            HourlyUsage(**json.loads(line))
            for line in self._execute(_HOURLY, params, None).splitlines()
            if line
        ]

    def sessions(self, session_ids: Iterable[str]) -> dict[str, SessionUsage]:
        params = {
            "workspace": self.caller.workspace_id,
            "ids": json.dumps(list(session_ids)).replace('"', "'"),
        }
        sessions = {}
        for line in self._execute(_SESSIONS, params, None).splitlines():
            if line:
                record = json.loads(line)
                sessions[record["session_id"]] = SessionUsage(
                    record["user_id"], record["labels"]
                )
        return sessions

    def _insert(self, table: str, body: str) -> None:
        self._execute(
            f"INSERT INTO {table} SETTINGS date_time_input_format = 'best_effort' "
            "FORMAT JSONEachRow",
            {},
            body,
        )


def clickhouse_http(
    url: str, *, user: str = "default", password: str = "", database: str = "default"
) -> Callable[[str, dict[str, str], str | None], str]:
    """An `execute` for `ClickHouseUsageStore` that talks to ClickHouse over HTTP."""
    client = httpx.Client(base_url=url, auth=(user, password), timeout=60.0)

    def execute(query: str, params: dict[str, str], body: str | None) -> str:
        query_params = {
            "database": database,
            **{f"param_{name}": value for name, value in params.items()},
        }
        if body is None:
            response = client.post("/", params=query_params, content=query)
        else:
            response = client.post(
                "/", params={**query_params, "query": query}, content=body
            )
        response.raise_for_status()
        return response.text

    return execute
