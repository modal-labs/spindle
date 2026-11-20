"""Replica admission is shared across independent dispatchers and retried deliveries."""

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app


@pytest.mark.parametrize("finish", ["success", "error", "cancel"])
def test_replica_rejects_excess_work_and_releases_slots(
    tmp_path: Path, finish: str
) -> None:
    async def scenario() -> None:
        arrived = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def engine(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            if request.url.path != "/generate":
                return httpx.Response(200, json={})
            calls += 1
            arrived.set()
            await release.wait()
            return httpx.Response(500 if finish == "error" else 200, json={})

        app = create_app(
            SnapshotBulletin(tmp_path),
            "http://engine",
            max_running_requests=1,
            transport=httpx.MockTransport(engine),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://replica"
            ) as first,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://replica"
            ) as second,
        ):
            payload = {"rid": "original", "input_ids": [1]}
            running = asyncio.create_task(first.post("/generate", json=payload))
            await asyncio.wait_for(arrived.wait(), 1)
            duplicate = asyncio.create_task(second.post("/generate", json=payload))
            rejections = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        second.post(
                            "/generate", json={"rid": f"excess-{i}", "input_ids": [2]}
                        )
                        for i in range(20)
                    ]
                ),
                2,
            )
            assert all(r.status_code == 503 for r in rejections)
            assert all(r.headers["x-spindle-replica"] for r in rejections)
            assert calls == 1
            assert (await second.get("/health")).status_code == 200
            if finish == "cancel":
                running.cancel()
                duplicate.cancel()
                await asyncio.gather(running, duplicate, return_exceptions=True)
                async with asyncio.timeout(3):
                    while app.state.generation_slots:
                        await asyncio.sleep(0.01)
            release.set()
            if finish != "cancel":
                results = await asyncio.gather(running, duplicate)
                assert all(
                    r.status_code == (500 if finish == "error" else 200)
                    for r in results
                )
            # Rejected IDs are not cached, and freed slots admit new work.
            retried = await second.post(
                "/generate", json={"rid": "excess-0", "input_ids": [2]}
            )
            assert retried.status_code == (500 if finish == "error" else 200)
            assert calls == 2
            assert app.state.generation_slots == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"input_ids": [[1], [2]]},
        {"rid": ["a", "b"], "input_ids": [[1], [2]]},
        {"input_ids": [1], "sampling_params": {"n": 2}},
        {"text": ["one", "two"]},
    ],
)
def test_batch_and_parallel_samples_reserve_all_slots(
    tmp_path: Path, payload: dict
) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def engine(request: httpx.Request) -> httpx.Response:
            entered.set()
            await release.wait()
            return httpx.Response(200, json={})

        app = create_app(
            SnapshotBulletin(tmp_path),
            "http://engine",
            max_running_requests=2,
            transport=httpx.MockTransport(engine),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://replica"
            ) as client,
        ):
            running = asyncio.create_task(client.post("/generate", json=payload))
            await asyncio.wait_for(entered.wait(), 1)
            assert (
                await client.post("/generate", json={"input_ids": [3]})
            ).status_code == 503
            release.set()
            assert (await running).status_code == 200
            assert app.state.generation_slots == 0

    asyncio.run(scenario())


def test_batch_larger_than_replica_capacity_is_not_retried(tmp_path: Path) -> None:
    async def engine(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an oversized batch must not reach SGLang")

    app = create_app(
        SnapshotBulletin(tmp_path),
        "http://engine",
        max_running_requests=1,
        transport=httpx.MockTransport(engine),
    )
    with TestClient(app) as client:
        response = client.post("/generate", json={"input_ids": [[1], [2]]})
        assert response.status_code == 400
        assert "split" in response.json()["error"]
        assert client.get("/routing-state").json()["active"] == 0
