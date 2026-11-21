import asyncio
import json
from pathlib import Path

import httpx
import pytest
from stitch.types import VersionRef

from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app


def publish(root: Path, bulletin: SnapshotBulletin, run: str, version: int) -> None:
    source = root / f"{run}-{version}"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(f"{run}-{version}".encode())
    (source / "adapter_config.json").write_text("{}")
    bulletin.publish(VersionRef(run, version), source)


def payload(run: str, version: int, *, exact: bool = False) -> dict:
    return {
        "weight_run_id": run,
        "weight_version": {"exact_version" if exact else "min_version": version},
        "input_ids": [1, 2],
    }


def test_live_update_preserves_other_runs_and_reports_mixed_policy(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        bulletin = SnapshotBulletin(tmp_path / "bulletin")
        for run in ("a", "b"):
            publish(tmp_path, bulletin, run, 1)
        active, update_started, release = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        loaded: dict[str, str] = {}
        calls = []
        first = True

        async def upstream(request: httpx.Request) -> httpx.Response:
            nonlocal first
            body = json.loads(request.content)
            name = body.get("lora_name") or body.get("lora_path")
            if request.url.path == "/load_lora_adapter":
                loaded[name] = body["lora_path"]
                return httpx.Response(200, json={"success": True})
            if request.url.path == "/update_lora_adapter":
                calls.append(body)
                assert body["expected_version"] == "live:a"
                update_started.set()
                await release.wait()
                loaded[name] = body["lora_path"]
                return httpx.Response(200, json={"success": True})
            if name == "live:a" and first:
                first = False
                active.set()
                await release.wait()
                spans = [
                    {"version": "live:a", "start": 0, "end": 2},
                    {"version": "2", "start": 2, "end": 4},
                ]
            else:
                version = "2" if name == "live:a" and calls else "default"
                spans = [{"version": version, "start": 0, "end": 4}]
            return httpx.Response(200, json={"meta_info": {"weight_versions": spans}})

        app = create_app(
            bulletin,
            "http://engine",
            transport=httpx.MockTransport(upstream),
            update_lora_in_place=True,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            assert (
                await client.post("/generate", json=payload("b", 1))
            ).status_code == 200
            original = asyncio.create_task(
                client.post("/generate", json=payload("a", 1))
            )
            await asyncio.wait_for(active.wait(), 2)
            publish(tmp_path, bulletin, "a", 2)
            newer = asyncio.create_task(client.post("/generate", json=payload("a", 2)))
            await asyncio.wait_for(update_started.wait(), 2)
            b = await asyncio.wait_for(
                client.post("/generate", json=payload("b", 1)), 2
            )
            assert b.json()["meta_info"]["weight_version_end"] == 1
            release.set()
            old, new = await asyncio.gather(original, newer)
            assert old.json()["meta_info"]["weight_version_start"] == 1
            assert old.json()["meta_info"]["weight_version_end"] == 2
            assert new.json()["meta_info"]["weight_version_start"] == 2
            assert len(calls) == 1
            assert set(loaded) == {"live:a", "live:b"}
            stale = await client.post("/generate", json=payload("a", 1))
            assert stale.json()["meta_info"]["weight_version_start"] == 2
            pinned = await client.post("/generate", json=payload("a", 1, exact=True))
            assert pinned.json()["meta_info"]["weight_version_end"] == 1
            assert VersionRef("a", 1).identity in loaded
            assert len(calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("evict_on_update", [False, True])
def test_live_adapter_recovers_after_engine_eviction(
    tmp_path: Path, evict_on_update: bool
) -> None:
    async def scenario() -> None:
        bulletin = SnapshotBulletin(tmp_path / "bulletin")
        publish(tmp_path, bulletin, "a", 1)
        registered = False
        loads = 0

        async def upstream(request: httpx.Request) -> httpx.Response:
            nonlocal registered, loads
            if request.url.path == "/load_lora_adapter":
                registered = True
                loads += 1
                return httpx.Response(200, json={"success": True})
            if not registered:
                return httpx.Response(400, json={"error": "LoRA adapter is not loaded"})
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin,
            "http://engine",
            transport=httpx.MockTransport(upstream),
            update_lora_in_place=True,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            await client.post("/generate", json=payload("a", 1))
            registered = False
            version = 2 if evict_on_update else 1
            if evict_on_update:
                publish(tmp_path, bulletin, "a", version)
            result = await client.post("/generate", json=payload("a", version))
            assert result.status_code == 200
            assert result.json()["meta_info"]["weight_version_end"] == version
            assert loads == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("disconnect", [False, True])
def test_failed_or_disconnected_update_keeps_retryable_registration(
    tmp_path: Path, disconnect: bool
) -> None:
    async def scenario() -> None:
        bulletin = SnapshotBulletin(tmp_path / "bulletin")
        publish(tmp_path, bulletin, "a", 1)
        entered, release = asyncio.Event(), asyncio.Event()
        attempts = 0

        async def upstream(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if request.url.path == "/update_lora_adapter":
                attempts += 1
                entered.set()
                await release.wait()
                if not disconnect and attempts == 1:
                    return httpx.Response(400, json={"error": "preparation failed"})
            return httpx.Response(200, json={"meta_info": {}})

        app = create_app(
            bulletin,
            "http://engine",
            transport=httpx.MockTransport(upstream),
            update_lora_in_place=True,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
            ) as client,
        ):
            await client.post("/generate", json=payload("a", 1))
            publish(tmp_path, bulletin, "a", 2)
            call = asyncio.create_task(client.post("/generate", json=payload("a", 2)))
            await asyncio.wait_for(entered.wait(), 2)
            if disconnect:
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
            release.set()
            if not disconnect:
                assert (await call).status_code == 400
                assert app.state.live_adapters["a"].version == 1
            else:
                await asyncio.gather(*app.state.adapter_resolutions.values())
            result = await client.post("/generate", json=payload("a", 2))
            assert result.json()["meta_info"]["weight_version_end"] == 2
            assert app.state.live_adapters["a"].version == 2
            assert attempts == (1 if disconnect else 2)

    asyncio.run(scenario())
