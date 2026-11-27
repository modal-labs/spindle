from __future__ import annotations

import asyncio
import contextlib
import io
import math
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spindle.providers.modal.capture_store import (
    ModalDictCaptureStore,
    ModalVolumeCaptureStore,
    _CommitCoalescer,
    _ReloadGate,
)
from spindle.stability import CapturedSequence, CaptureStore, InMemoryCaptureStore


class _FakeModalDict:
    def __init__(self, name: str) -> None:
        self.name = name
        self.data: dict[str, object] = {}
        self.put = SimpleNamespace(aio=self._put)
        self.get = SimpleNamespace(aio=self._get)
        self.pop = SimpleNamespace(aio=self._pop)
        self.keys = SimpleNamespace(aio=self._keys)
        self.items = SimpleNamespace(aio=self._items_forbidden)

    async def _put(self, key: str, value: object) -> None:
        self.data[key] = value

    async def _get(self, key: str) -> object | None:
        return self.data.get(key)

    async def _pop(self, key: str, default: object = None) -> object | None:
        return self.data.pop(key, default)

    async def _keys(self) -> AsyncGenerator[str]:
        for key in list(self.data):
            yield key

    async def _items_forbidden(self) -> AsyncGenerator[tuple[str, object]]:
        raise AssertionError("expire_before must iterate keys(), not items()")
        yield ("", None)  # pragma: no cover


class _FakeModalVolume:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, bytes] = {}
        self.commit_count = 0
        self.reload_count = 0
        self.commit = SimpleNamespace(aio=self._commit)
        self.reload = SimpleNamespace(aio=self._reload)
        self.batch_upload = SimpleNamespace(aio=self._batch_upload)
        self.read_file = SimpleNamespace(aio=self._read_file)
        self.remove_file = SimpleNamespace(aio=self._remove_file)
        self.listdir = SimpleNamespace(aio=self._listdir)

    async def _commit(self) -> None:
        self.commit_count += 1

    async def _reload(self) -> None:
        self.reload_count += 1

    @contextlib.asynccontextmanager
    async def _batch_upload(self, *, force: bool = False) -> AsyncGenerator[object]:
        staged: dict[str, bytes] = {}

        class _Batch:
            def put_file(self, data: io.BytesIO, remote_path: str) -> None:
                staged["/" + remote_path.lstrip("/")] = data.read()

        yield _Batch()
        self.files.update(staged)

    async def _read_file(self, path: str) -> AsyncGenerator[bytes]:
        norm = "/" + path.lstrip("/")
        if norm not in self.files:
            raise FileNotFoundError(norm)
        yield self.files[norm]

    async def _remove_file(self, path: str, *, recursive: bool = False) -> None:
        norm = "/" + path.lstrip("/")
        if recursive:
            prefix = norm.rstrip("/") + "/"
            removed = [k for k in list(self.files) if k == norm or k.startswith(prefix)]
            if not removed:
                raise FileNotFoundError(norm)
            for k in removed:
                del self.files[k]
        elif norm in self.files:
            del self.files[norm]
        else:
            raise FileNotFoundError(norm)

    async def _listdir(
        self, path: str, *, recursive: bool = False
    ) -> list[SimpleNamespace]:
        norm = "/" + path.lstrip("/")
        prefix = norm.rstrip("/") + "/"
        matches = [k for k in sorted(self.files) if k.startswith(prefix)]
        if not matches:
            raise FileNotFoundError(norm)
        return [SimpleNamespace(path=k.lstrip("/")) for k in matches]


def _make_sequence(
    session_id: str = "sess-a",
    sequence_id: str = "req-1:0",
    *,
    created_at: float = 1000.0,
) -> CapturedSequence:
    prompt = np.array([1, 2, 3, 4], dtype=np.int64)
    response = np.array([10, 11, 12], dtype=np.int64)
    routes = np.arange(6 * 2 * 4, dtype=np.int32).reshape(6, 2, 4)
    mask_ids = np.array([10, 20, 11, 21, 12, 22], dtype=np.int64)
    mask_offsets = np.array([0, 2, 4, 6], dtype=np.int64)
    mask_logprobs = np.array([math.log(0.7), math.log(0.3)] * 3, dtype=np.float32)
    sampling_logprobs = np.array([math.log(0.7)] * 3, dtype=np.float32)
    draws = np.arange(3 * 8, dtype=np.int64).reshape(3, 8)
    return CapturedSequence(
        sequence_id=sequence_id,
        session_id=session_id,
        created_at=created_at,
        prompt_tokens=prompt,
        response_tokens=response,
        temperature=0.8,
        routed_experts=routes,
        sampling_mask_ids=mask_ids,
        sampling_mask_offsets=mask_offsets,
        sampling_mask_logprobs=mask_logprobs,
        sampling_logprobs=sampling_logprobs,
        sampler_draws=draws,
    )


def _build_store(kind: str, tmp_path: Path, now: list[float]) -> CaptureStore:
    if kind == "memory":
        return InMemoryCaptureStore(ttl_seconds=100.0, clock=lambda: now[0])
    if kind == "modal_dict":
        dicts: dict[str, _FakeModalDict] = {}

        def factory(name: str) -> _FakeModalDict:
            return dicts.setdefault(name, _FakeModalDict(name))

        async def delete_dict(
            name: str,
            *,
            allow_missing: bool = False,
            environment_name: str | None = None,
        ) -> None:
            dicts.pop(name, None)

        async def list_dicts(*, environment_name: str = "") -> list[_FakeModalDict]:
            return list(dicts.values())

        return ModalDictCaptureStore(
            app_id="ap-test",
            ttl_seconds=100.0,
            clock=lambda: now[0],
            dict_factory=factory,
            delete_dict=delete_dict,
            list_dicts=list_dicts,
        )
    volume = _FakeModalVolume(tmp_path)
    mode = "mount" if kind == "modal_volume_mount" else "api"
    return ModalVolumeCaptureStore(
        volume,  # type: ignore[arg-type]
        app_id="ap-test",
        mode=mode,
        mount_root=tmp_path,
        ttl_seconds=100.0,
        clock=lambda: now[0],
    )


@pytest.mark.parametrize(
    "kind",
    ["memory", "modal_dict", "modal_volume_mount", "modal_volume_api"],
)
def test_capture_store_roundtrip_isolation_ttl_and_cleanup(
    kind: str, tmp_path: Path
) -> None:
    now = [1000.0]
    store = _build_store(kind, tmp_path, now)

    async def scenario() -> None:
        seq1 = _make_sequence("sess-a", "req-1:0", created_at=1000.0)
        seq2 = _make_sequence("sess-a", "req-2:0", created_at=1080.0)
        seq_b = _make_sequence("sess-b", "req-1:0", created_at=1000.0)
        stats = await store.write_sequence(seq1)
        assert stats.total_chunk_bytes > 0
        await store.write_sequence(seq2)
        await store.write_sequence(seq_b)

        # Full read matches written data.
        got = await store.read_sequence("sess-a", "req-1:0")
        assert got is not None
        np.testing.assert_array_equal(got.routed_experts, seq1.routed_experts)  # type: ignore[arg-type]
        np.testing.assert_array_equal(got.sampler_draws, seq1.sampler_draws)  # type: ignore[arg-type]

        # Selective read returns requested fields only.
        draws_only = await store.read_sequence(
            "sess-a", "req-1:0", fields={"sampler_draws"}
        )
        assert draws_only is not None
        assert draws_only.routed_experts is None
        assert draws_only.sampling_mask_ids is None
        np.testing.assert_array_equal(draws_only.sampler_draws, seq1.sampler_draws)  # type: ignore[arg-type]

        # Cross-session isolation.
        assert await store.read_sequence("sess-other", "req-1:0") is None

        # TTL enforcement on read and expire_before sweep.
        now[0] = 1150.0
        assert await store.read_sequence("sess-a", "req-1:0") is None
        assert await store.read_sequence("sess-a", "req-2:0") is not None
        removed = await store.expire_before(1050.0)
        assert removed == 2  # seq1 in sess-a and seq_b in sess-b
        assert await store.read_sequence("sess-a", "req-2:0") is not None

        # Session deletion removes remaining captures for sess-a.
        await store.delete_session("sess-a")
        assert await store.read_sequence("sess-a", "req-2:0") is None

    asyncio.run(scenario())


def test_commit_coalescer_batches_concurrent_commits_safely() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        commits = 0

        async def do_commit() -> None:
            nonlocal commits
            commits += 1
            if commits == 1:
                started.set()
                await release.wait()

        coalescer = _CommitCoalescer(do_commit)
        first = asyncio.create_task(coalescer.commit())
        await asyncio.wait_for(started.wait(), 1)
        # Concurrent callers arriving mid-commit must wait for a second commit
        # that starts after their writes, and share that single second commit.
        others = [asyncio.create_task(coalescer.commit()) for _ in range(10)]
        await asyncio.sleep(0.01)
        assert not any(t.done() for t in others)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, *others), 2)
        assert commits == 2

    asyncio.run(scenario())


def test_reload_gate_never_reloads_with_active_readers_and_coalesces_misses() -> None:
    async def scenario() -> None:
        reader_entered = asyncio.Event()
        release_reader = asyncio.Event()
        active_at_reload: list[int] = []
        active = 0
        reloads = 0

        async def do_reload() -> None:
            nonlocal reloads
            reloads += 1
            active_at_reload.append(active)
            await asyncio.sleep(0.01)

        gate = _ReloadGate(do_reload)

        async def slow_reader() -> None:
            nonlocal active
            async with gate.reading():
                active += 1
                reader_entered.set()
                await release_reader.wait()
                active -= 1

        reader = asyncio.create_task(slow_reader())
        await asyncio.wait_for(reader_entered.wait(), 1)
        missers = [asyncio.create_task(gate.reload(0)) for _ in range(5)]
        await asyncio.sleep(0.01)
        assert reloads == 0
        release_reader.set()
        gens = await asyncio.wait_for(asyncio.gather(reader, *missers), 2)
        assert reloads == 1
        assert active_at_reload == [0]
        assert all(g == 1 for g in gens[1:])

    asyncio.run(scenario())


def test_reload_gate_cancelled_leader_and_cancelled_thread_read() -> None:
    async def scenario() -> None:
        thread_started = threading.Event()
        release_thread = threading.Event()
        thread_done = False
        active_at_reload: list[bool] = []
        reloads = 0

        async def do_reload() -> None:
            nonlocal reloads
            reloads += 1
            active_at_reload.append(thread_done)

        def blocking_read() -> str:
            nonlocal thread_done
            thread_started.set()
            release_thread.wait(timeout=2)
            thread_done = True
            return "ok"

        gate = _ReloadGate(do_reload)
        reader = asyncio.create_task(gate.run_read(blocking_read))
        await asyncio.to_thread(thread_started.wait, 2)

        # Two waiters try to reload; leader is cancelled before thread finishes.
        leader = asyncio.create_task(gate.reload(0))
        await asyncio.sleep(0.01)
        waiter = asyncio.create_task(gate.reload(0))
        await asyncio.sleep(0.01)
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader

        # Cancel the reader coroutine while the OS thread is still inside blocking_read().
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        await asyncio.sleep(0.02)
        # Reload must NOT have run yet because the OS thread still holds the lease.
        assert reloads == 0

        release_thread.set()
        gen = await asyncio.wait_for(waiter, 2)
        assert gen == 1
        assert reloads == 1
        assert active_at_reload == [True]

    asyncio.run(scenario())


def test_retry_attempt_id_prevents_chunk_mixing(tmp_path: Path) -> None:
    dicts: dict[str, _FakeModalDict] = {}
    store = ModalDictCaptureStore(
        app_id="ap-test",
        clock=lambda: 1005.0,
        dict_factory=lambda name: dicts.setdefault(name, _FakeModalDict(name)),
    )

    async def scenario() -> None:
        seq_a = _make_sequence("sess-r", "req-retry:0", created_at=1000.0)
        seq_b = CapturedSequence(
            sequence_id="req-retry:0",
            session_id="sess-r",
            created_at=1001.0,
            prompt_tokens=seq_a.prompt_tokens,
            response_tokens=seq_a.response_tokens,
            temperature=seq_a.temperature,
            routed_experts=seq_a.routed_experts + 10,  # type: ignore[operator]
        )
        await store.write_sequence(seq_a)
        await store.write_sequence(seq_b)
        got = await store.read_sequence("sess-r", "req-retry:0")
        assert got is not None
        np.testing.assert_array_equal(got.routed_experts, seq_b.routed_experts)  # type: ignore[arg-type]
        # Distinct attempt IDs exist in the Dict, so attempt A's chunks cannot overwrite attempt B's.
        raw_dict = next(iter(dicts.values())).data
        route_keys = [k for k in raw_dict if k.startswith("chk:req-retry:0:")]
        assert len(route_keys) >= 2

    asyncio.run(scenario())
