"""Modal-backed stability capture stores: per-session Dict and v2 Volume."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import modal

from spindle.stability.codec import (
    DEFAULT_CHUNK_BYTES,
    CapturedSequence,
    Compression,
    needed_chunk_keys,
    pack_capture,
    unpack_capture,
)
from spindle.stability.config import DEFAULT_CAPTURE_TTL_SECONDS
from spindle.stability.store import CaptureStore, WriteStats

from .kv import current_app_id

CAPTURE_RETRY_DELAYS_S = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_CAPTURE_VOLUME_NAME = "spindle-captures"
DEFAULT_CAPTURE_MOUNT_ROOT = "/captures"
DEFAULT_DICT_CONCURRENCY = 16
DEFAULT_DICT_CACHE_SIZE = 256
DEFAULT_VOLUME_API_CONCURRENCY = 16

VolumeMode = Literal["mount", "api"]


async def _retry_transient[T](call: Callable[[], Awaitable[T]]) -> T:
    for delay in (*CAPTURE_RETRY_DELAYS_S, None):
        try:
            return await call()
        except (
            modal.exception.ResourceExhaustedError,
            modal.exception.InternalError,
            modal.exception.ServiceError,
        ):
            if delay is None:
                raise
            await asyncio.sleep(delay * (1.0 + random.random()))
    raise AssertionError("unreachable")


def _short_digest(value: str) -> str:
    if not value:
        raise ValueError("identifier must be non-empty")
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def session_capture_dict_name(session_id: str, app_id: str) -> str:
    return f"{app_id}-captures-{_short_digest(session_id)}"


class ModalDictCaptureStore:
    """One Modal Dict per session; chunks written concurrently, header written last."""

    def __init__(
        self,
        app_id: str | None = None,
        *,
        environment_name: str | None = None,
        ttl_seconds: float = DEFAULT_CAPTURE_TTL_SECONDS,
        max_concurrency: int = DEFAULT_DICT_CONCURRENCY,
        max_cached_dicts: int = DEFAULT_DICT_CACHE_SIZE,
        clock: Callable[[], float] = time.time,
        dict_factory: Callable[[str], Any] | None = None,
        delete_dict: Callable[..., Awaitable[None]] | None = None,
        list_dicts: Callable[..., Awaitable[list[Any]]] | None = None,
    ) -> None:
        self.app_id = (
            app_id or os.environ.get("SPINDLE_FRONTEND_APP_ID") or current_app_id()
        )
        self.environment_name = environment_name
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._max_cached_dicts = max(1, max_cached_dicts)
        self._dicts: OrderedDict[str, Any] = OrderedDict()
        self._dict_factory = dict_factory or self._default_dict
        self._delete_dict = delete_dict or modal.Dict.objects.delete.aio
        self._list_dicts = list_dicts or modal.Dict.objects.list.aio

    def _default_dict(self, name: str) -> modal.Dict:
        return modal.Dict.from_name(
            name,
            environment_name=self.environment_name,
            create_if_missing=True,
        )

    def _for_session(self, session_id: str) -> modal.Dict:
        handle = self._dicts.get(session_id)
        if handle is not None:
            self._dicts.move_to_end(session_id)
            return handle
        handle = self._dict_factory(session_capture_dict_name(session_id, self.app_id))
        self._dicts[session_id] = handle
        while len(self._dicts) > self._max_cached_dicts:
            self._dicts.popitem(last=False)
        return handle

    async def _put(self, store: modal.Dict, key: str, value: object) -> None:
        async with self._sem:
            await _retry_transient(lambda: store.put.aio(key, value))

    async def _get(self, store: modal.Dict, key: str) -> object | None:
        async with self._sem:
            return await _retry_transient(lambda: store.get.aio(key))

    async def _pop(self, store: modal.Dict, key: str) -> object | None:
        async with self._sem:
            return await _retry_transient(lambda: store.pop.aio(key, None))

    @staticmethod
    def _chunk_key(sequence_id: str, attempt_id: str, key: str) -> str:
        return (
            f"chk:{sequence_id}:{attempt_id}:{key}"
            if attempt_id
            else f"chk:{sequence_id}:{key}"
        )

    async def write_sequence(
        self,
        sequence: CapturedSequence,
        *,
        target_chunk_bytes: int | None = DEFAULT_CHUNK_BYTES,
        compression: Compression = "none",
    ) -> WriteStats:
        t0 = time.perf_counter()
        packed = await asyncio.to_thread(
            pack_capture,
            sequence,
            target_chunk_bytes=target_chunk_bytes,
            compression=compression,
        )
        pack_s = time.perf_counter() - t0
        attempt_id = str(packed.header["attempt_id"])
        t1 = time.perf_counter()
        store = self._for_session(sequence.session_id)
        try:
            if packed.chunks:
                await asyncio.gather(
                    *(
                        self._put(
                            store,
                            self._chunk_key(sequence.sequence_id, attempt_id, key),
                            data,
                        )
                        for key, data in packed.chunks
                    )
                )
            # Header is written last and names attempt_id so retries never mix chunks.
            await self._put(store, f"hdr:{sequence.sequence_id}", packed.header)
        except modal.exception.NotFoundError:
            self._dicts.pop(sequence.session_id, None)
            raise
        write_s = time.perf_counter() - t1
        return WriteStats(
            num_chunks=len(packed.chunks),
            total_chunk_bytes=packed.total_chunk_bytes,
            pack_seconds=pack_s,
            write_seconds=write_s,
        )

    async def read_sequence(
        self,
        session_id: str,
        sequence_id: str,
        *,
        fields: Iterable[str] | None = None,
    ) -> CapturedSequence | None:
        store = self._for_session(session_id)
        try:
            header = await self._get(store, f"hdr:{sequence_id}")
        except modal.exception.NotFoundError:
            self._dicts.pop(session_id, None)
            return None
        if not isinstance(header, dict):
            return None
        if header.get("session_id") != session_id:
            return None
        if (
            self.ttl_seconds is not None
            and self.clock() - float(header["created_at"]) > self.ttl_seconds
        ):
            return None
        attempt_id = str(header.get("attempt_id", ""))
        keys = needed_chunk_keys(header, fields)
        try:
            values = await asyncio.gather(
                *(
                    self._get(store, self._chunk_key(sequence_id, attempt_id, key))
                    for key in keys
                )
            )
        except modal.exception.NotFoundError:
            self._dicts.pop(session_id, None)
            return None
        chunks: dict[str, bytes] = {}
        for key, value in zip(keys, values, strict=True):
            if not isinstance(value, (bytes, bytearray)):
                return None
            chunks[key] = bytes(value)
        return await asyncio.to_thread(unpack_capture, header, chunks, fields=fields)

    async def delete_session(self, session_id: str) -> None:
        self._dicts.pop(session_id, None)
        await self._delete_dict(
            session_capture_dict_name(session_id, self.app_id),
            allow_missing=True,
            environment_name=self.environment_name,
        )

    async def expire_before(self, cutoff: float) -> int:
        prefix = f"{self.app_id}-captures-"
        dicts = await self._list_dicts(environment_name=self.environment_name or "")
        expired = 0
        for handle in dicts:
            name = getattr(handle, "name", None)
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            try:
                # Stream keys only so multi-MiB chunk values are never downloaded during sweep.
                header_keys = [
                    key
                    async for key in handle.keys.aio()
                    if isinstance(key, str) and key.startswith("hdr:")
                ]
            except modal.exception.NotFoundError:
                continue
            if not header_keys:
                continue
            remaining_headers = len(header_keys)
            for hdr_key in header_keys:
                try:
                    value = await self._get(handle, hdr_key)
                except modal.exception.NotFoundError:
                    break
                if not isinstance(value, dict):
                    continue
                if float(value.get("created_at", 0.0)) >= cutoff:
                    continue
                seq_id = str(value["sequence_id"])
                attempt_id = str(value.get("attempt_id", ""))
                chunk_keys = [
                    self._chunk_key(seq_id, attempt_id, chunk["key"])
                    for chunk in value.get("chunks", ())
                ]
                with contextlib.suppress(modal.exception.NotFoundError):
                    await self._pop(handle, hdr_key)
                    if chunk_keys:
                        await asyncio.gather(
                            *(self._pop(handle, k) for k in chunk_keys)
                        )
                remaining_headers -= 1
                expired += 1
            if remaining_headers == 0:
                await self._delete_dict(
                    name,
                    allow_missing=True,
                    environment_name=self.environment_name,
                )
        return expired


class _CommitCoalescer:
    """Coalesce concurrent local volume commits while preserving write-after-commit order."""

    def __init__(self, commit_fn: Callable[[], Awaitable[None]]) -> None:
        self._commit_fn = commit_fn
        self._lock = asyncio.Lock()
        self._version = 0
        self._committed = 0
        self._inflight: asyncio.Task[None] | None = None

    async def commit(self) -> None:
        async with self._lock:
            self._version += 1
            target = self._version
        while True:
            async with self._lock:
                if self._committed >= target:
                    return
                task = self._inflight
                if task is None:
                    snapshot = self._version
                    task = asyncio.create_task(self._run(snapshot))
                    self._inflight = task
            await asyncio.shield(task)

    async def _run(self, snapshot: int) -> None:
        try:
            await _retry_transient(self._commit_fn)
            async with self._lock:
                self._committed = max(self._committed, snapshot)
        finally:
            async with self._lock:
                self._inflight = None


class _ReloadGate:
    """Coordinate concurrent file reads with Modal Volume reloads (which reject open FDs)."""

    def __init__(self, reload_fn: Callable[[], Awaitable[None]]) -> None:
        self._reload_fn = reload_fn
        self._cond = asyncio.Condition()
        self._active_readers = 0
        self._reloading = False
        self._generation = 0
        self._reload_task: asyncio.Task[None] | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @contextlib.asynccontextmanager
    async def reading(self) -> AsyncGenerator[int]:
        async with self._cond:
            while self._reloading:
                await self._cond.wait()
            gen = self._generation
            self._active_readers += 1
        try:
            yield gen
        finally:
            async with self._cond:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._cond.notify_all()

    async def run_read[T](self, fn: Callable[..., T], *args: object) -> tuple[int, T]:
        """Run a blocking volume read on a worker thread, keeping the lease until the thread exits."""
        async with self._cond:
            while self._reloading:
                await self._cond.wait()
            gen = self._generation
            self._active_readers += 1
        worker = asyncio.create_task(asyncio.to_thread(fn, *args))

        def _release(_done: asyncio.Task[T]) -> None:
            asyncio.get_running_loop().create_task(self._decrement_readers())

        worker.add_done_callback(_release)
        return gen, await asyncio.shield(worker)

    async def _decrement_readers(self) -> None:
        async with self._cond:
            self._active_readers -= 1
            if self._active_readers == 0:
                self._cond.notify_all()

    async def reload(self, seen_generation: int) -> int:
        while True:
            task: asyncio.Task[None] | None = None
            async with self._cond:
                if self._generation > seen_generation:
                    return self._generation
                if self._reload_task is not None:
                    task = self._reload_task
                elif not self._reloading:
                    self._reloading = True
                    try:
                        while self._active_readers > 0:
                            await self._cond.wait()
                    except BaseException:
                        self._reloading = False
                        self._cond.notify_all()
                        raise
                    if self._generation > seen_generation:
                        self._reloading = False
                        self._cond.notify_all()
                        return self._generation
                    task = asyncio.create_task(self._run())
                    self._reload_task = task
                else:
                    while self._reloading and self._reload_task is None:
                        await self._cond.wait()
                    if self._generation > seen_generation:
                        return self._generation
                    if self._reload_task is None:
                        # The previous waiting leader was cancelled before starting; retry to take over.
                        continue
                    task = self._reload_task
            if task is not None:
                await asyncio.shield(task)
            if self._generation > seen_generation:
                return self._generation

    async def _run(self) -> None:
        try:
            await _retry_transient(self._reload_fn)
            async with self._cond:
                self._generation += 1
        finally:
            async with self._cond:
                self._reload_task = None
                self._reloading = False
                self._cond.notify_all()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _chunk_filename(chunk_key: str) -> str:
    return chunk_key.replace(":", "_") + ".bin"


class ModalVolumeCaptureStore:
    """Modal v2 Volume capture store supporting mounted FS and remote Volume API modes."""

    def __init__(
        self,
        volume: modal.Volume,
        app_id: str | None = None,
        *,
        mode: VolumeMode = "mount",
        mount_root: str | Path = DEFAULT_CAPTURE_MOUNT_ROOT,
        ttl_seconds: float = DEFAULT_CAPTURE_TTL_SECONDS,
        max_concurrency: int = DEFAULT_VOLUME_API_CONCURRENCY,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.volume = volume
        self.app_id = (
            app_id or os.environ.get("SPINDLE_FRONTEND_APP_ID") or current_app_id()
        )
        self.mode: VolumeMode = mode
        self.mount_root = Path(mount_root)
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._committer = _CommitCoalescer(lambda: self.volume.commit.aio())
        self._reload_gate = _ReloadGate(lambda: self.volume.reload.aio())

    def _rel_session_dir(self, session_id: str) -> str:
        return f"/{self.app_id}/{_short_digest(session_id)}"

    def _rel_seq_dir(self, session_id: str, sequence_id: str) -> str:
        return f"{self._rel_session_dir(session_id)}/{_short_digest(sequence_id)}"

    def _local_seq_dir(self, session_id: str, sequence_id: str) -> Path:
        return (
            self.mount_root
            / self.app_id
            / _short_digest(session_id)
            / _short_digest(sequence_id)
        )

    async def write_sequence(
        self,
        sequence: CapturedSequence,
        *,
        target_chunk_bytes: int | None = None,
        compression: Compression = "none",
    ) -> WriteStats:
        t0 = time.perf_counter()
        packed = await asyncio.to_thread(
            pack_capture,
            sequence,
            target_chunk_bytes=target_chunk_bytes,
            compression=compression,
        )
        pack_s = time.perf_counter() - t0
        attempt_id = str(packed.header["attempt_id"])
        header_bytes = json.dumps(packed.header, separators=(",", ":")).encode("utf-8")
        t1 = time.perf_counter()
        if self.mode == "mount":
            seq_dir = self._local_seq_dir(sequence.session_id, sequence.sequence_id)
            await asyncio.to_thread(
                self._write_local_files,
                seq_dir,
                attempt_id,
                packed.chunks,
                header_bytes,
            )
            await self._committer.commit()
        else:
            rel_dir = self._rel_seq_dir(sequence.session_id, sequence.sequence_id)
            await self._write_api_files(
                rel_dir, attempt_id, packed.chunks, header_bytes
            )
        write_s = time.perf_counter() - t1
        return WriteStats(
            num_chunks=len(packed.chunks),
            total_chunk_bytes=packed.total_chunk_bytes,
            pack_seconds=pack_s,
            write_seconds=write_s,
        )

    @staticmethod
    def _write_local_files(
        seq_dir: Path,
        attempt_id: str,
        chunks: tuple[tuple[str, bytes], ...],
        header_bytes: bytes,
    ) -> None:
        attempt_dir = seq_dir / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        for key, data in chunks:
            _atomic_write_bytes(attempt_dir / _chunk_filename(key), data)
        _atomic_write_bytes(seq_dir / "header.json", header_bytes)

    async def _write_api_files(
        self,
        rel_dir: str,
        attempt_id: str,
        chunks: tuple[tuple[str, bytes], ...],
        header_bytes: bytes,
    ) -> None:
        async def upload() -> None:
            async with self.volume.batch_upload.aio(force=True) as batch:
                for key, data in chunks:
                    batch.put_file(
                        io.BytesIO(data),
                        f"{rel_dir}/{attempt_id}/{_chunk_filename(key)}",
                    )
                batch.put_file(io.BytesIO(header_bytes), f"{rel_dir}/header.json")

        async with self._sem:
            await _retry_transient(upload)

    async def read_sequence(
        self,
        session_id: str,
        sequence_id: str,
        *,
        fields: Iterable[str] | None = None,
    ) -> CapturedSequence | None:
        if self.mode == "mount":
            return await self._read_mounted(session_id, sequence_id, fields=fields)
        return await self._read_api(session_id, sequence_id, fields=fields)

    def _read_local_sequence_bytes(
        self,
        seq_dir: Path,
        session_id: str,
        fields: Iterable[str] | None,
    ) -> tuple[dict[str, Any], dict[str, bytes]] | None:
        header_path = seq_dir / "header.json"
        try:
            header = json.loads(header_path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if not isinstance(header, dict) or header.get("session_id") != session_id:
            return None
        if (
            self.ttl_seconds is not None
            and self.clock() - float(header["created_at"]) > self.ttl_seconds
        ):
            return None
        attempt_id = str(header.get("attempt_id", ""))
        chunk_dir = seq_dir / attempt_id if attempt_id else seq_dir
        keys = needed_chunk_keys(header, fields)
        chunks: dict[str, bytes] = {}
        for key in keys:
            try:
                chunks[key] = (chunk_dir / _chunk_filename(key)).read_bytes()
            except FileNotFoundError:
                return None
        return header, chunks

    async def _read_mounted(
        self,
        session_id: str,
        sequence_id: str,
        *,
        fields: Iterable[str] | None,
    ) -> CapturedSequence | None:
        seq_dir = self._local_seq_dir(session_id, sequence_id)
        gen, result = await self._reload_gate.run_read(
            self._read_local_sequence_bytes, seq_dir, session_id, fields
        )
        if result is None:
            await self._reload_gate.reload(gen)
            _, result = await self._reload_gate.run_read(
                self._read_local_sequence_bytes, seq_dir, session_id, fields
            )
        if result is None:
            return None
        header, chunks = result
        return await asyncio.to_thread(unpack_capture, header, chunks, fields=fields)

    async def _read_api_file_once(self, path: str) -> bytes | None:
        try:
            chunks = [chunk async for chunk in self.volume.read_file.aio(path)]
            return b"".join(chunks)
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None

    async def _read_api_file(self, path: str) -> bytes | None:
        async with self._sem:
            return await _retry_transient(lambda: self._read_api_file_once(path))

    async def _read_api(
        self,
        session_id: str,
        sequence_id: str,
        *,
        fields: Iterable[str] | None,
    ) -> CapturedSequence | None:
        rel_dir = self._rel_seq_dir(session_id, sequence_id)
        raw_header = await self._read_api_file(f"{rel_dir}/header.json")
        if raw_header is None:
            return None
        try:
            header = json.loads(raw_header)
        except json.JSONDecodeError:
            return None
        if not isinstance(header, dict) or header.get("session_id") != session_id:
            return None
        if (
            self.ttl_seconds is not None
            and self.clock() - float(header["created_at"]) > self.ttl_seconds
        ):
            return None
        attempt_id = str(header.get("attempt_id", ""))
        chunk_prefix = f"{rel_dir}/{attempt_id}" if attempt_id else rel_dir
        keys = needed_chunk_keys(header, fields)
        values = await asyncio.gather(
            *(
                self._read_api_file(f"{chunk_prefix}/{_chunk_filename(key)}")
                for key in keys
            )
        )
        chunks: dict[str, bytes] = {}
        for key, data in zip(keys, values, strict=True):
            if data is None:
                return None
            chunks[key] = data
        return await asyncio.to_thread(unpack_capture, header, chunks, fields=fields)

    async def delete_session(self, session_id: str) -> None:
        if self.mode == "mount":
            session_dir = self.mount_root / self.app_id / _short_digest(session_id)
            removed = await asyncio.to_thread(self._rmtree_if_exists, session_dir)
            if removed:
                await self._committer.commit()
            return
        try:
            await self.volume.remove_file.aio(
                self._rel_session_dir(session_id), recursive=True
            )
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass

    @staticmethod
    def _rmtree_if_exists(path: Path) -> bool:
        if not path.exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        return True

    async def expire_before(self, cutoff: float) -> int:
        if self.mode == "mount":
            gen = self._reload_gate.generation
            await self._reload_gate.reload(gen)
            expired = await asyncio.to_thread(
                self._expire_mounted_sync, self.mount_root / self.app_id, cutoff
            )
            if expired > 0:
                await self._committer.commit()
            return expired
        return await self._expire_api(cutoff)

    @staticmethod
    def _expire_mounted_sync(root: Path, cutoff: float) -> int:
        if not root.is_dir():
            return 0
        expired = 0
        for session_dir in list(root.iterdir()):
            if not session_dir.is_dir():
                continue
            for seq_dir in list(session_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                header_path = seq_dir / "header.json"
                try:
                    header = json.loads(header_path.read_bytes())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if float(header.get("created_at", 0.0)) < cutoff:
                    shutil.rmtree(seq_dir, ignore_errors=True)
                    expired += 1
            with contextlib.suppress(OSError):
                session_dir.rmdir()
        return expired

    async def _expire_api(self, cutoff: float) -> int:
        try:
            entries = await _retry_transient(
                lambda: self.volume.listdir.aio(f"/{self.app_id}", recursive=True)
            )
        except (FileNotFoundError, modal.exception.NotFoundError):
            return 0
        # Group entries by sequence directory (<app_id>/<session_digest>/<seq_digest>).
        # Use entry.mtime when available so we avoid downloading every header.json and also
        # clean up headerless orphan directories left by mid-write replica crashes.
        seq_dirs: dict[str, list[Any]] = {}
        for entry in entries:
            parts = entry.path.lstrip("/").split("/")
            if len(parts) < 3:
                continue
            seq_dir = "/" + "/".join(parts[:3])
            seq_dirs.setdefault(seq_dir, []).append(entry)
        expired = 0
        for seq_dir, dir_entries in seq_dirs.items():
            mtimes = [
                float(m)
                for e in dir_entries
                if (m := getattr(e, "mtime", None)) is not None and m > 0
            ]
            if mtimes:
                is_expired = max(mtimes) < cutoff
            else:
                raw = await self._read_api_file(f"{seq_dir}/header.json")
                if raw is None:
                    continue
                try:
                    header = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                is_expired = float(header.get("created_at", 0.0)) < cutoff
            if not is_expired:
                continue
            with contextlib.suppress(FileNotFoundError, modal.exception.NotFoundError):
                await _retry_transient(
                    lambda d=seq_dir: self.volume.remove_file.aio(d, recursive=True)
                )
            expired += 1
        return expired


def build_modal_capture_store(
    kind: Literal["dict", "volume"],
    *,
    app_id: str | None = None,
    volume_name: str = DEFAULT_CAPTURE_VOLUME_NAME,
    mount_root: str | Path = DEFAULT_CAPTURE_MOUNT_ROOT,
    volume_mode: VolumeMode = "mount",
    environment_name: str | None = None,
    ttl_seconds: float = DEFAULT_CAPTURE_TTL_SECONDS,
) -> CaptureStore:
    """Build either a Modal Dict or Modal v2 Volume capture store."""
    if kind == "dict":
        return ModalDictCaptureStore(
            app_id=app_id,
            environment_name=environment_name,
            ttl_seconds=ttl_seconds,
        )
    if kind == "volume":
        volume = modal.Volume.from_name(
            volume_name,
            environment_name=environment_name,
            create_if_missing=True,
            version=2,
        )
        return ModalVolumeCaptureStore(
            volume,
            app_id=app_id,
            mode=volume_mode,
            mount_root=mount_root,
            ttl_seconds=ttl_seconds,
        )
    raise ValueError(f"unsupported capture store kind: {kind!r}")
