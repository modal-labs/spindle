"""Pluggable capture store contract and in-memory implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from .codec import (
    DEFAULT_CHUNK_BYTES,
    CapturedSequence,
    Compression,
    needed_chunk_keys,
    pack_capture,
    unpack_capture,
)
from .config import DEFAULT_CAPTURE_TTL_SECONDS


@dataclass(frozen=True)
class WriteStats:
    """Summary of one capture store write, useful for benchmarking and telemetry."""

    num_chunks: int
    total_chunk_bytes: int
    pack_seconds: float
    write_seconds: float


class CaptureStore(Protocol):
    """Session-scoped store for server-side stability info captures."""

    async def write_sequence(
        self,
        sequence: CapturedSequence,
        *,
        target_chunk_bytes: int | None = DEFAULT_CHUNK_BYTES,
        compression: Compression = "none",
    ) -> WriteStats: ...

    async def read_sequence(
        self,
        session_id: str,
        sequence_id: str,
        *,
        fields: Iterable[str] | None = None,
    ) -> CapturedSequence | None: ...

    async def delete_session(self, session_id: str) -> None: ...

    async def expire_before(self, cutoff: float) -> int: ...


class InMemoryCaptureStore:
    """In-memory CaptureStore for tests and local runs; exercises the full codec."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_CAPTURE_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._sessions: dict[str, dict[str, object]] = {}
        self._lock = asyncio.Lock()

    async def write_sequence(
        self,
        sequence: CapturedSequence,
        *,
        target_chunk_bytes: int | None = DEFAULT_CHUNK_BYTES,
        compression: Compression = "none",
    ) -> WriteStats:
        t0 = time.perf_counter()
        packed = pack_capture(
            sequence,
            target_chunk_bytes=target_chunk_bytes,
            compression=compression,
        )
        pack_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        attempt_id = str(packed.header["attempt_id"])
        async with self._lock:
            bucket = self._sessions.setdefault(sequence.session_id, {})
            for key, data in packed.chunks:
                bucket[f"chk:{sequence.sequence_id}:{attempt_id}:{key}"] = data
            bucket[f"hdr:{sequence.sequence_id}"] = packed.header
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
        async with self._lock:
            bucket = self._sessions.get(session_id)
            if bucket is None:
                return None
            header = bucket.get(f"hdr:{sequence_id}")
            if not isinstance(header, dict):
                return None
            if (
                self.ttl_seconds is not None
                and self.clock() - float(header["created_at"]) > self.ttl_seconds
            ):
                return None
            keys = needed_chunk_keys(header, fields)
            attempt_id = str(header.get("attempt_id", ""))
            chunks: dict[str, bytes] = {}
            for key in keys:
                chunk_key = (
                    f"chk:{sequence_id}:{attempt_id}:{key}"
                    if attempt_id
                    else f"chk:{sequence_id}:{key}"
                )
                raw = bucket.get(chunk_key)
                if not isinstance(raw, bytes):
                    return None
                chunks[key] = raw
        return unpack_capture(header, chunks, fields=fields)

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def expire_before(self, cutoff: float) -> int:
        expired = 0
        async with self._lock:
            for session_id in list(self._sessions):
                bucket = self._sessions[session_id]
                for key in list(bucket):
                    if not key.startswith("hdr:"):
                        continue
                    header = bucket[key]
                    if not isinstance(header, dict):
                        continue
                    if float(header["created_at"]) >= cutoff:
                        continue
                    seq_id = str(header["sequence_id"])
                    attempt_id = str(header.get("attempt_id", ""))
                    bucket.pop(key, None)
                    for chunk in header.get("chunks", ()):
                        chunk_key = (
                            f"chk:{seq_id}:{attempt_id}:{chunk['key']}"
                            if attempt_id
                            else f"chk:{seq_id}:{chunk['key']}"
                        )
                        bucket.pop(chunk_key, None)
                    expired += 1
                if not bucket:
                    self._sessions.pop(session_id, None)
        return expired
