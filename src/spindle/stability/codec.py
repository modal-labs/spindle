"""Compact binary encoding, chunking, and SGLang extraction for stability captures."""

from __future__ import annotations

import base64
import math
import struct
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import zstandard

from .config import CapturePlan

Compression = Literal["none", "zstd"]
RouteEncoding = Literal["uint8", "bitpack", "uint16", "int32"]

FORMAT_VERSION = 1
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
RECORDED_FIELDS = frozenset(
    {
        "routed_experts",
        "sampling_mask",
        "sampling_probabilities",
        "sampler_draws",
    }
)
SGLANG_CAPTURE_META_KEYS = (
    "routed_experts",
    "routed_experts_dtype",
    "routed_experts_num_layers",
    "routed_experts_per_token",
    "output_token_sampling_mask",
    "output_token_sampling_logprobs",
    "output_token_alt_tokens",
    "sampler_draws",
)
ROUTED_EXPERTS_WIRE_DTYPES: dict[str, np.dtype] = {
    "int32": np.dtype("<i4"),
    "int16": np.dtype("<i2"),
    "uint16": np.dtype("<u2"),
    "uint8": np.dtype("u1"),
}

_MASK_HEADER = struct.Struct("<II")


@dataclass(frozen=True)
class CapturedSequence:
    """In-memory stability info captured for one sampled sequence."""

    sequence_id: str
    session_id: str
    created_at: float
    prompt_tokens: np.ndarray
    response_tokens: np.ndarray
    temperature: float
    routed_experts: np.ndarray | None = None
    sampling_mask_ids: np.ndarray | None = None
    sampling_mask_offsets: np.ndarray | None = None
    sampling_mask_logprobs: np.ndarray | None = None
    sampling_logprobs: np.ndarray | None = None
    sampler_draws: np.ndarray | None = None
    recorded_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        inferred: set[str] = set(self.recorded_fields)
        if self.routed_experts is not None:
            inferred.add("routed_experts")
        if self.sampling_mask_ids is not None:
            inferred.add("sampling_mask")
        if self.sampling_mask_logprobs is not None:
            inferred.add("sampling_probabilities")
        if self.sampler_draws is not None:
            inferred.add("sampler_draws")
        object.__setattr__(self, "recorded_fields", frozenset(inferred))


@dataclass(frozen=True)
class PackedCapture:
    """Header-last chunked representation ready for any key/file capture store."""

    header: dict[str, Any]
    chunks: tuple[tuple[str, bytes], ...]

    @property
    def total_chunk_bytes(self) -> int:
        return sum(len(data) for _, data in self.chunks)


def pack_bits_u16(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integers in [0, 2**bits) tightly in little-endian bit order."""
    if not 1 <= bits <= 15:
        raise ValueError(f"bit-packing supports 1..15 bits, got {bits}")
    flat = np.asarray(values, dtype=np.uint64).reshape(-1)
    if flat.size == 0:
        return b""
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint64)])
    groups = flat.reshape(-1, 8)
    n_groups = groups.shape[0]
    low = np.zeros(n_groups, dtype=np.uint64)
    high = np.zeros(n_groups, dtype=np.uint64)
    for idx in range(8):
        bit_pos = idx * bits
        col = groups[:, idx]
        if bit_pos < 64:
            low |= col << np.uint64(bit_pos)
            if bit_pos + bits > 64:
                high |= col >> np.uint64(64 - bit_pos)
        else:
            high |= col << np.uint64(bit_pos - 64)
    stacked = np.empty((n_groups, 2), dtype="<u8")
    stacked[:, 0] = low
    stacked[:, 1] = high
    return np.ascontiguousarray(stacked.view(np.uint8)[:, :bits]).tobytes()


def unpack_bits_u16(raw: bytes | memoryview, count: int, bits: int) -> np.ndarray:
    """Unpack count integers packed by pack_bits_u16 into an int32 1D array."""
    if not 1 <= bits <= 15:
        raise ValueError(f"bit-packing supports 1..15 bits, got {bits}")
    if count == 0:
        return np.empty(0, dtype=np.int32)
    n_groups = (count + 7) // 8
    expected = n_groups * bits
    if len(raw) != expected:
        raise ValueError(f"bit-packed buffer has {len(raw)} bytes, expected {expected}")
    buf = np.zeros((n_groups, 16), dtype=np.uint8)
    buf[:, :bits] = np.frombuffer(raw, dtype=np.uint8).reshape(n_groups, bits)
    words = buf.view("<u8")
    low = words[:, 0]
    high = words[:, 1]
    mask = np.uint64((1 << bits) - 1)
    out = np.empty((n_groups, 8), dtype=np.uint16)
    for idx in range(8):
        bit_pos = idx * bits
        if bit_pos + bits <= 64:
            vals = (low >> np.uint64(bit_pos)) & mask
        elif bit_pos >= 64:
            vals = (high >> np.uint64(bit_pos - 64)) & mask
        else:
            vals = (
                (low >> np.uint64(bit_pos)) | (high << np.uint64(64 - bit_pos))
            ) & mask
        out[:, idx] = vals.astype(np.uint16)
    return out.reshape(-1)[:count].astype(np.int32)


def _compress(data: bytes, compression: Compression, level: int = 1) -> bytes:
    if compression == "none":
        return data
    return zstandard.ZstdCompressor(level=level).compress(data)


def _decompress(data: bytes, compression: Compression) -> bytes:
    if compression == "none":
        return data
    return zstandard.ZstdDecompressor().decompress(data)


def _encode_token_array(tokens: np.ndarray) -> str:
    arr = np.asarray(tokens, dtype="<u4").reshape(-1)
    compressed = zstandard.ZstdCompressor(level=1).compress(arr.tobytes())
    return base64.b64encode(compressed).decode("ascii")


def _decode_token_array(encoded: str, expected_len: int) -> np.ndarray:
    raw = zstandard.ZstdDecompressor().decompress(base64.b64decode(encoded))
    arr = np.frombuffer(raw, dtype="<u4").astype(np.int64)
    if arr.size != expected_len:
        raise ValueError(
            f"token array length {arr.size} does not match header {expected_len}"
        )
    return arr


def _choose_route_encoding(
    routes: np.ndarray,
    *,
    bitpack_routes: bool = True,
) -> tuple[RouteEncoding, int]:
    if routes.size == 0:
        return "uint8", 8
    low = int(routes.min())
    high = int(routes.max())
    if low < 0:
        return "int32", 32
    if high <= 255:
        return "uint8", 8
    if high <= 65_535:
        bits = high.bit_length()
        if bitpack_routes and bits <= 15:
            return "bitpack", bits
        return "uint16", 16
    return "int32", 32


def _encode_routes_slice(
    values: np.ndarray, encoding: RouteEncoding, bits: int
) -> bytes:
    if encoding == "uint8":
        return np.ascontiguousarray(values, dtype=np.uint8).tobytes()
    if encoding == "bitpack":
        return pack_bits_u16(values, bits)
    if encoding == "uint16":
        return np.ascontiguousarray(values, dtype="<u2").tobytes()
    return np.ascontiguousarray(values, dtype="<i4").tobytes()


def _decode_routes_slice(
    raw: bytes, count: int, encoding: RouteEncoding, bits: int
) -> np.ndarray:
    if encoding == "uint8":
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    if encoding == "bitpack":
        return unpack_bits_u16(raw, count, bits)
    if encoding == "uint16":
        return np.frombuffer(raw, dtype="<u2").astype(np.int32)
    return np.frombuffer(raw, dtype="<i4").copy()


def _encode_mask_chunk(
    ids: np.ndarray, offsets: np.ndarray, logprobs: np.ndarray
) -> bytes:
    rel_offsets = np.ascontiguousarray(offsets - offsets[0], dtype="<u4")
    ids_u32 = np.ascontiguousarray(ids, dtype="<u4")
    logprobs_f32 = np.ascontiguousarray(logprobs, dtype="<f4")
    return b"".join(
        (
            _MASK_HEADER.pack(rel_offsets.size, ids_u32.size),
            rel_offsets.tobytes(),
            ids_u32.tobytes(),
            logprobs_f32.tobytes(),
        )
    )


def _decode_mask_chunk(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_offsets, n_ids = _MASK_HEADER.unpack_from(raw, 0)
    pos = _MASK_HEADER.size
    offsets_end = pos + n_offsets * 4
    ids_end = offsets_end + n_ids * 4
    offsets = np.frombuffer(raw[pos:offsets_end], dtype="<u4").astype(np.int64)
    ids = np.frombuffer(raw[offsets_end:ids_end], dtype="<u4").astype(np.int64)
    logprobs = np.frombuffer(raw[ids_end:], dtype="<f4").copy()
    if logprobs.size != max(n_offsets - 1, 0):
        raise ValueError("corrupt sampling mask chunk: logprob length mismatch")
    return ids, offsets, logprobs


def pack_capture(
    sequence: CapturedSequence,
    *,
    target_chunk_bytes: int | None = DEFAULT_CHUNK_BYTES,
    compression: Compression = "none",
    zstd_level: int = 1,
    bitpack_routes: bool = True,
    attempt_id: str | None = None,
) -> PackedCapture:
    """Pack a CapturedSequence into a header dict and bounded-size binary chunks."""
    if target_chunk_bytes is not None and target_chunk_bytes < 1024:
        raise ValueError("target_chunk_bytes must be at least 1024 or None")
    prompt = np.asarray(sequence.prompt_tokens, dtype=np.int64).reshape(-1)
    response = np.asarray(sequence.response_tokens, dtype=np.int64).reshape(-1)
    chunk_meta: list[dict[str, Any]] = []
    chunks: list[tuple[str, bytes]] = []
    recorded: list[str] = []
    routes_meta: dict[str, Any] | None = None
    draws_meta: dict[str, Any] | None = None

    if sequence.routed_experts is not None:
        routes = np.asarray(sequence.routed_experts, dtype=np.int32)
        if routes.ndim != 3:
            raise ValueError(
                f"routed_experts must be 3D [tokens, layers, experts], got {routes.shape}"
            )
        recorded.append("routed_experts")
        encoding, bits = _choose_route_encoding(routes, bitpack_routes=bitpack_routes)
        n_rows, n_layers, n_experts = routes.shape
        routes_meta = {
            "num_rows": n_rows,
            "num_layers": n_layers,
            "experts_per_token": n_experts,
            "encoding": encoding,
            "bits": bits,
        }
        bytes_per_row = max(1, (n_layers * n_experts * bits + 7) // 8)
        rows_per_chunk = (
            n_rows
            if target_chunk_bytes is None or n_rows == 0
            else max(1, target_chunk_bytes // bytes_per_row)
        )
        for idx, start in enumerate(range(0, n_rows, max(rows_per_chunk, 1))):
            end = min(n_rows, start + rows_per_chunk)
            raw = _encode_routes_slice(routes[start:end], encoding, bits)
            payload = _compress(raw, compression, zstd_level)
            key = f"routes:{idx}"
            chunks.append((key, payload))
            chunk_meta.append(
                {
                    "key": key,
                    "field": "routed_experts",
                    "start": start,
                    "end": end,
                    "count": (end - start) * n_layers * n_experts,
                }
            )

    if sequence.sampling_mask_ids is not None:
        if sequence.sampling_mask_offsets is None or sequence.sampling_logprobs is None:
            raise ValueError(
                "sampling_mask requires sampling_mask_offsets and sampling_logprobs"
            )
        recorded.append("sampling_mask")
        ids = np.asarray(sequence.sampling_mask_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(sequence.sampling_mask_offsets, dtype=np.int64).reshape(-1)
        logprobs = np.asarray(sequence.sampling_logprobs, dtype=np.float32).reshape(-1)
        has_probs = sequence.sampling_mask_logprobs is not None
        mask_logprobs = (
            np.asarray(sequence.sampling_mask_logprobs, dtype=np.float32).reshape(-1)
            if has_probs
            else None
        )
        if has_probs:
            recorded.append("sampling_probabilities")
        n_resp = response.size
        # Partition response positions so each chunk's byte size stays <= target_chunk_bytes.
        ranges: list[tuple[int, int]] = []
        if n_resp == 0:
            ranges.append((0, 0))
        elif target_chunk_bytes is None:
            ranges.append((0, n_resp))
        else:
            start = 0
            while start < n_resp:
                end = start
                est_bytes = _MASK_HEADER.size + 4
                while end < n_resp:
                    support_len = int(offsets[end + 1] - offsets[end])
                    step_bytes = 8 + support_len * 4
                    if end > start and est_bytes + step_bytes > target_chunk_bytes:
                        break
                    est_bytes += step_bytes
                    end += 1
                ranges.append((start, end))
                start = end

        for idx, (start, end) in enumerate(ranges):
            id_start = int(offsets[start])
            id_end = int(offsets[end])
            raw_mask = _encode_mask_chunk(
                ids[id_start:id_end],
                offsets[start : end + 1],
                logprobs[start:end],
            )
            mask_key = f"mask:{idx}"
            chunks.append((mask_key, _compress(raw_mask, compression, zstd_level)))
            chunk_meta.append(
                {
                    "key": mask_key,
                    "field": "sampling_mask",
                    "start": start,
                    "end": end,
                    "count": id_end - id_start,
                }
            )
            if mask_logprobs is not None:
                raw_probs = np.ascontiguousarray(
                    mask_logprobs[id_start:id_end], dtype="<f4"
                ).tobytes()
                probs_key = f"mask_probs:{idx}"
                chunks.append(
                    (probs_key, _compress(raw_probs, compression, zstd_level))
                )
                chunk_meta.append(
                    {
                        "key": probs_key,
                        "field": "sampling_probabilities",
                        "start": start,
                        "end": end,
                        "count": id_end - id_start,
                    }
                )

    if sequence.sampler_draws is not None:
        draws = np.asarray(sequence.sampler_draws, dtype=np.int64)
        if draws.ndim != 2:
            raise ValueError(
                f"sampler_draws must be 2D [response_tokens, num_draws], got {draws.shape}"
            )
        recorded.append("sampler_draws")
        n_resp, n_draws = draws.shape
        draws_meta = {"num_rows": n_resp, "num_draws": n_draws}
        bytes_per_row = max(1, n_draws * 4)
        rows_per_chunk = (
            n_resp
            if target_chunk_bytes is None or n_resp == 0
            else max(1, target_chunk_bytes // bytes_per_row)
        )
        for idx, start in enumerate(range(0, n_resp, max(rows_per_chunk, 1))):
            end = min(n_resp, start + rows_per_chunk)
            raw = np.ascontiguousarray(draws[start:end], dtype="<u4").tobytes()
            key = f"draws:{idx}"
            chunks.append((key, _compress(raw, compression, zstd_level)))
            chunk_meta.append(
                {
                    "key": key,
                    "field": "sampler_draws",
                    "start": start,
                    "end": end,
                    "count": (end - start) * n_draws,
                }
            )

    header: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "attempt_id": attempt_id or uuid.uuid4().hex,
        "sequence_id": sequence.sequence_id,
        "session_id": sequence.session_id,
        "created_at": float(sequence.created_at),
        "temperature": float(sequence.temperature),
        "num_prompt_tokens": int(prompt.size),
        "num_response_tokens": int(response.size),
        "prompt_tokens_b64": _encode_token_array(prompt),
        "response_tokens_b64": _encode_token_array(response),
        "recorded": recorded,
        "compression": compression,
        "routes_meta": routes_meta,
        "draws_meta": draws_meta,
        "chunks": chunk_meta,
    }
    return PackedCapture(header=header, chunks=tuple(chunks))


def needed_chunk_keys(
    header: Mapping[str, Any],
    fields: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return chunk keys required to unpack the requested fields (all if None)."""
    wanted = set(RECORDED_FIELDS) if fields is None else set(fields)
    if "sampling_probabilities" in wanted:
        wanted.add("sampling_mask")
    return tuple(
        chunk["key"] for chunk in header.get("chunks", ()) if chunk["field"] in wanted
    )


def unpack_capture(
    header: Mapping[str, Any],
    chunks: Mapping[str, bytes],
    *,
    fields: Iterable[str] | None = None,
) -> CapturedSequence:
    """Reconstruct a CapturedSequence from header metadata and fetched chunks."""
    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported stability capture format: {header.get('format_version')!r}"
        )
    compression: Compression = header.get("compression", "none")
    prompt = _decode_token_array(
        header["prompt_tokens_b64"], int(header["num_prompt_tokens"])
    )
    response = _decode_token_array(
        header["response_tokens_b64"], int(header["num_response_tokens"])
    )
    recorded = frozenset(header.get("recorded", ()))
    wanted = set(RECORDED_FIELDS) if fields is None else set(fields)
    if "sampling_probabilities" in wanted:
        wanted.add("sampling_mask")

    by_field: dict[str, list[dict[str, Any]]] = {}
    for entry in header.get("chunks", ()):
        by_field.setdefault(entry["field"], []).append(entry)

    routed_experts: np.ndarray | None = None
    if "routed_experts" in recorded and "routed_experts" in wanted:
        meta = header["routes_meta"]
        encoding: RouteEncoding = meta["encoding"]
        bits = int(meta["bits"])
        parts: list[np.ndarray] = []
        for entry in by_field.get("routed_experts", ()):
            raw = _decompress(chunks[entry["key"]], compression)
            parts.append(_decode_routes_slice(raw, int(entry["count"]), encoding, bits))
        flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
        routed_experts = flat.reshape(
            int(meta["num_rows"]),
            int(meta["num_layers"]),
            int(meta["experts_per_token"]),
        )

    sampling_mask_ids: np.ndarray | None = None
    sampling_mask_offsets: np.ndarray | None = None
    sampling_logprobs: np.ndarray | None = None
    if "sampling_mask" in recorded and "sampling_mask" in wanted:
        id_parts: list[np.ndarray] = []
        offset_parts: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        lp_parts: list[np.ndarray] = []
        running_offset = 0
        for entry in by_field.get("sampling_mask", ()):
            raw = _decompress(chunks[entry["key"]], compression)
            c_ids, c_offsets, c_lps = _decode_mask_chunk(raw)
            id_parts.append(c_ids)
            offset_parts.append(c_offsets[1:] + running_offset)
            running_offset += int(c_ids.size)
            lp_parts.append(c_lps)
        sampling_mask_ids = (
            np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.int64)
        )
        sampling_mask_offsets = np.concatenate(offset_parts)
        sampling_logprobs = (
            np.concatenate(lp_parts) if lp_parts else np.empty(0, dtype=np.float32)
        )

    sampling_mask_logprobs: np.ndarray | None = None
    if "sampling_probabilities" in recorded and "sampling_probabilities" in wanted:
        prob_parts: list[np.ndarray] = []
        for entry in by_field.get("sampling_probabilities", ()):
            raw = _decompress(chunks[entry["key"]], compression)
            prob_parts.append(np.frombuffer(raw, dtype="<f4").copy())
        sampling_mask_logprobs = (
            np.concatenate(prob_parts) if prob_parts else np.empty(0, dtype=np.float32)
        )

    sampler_draws: np.ndarray | None = None
    if "sampler_draws" in recorded and "sampler_draws" in wanted:
        meta = header["draws_meta"]
        draw_parts: list[np.ndarray] = []
        for entry in by_field.get("sampler_draws", ()):
            raw = _decompress(chunks[entry["key"]], compression)
            draw_parts.append(np.frombuffer(raw, dtype="<u4").astype(np.int64))
        flat = np.concatenate(draw_parts) if draw_parts else np.empty(0, dtype=np.int64)
        sampler_draws = flat.reshape(int(meta["num_rows"]), int(meta["num_draws"]))

    return CapturedSequence(
        sequence_id=str(header["sequence_id"]),
        session_id=str(header["session_id"]),
        created_at=float(header["created_at"]),
        prompt_tokens=prompt,
        response_tokens=response,
        temperature=float(header["temperature"]),
        routed_experts=routed_experts,
        sampling_mask_ids=sampling_mask_ids,
        sampling_mask_offsets=sampling_mask_offsets,
        sampling_mask_logprobs=sampling_mask_logprobs,
        sampling_logprobs=sampling_logprobs,
        sampler_draws=sampler_draws,
        recorded_fields=recorded,
    )


def extract_captured_sequence(
    meta: dict[str, Any],
    prompt_tokens: list[int],
    response_tokens: list[int],
    *,
    session_id: str,
    sequence_id: str,
    temperature: float,
    plan: CapturePlan,
    num_layers: int | None = None,
    experts_per_token: int | None = None,
    strip_meta: bool = True,
    created_at: float | None = None,
) -> CapturedSequence:
    """Validate and extract requested stability info from SGLang meta_info."""
    routed_experts: np.ndarray | None = None
    if plan.routed_experts:
        encoded = meta.get("routed_experts")
        if not isinstance(encoded, str) or (response_tokens and not encoded):
            raise ValueError(
                "sampler did not return routed_experts; enable route capture on the rollout server"
            )
        dtype_name = meta.get("routed_experts_dtype", "int32")
        if (
            not isinstance(dtype_name, str)
            or dtype_name not in ROUTED_EXPERTS_WIRE_DTYPES
        ):
            raise ValueError(f"unsupported routed_experts_dtype: {dtype_name!r}")
        wire_dtype = ROUTED_EXPERTS_WIRE_DTYPES[dtype_name]
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) % wire_dtype.itemsize:
            raise ValueError(f"invalid routed_experts {dtype_name} buffer")
        values = np.frombuffer(raw, dtype=wire_dtype)
        resolved_layers = meta.get("routed_experts_num_layers") or num_layers
        resolved_experts = meta.get("routed_experts_per_token") or experts_per_token
        n_input = len(prompt_tokens) + len(response_tokens) - 1
        if not resolved_layers or not resolved_experts:
            raise ValueError(
                "routed_experts capture requires configured num_layers and experts_per_token"
            )
        stride = int(resolved_layers) * int(resolved_experts)
        if values.size == (n_input + 1) * stride:
            values = values[: n_input * stride]
        if values.size != n_input * stride:
            raise ValueError(
                "routed_experts length does not match prompt + response - 1"
            )
        routed_experts = values.reshape(
            n_input, int(resolved_layers), int(resolved_experts)
        ).astype(np.int32, copy=True)

    mask_ids: np.ndarray | None = None
    mask_offsets: np.ndarray | None = None
    mask_logprobs: np.ndarray | None = None
    sampling_logprobs: np.ndarray | None = None
    if plan.sampling_mask or plan.sampling_probabilities:
        supports = meta.get("output_token_sampling_mask")
        logprobs = meta.get("output_token_sampling_logprobs")
        if (
            not isinstance(supports, list)
            or not isinstance(logprobs, list)
            or len(supports) != len(response_tokens)
            or len(logprobs) != len(response_tokens)
        ):
            raise ValueError(
                "sampler did not return token-aligned sampling masks and sampling logprobs"
            )
        ids_list: list[int] = []
        offsets_list: list[int] = [0]
        for token, support in zip(response_tokens, supports, strict=True):
            if (
                not isinstance(support, list)
                or not support
                or any(type(item) is not int or item < 0 for item in support)
                or token not in support
                or len(set(support)) != len(support)
            ):
                raise ValueError(
                    "invalid sampling support: every emitted token must be in its mask"
                )
            ids_list.extend(support)
            offsets_list.append(len(ids_list))
        mask_ids = np.asarray(ids_list, dtype=np.int64)
        mask_offsets = np.asarray(offsets_list, dtype=np.int64)
        if plan.sampling_probabilities:
            flat_probs: list[float] = []
            chosen_probs: list[float] = []
            for token, support, row in zip(
                response_tokens, supports, logprobs, strict=True
            ):
                if (
                    not isinstance(row, list)
                    or len(row) != len(support)
                    or any(
                        not isinstance(val, (int, float))
                        or not math.isfinite(val)
                        or val > 0
                        for val in row
                    )
                    or not math.isclose(
                        sum(math.exp(val) for val in row), 1.0, abs_tol=1e-4
                    )
                ):
                    raise ValueError(
                        "sampling support logprobs must be a normalized distribution over the mask"
                    )
                flat_probs.extend(float(val) for val in row)
                chosen_probs.append(float(row[support.index(token)]))
            mask_logprobs = np.asarray(flat_probs, dtype=np.float32)
            sampling_logprobs = np.asarray(chosen_probs, dtype=np.float32)
        else:
            if any(
                not isinstance(val, (int, float)) or not math.isfinite(val) or val > 0
                for val in logprobs
            ):
                raise ValueError("invalid sampling logprobs")
            sampling_logprobs = np.asarray(
                [float(val) for val in logprobs], dtype=np.float32
            )

    sampler_draws: np.ndarray | None = None
    if plan.sampler_draws:
        raw_draws = meta.get("output_token_alt_tokens")
        if raw_draws is None:
            raw_draws = meta.get("sampler_draws")
        if raw_draws is not None:
            arr = np.asarray(raw_draws, dtype=np.int64)
            if (
                arr.ndim != 2
                or arr.shape[0] != len(response_tokens)
                or arr.shape[1] < 1
            ):
                raise ValueError(
                    "sampler_draws must have shape [response_tokens, num_draws]"
                )
            if arr.size and arr.min() < 0:
                raise ValueError("sampler_draws token IDs must be non-negative")
            sampler_draws = arr

    if strip_meta:
        for key in SGLANG_CAPTURE_META_KEYS:
            meta.pop(key, None)

    return CapturedSequence(
        sequence_id=sequence_id,
        session_id=session_id,
        created_at=time.time() if created_at is None else created_at,
        prompt_tokens=np.asarray(prompt_tokens, dtype=np.int64),
        response_tokens=np.asarray(response_tokens, dtype=np.int64),
        temperature=float(temperature),
        routed_experts=routed_experts,
        sampling_mask_ids=mask_ids,
        sampling_mask_offsets=mask_offsets,
        sampling_mask_logprobs=mask_logprobs,
        sampling_logprobs=sampling_logprobs,
        sampler_draws=sampler_draws,
    )
