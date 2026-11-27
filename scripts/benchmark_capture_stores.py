"""Benchmark stability capture stores (Modal Dict vs Modal v2 Volume) and codec.

Run:
    uv run --no-sync modal run scripts/benchmark_capture_stores.py

Measures:
  1. Local CPU codec packing/unpacking latency & payload sizes (8k and 64k GLM-Flash shapes).
  2. Cross-container read-after-write visibility for Dict, v2 Volume (mount), and v2 Volume (api).
  3. Concurrent write latency/throughput at 1, 16, and 128 writers across payload sizes
     (single-replica and multi-replica writer fanout).
  4. Trainer step-scale concurrent reads (16 and 128 sequences from a separate container).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import modal
import numpy as np

from spindle.providers.modal.capture_store import (
    DEFAULT_CAPTURE_MOUNT_ROOT,
    ModalDictCaptureStore,
    ModalVolumeCaptureStore,
    VolumeMode,
)
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from spindle.stability import (
    CapturedSequence,
    Compression,
    pack_capture,
    unpack_capture,
)

StoreVariant = Literal["dict", "volume_mount", "volume_api"]

BENCH_VOLUME_NAME = "spindle-capture-bench-v2"
app = modal.App("spindle-capture-store-bench")
bench_volume = modal.Volume.from_name(
    BENCH_VOLUME_NAME, create_if_missing=True, version=2
)
bench_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(*CORE_PACKAGES, TINKER_PACKAGE, STITCH_PACKAGE, "numpy>=1.26")
    .add_local_python_source("spindle", ignore=ignore_config_source)
)


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    prompt_tokens: int
    response_tokens: int
    with_routes: bool
    with_mask: bool
    with_probs: bool
    with_draws: bool
    num_layers: int = 45
    experts_per_token: int = 8
    num_experts: int = 288
    top_k: int = 64
    num_draws: int = 8


WORKLOADS: dict[str, WorkloadSpec] = {
    "8k_draws": WorkloadSpec(
        name="8k_draws",
        prompt_tokens=1024,
        response_tokens=7168,
        with_routes=False,
        with_mask=False,
        with_probs=False,
        with_draws=True,
    ),
    "8k_routes_9bit": WorkloadSpec(
        name="8k_routes_9bit",
        prompt_tokens=1024,
        response_tokens=7168,
        with_routes=True,
        with_mask=False,
        with_probs=False,
        with_draws=False,
    ),
    "8k_mask_probs_k64": WorkloadSpec(
        name="8k_mask_probs_k64",
        prompt_tokens=1024,
        response_tokens=7168,
        with_routes=False,
        with_mask=True,
        with_probs=True,
        with_draws=False,
    ),
    "8k_all": WorkloadSpec(
        name="8k_all",
        prompt_tokens=1024,
        response_tokens=7168,
        with_routes=True,
        with_mask=True,
        with_probs=True,
        with_draws=True,
    ),
    "64k_draws": WorkloadSpec(
        name="64k_draws",
        prompt_tokens=8192,
        response_tokens=57344,
        with_routes=False,
        with_mask=False,
        with_probs=False,
        with_draws=True,
    ),
    "64k_routes_9bit": WorkloadSpec(
        name="64k_routes_9bit",
        prompt_tokens=8192,
        response_tokens=57344,
        with_routes=True,
        with_mask=False,
        with_probs=False,
        with_draws=False,
    ),
    "64k_mask_probs_k64": WorkloadSpec(
        name="64k_mask_probs_k64",
        prompt_tokens=8192,
        response_tokens=57344,
        with_routes=False,
        with_mask=True,
        with_probs=True,
        with_draws=False,
    ),
}


def make_synthetic_sequence(
    spec: WorkloadSpec,
    *,
    session_id: str,
    sequence_id: str,
    seed: int = 0,
) -> CapturedSequence:
    """Build a realistic synthetic sequence (Zipf-ish expert routing & supports)."""
    rng = np.random.default_rng(seed)
    prompt = rng.integers(0, 150_000, size=spec.prompt_tokens, dtype=np.int64)
    response = rng.integers(0, 150_000, size=spec.response_tokens, dtype=np.int64)
    n_input = spec.prompt_tokens + spec.response_tokens - 1

    routes: np.ndarray | None = None
    if spec.with_routes:
        # Skewed expert IDs mimic real MoE routing entropy while spanning all 288 experts.
        raw = rng.integers(
            0,
            spec.num_experts,
            size=(n_input, spec.num_layers, spec.experts_per_token),
            dtype=np.int32,
        )
        raw[0, 0, 0] = spec.num_experts - 1
        routes = raw

    mask_ids: np.ndarray | None = None
    mask_offsets: np.ndarray | None = None
    mask_logprobs: np.ndarray | None = None
    sampling_logprobs: np.ndarray | None = None
    if spec.with_mask:
        # Average support ~50% of top_k after top-p truncation, bounded by top_k.
        lengths = rng.integers(
            max(1, spec.top_k // 4),
            spec.top_k + 1,
            size=spec.response_tokens,
            dtype=np.int64,
        )
        mask_offsets = np.empty(spec.response_tokens + 1, dtype=np.int64)
        mask_offsets[0] = 0
        np.cumsum(lengths, out=mask_offsets[1:])
        total_ids = int(mask_offsets[-1])
        mask_ids = rng.integers(0, 150_000, size=total_ids, dtype=np.int64)
        mask_ids[mask_offsets[:-1]] = response
        sampling_logprobs = np.full(spec.response_tokens, -0.5, dtype=np.float32)
        if spec.with_probs:
            mask_logprobs = -np.log(np.repeat(lengths.astype(np.float32), lengths))
            sampling_logprobs = mask_logprobs[mask_offsets[:-1]].copy()

    draws: np.ndarray | None = None
    if spec.with_draws:
        draws = rng.integers(
            0, 150_000, size=(spec.response_tokens, spec.num_draws), dtype=np.int64
        )

    return CapturedSequence(
        sequence_id=sequence_id,
        session_id=session_id,
        created_at=time.time(),
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


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "mean_ms": 0.0,
            "max_ms": 0.0,
        }
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
        return round(ordered[idx] * 1000.0, 2)

    return {
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "mean_ms": round(statistics.mean(ordered) * 1000.0, 2),
        "max_ms": round(ordered[-1] * 1000.0, 2),
    }


def _make_store(
    variant: StoreVariant,
    app_id: str,
    *,
    max_concurrency: int = 32,
) -> ModalDictCaptureStore | ModalVolumeCaptureStore:
    if variant == "dict":
        return ModalDictCaptureStore(app_id=app_id, max_concurrency=max_concurrency)
    mode: VolumeMode = "mount" if variant == "volume_mount" else "api"
    return ModalVolumeCaptureStore(
        bench_volume,
        app_id=app_id,
        mode=mode,
        mount_root=DEFAULT_CAPTURE_MOUNT_ROOT,
        max_concurrency=max_concurrency,
    )


def run_codec_benchmark() -> list[dict[str, Any]]:
    """Measure CPU pack/unpack latency and payload sizes across encodings."""
    rows: list[dict[str, Any]] = []
    for name in (
        "8k_draws",
        "8k_routes_9bit",
        "8k_mask_probs_k64",
        "8k_all",
        "64k_draws",
        "64k_routes_9bit",
        "64k_mask_probs_k64",
    ):
        spec = WORKLOADS[name]
        seq = make_synthetic_sequence(
            spec, session_id="codec", sequence_id=name, seed=7
        )
        raw_bytes = 0
        if seq.routed_experts is not None:
            raw_bytes += seq.routed_experts.size * 4  # SGLang int32 buffer
        if seq.sampling_mask_ids is not None:
            raw_bytes += seq.sampling_mask_ids.size * 8  # int64 ids
        if seq.sampling_mask_logprobs is not None:
            raw_bytes += seq.sampling_mask_logprobs.size * 4
        if seq.sampler_draws is not None:
            raw_bytes += seq.sampler_draws.size * 4

        for compression, bitpack in (("none", True), ("zstd", True)):
            comp: Compression = compression  # type: ignore[assignment]
            t0 = time.perf_counter()
            packed = pack_capture(
                seq,
                target_chunk_bytes=4 * 1024 * 1024,
                compression=comp,
                zstd_level=1,
                bitpack_routes=bitpack,
            )
            pack_ms = (time.perf_counter() - t0) * 1000.0

            chunk_map = dict(packed.chunks)
            t1 = time.perf_counter()
            restored = unpack_capture(packed.header, chunk_map)
            unpack_ms = (time.perf_counter() - t1) * 1000.0
            assert restored.sequence_id == seq.sequence_id

            packed_bytes = packed.total_chunk_bytes
            rows.append(
                {
                    "workload": name,
                    "compression": compression,
                    "raw_mib": round(raw_bytes / (1024 * 1024), 2),
                    "packed_mib": round(packed_bytes / (1024 * 1024), 2),
                    "ratio": round(raw_bytes / max(packed_bytes, 1), 2),
                    "num_chunks_4m": len(packed.chunks),
                    "pack_ms": round(pack_ms, 1),
                    "unpack_ms": round(unpack_ms, 1),
                }
            )
    return rows


def _make_fixed_size_sequence(
    session_id: str,
    sequence_id: str,
    payload_mib: float,
) -> CapturedSequence:
    """Create a sequence whose packed payload size is ~payload_mib."""
    # Routes with 45 layers x 8 experts at 9 bits => 405 bytes/token.
    # For small payloads (< 0.5 MiB), use draws (32 bytes/token).
    if payload_mib <= 0.5:
        n_resp = max(64, int((payload_mib * 1024 * 1024) / 32))
        spec = WorkloadSpec(
            name=f"{payload_mib}mib",
            prompt_tokens=256,
            response_tokens=n_resp,
            with_routes=False,
            with_mask=False,
            with_probs=False,
            with_draws=True,
        )
    else:
        n_tokens = max(64, int((payload_mib * 1024 * 1024) / 405))
        n_prompt = min(1024, n_tokens // 4)
        n_resp = max(64, n_tokens - n_prompt + 1)
        spec = WorkloadSpec(
            name=f"{payload_mib}mib",
            prompt_tokens=n_prompt,
            response_tokens=n_resp,
            with_routes=True,
            with_mask=False,
            with_probs=False,
            with_draws=False,
        )
    return make_synthetic_sequence(
        spec,
        session_id=session_id,
        sequence_id=sequence_id,
        seed=abs(hash(sequence_id)) % (2**31),
    )


@app.function(
    image=bench_image,
    volumes={DEFAULT_CAPTURE_MOUNT_ROOT: bench_volume},
    cpu=4,
    memory=8192,
    timeout=300,
)
async def write_batch_worker(
    variant: StoreVariant,
    app_id: str,
    session_id: str,
    sequence_ids: list[str],
    payload_mib: float,
    compression: Compression = "none",
) -> dict[str, Any]:
    """Write a batch of sequences concurrently from one container (simulating one replica)."""
    store = _make_store(variant, app_id, max_concurrency=32)
    chunk_bytes = 4 * 1024 * 1024 if variant == "dict" else None
    # Pre-build one template payload so we measure store I/O + packing, not RNG time.
    template = _make_fixed_size_sequence(session_id, sequence_ids[0], payload_mib)
    latencies: list[float] = []
    pack_times: list[float] = []
    write_times: list[float] = []
    errors: list[str] = []
    total_bytes = 0

    async def write_one(seq_id: str) -> None:
        nonlocal total_bytes
        seq = CapturedSequence(
            sequence_id=seq_id,
            session_id=session_id,
            created_at=time.time(),
            prompt_tokens=template.prompt_tokens,
            response_tokens=template.response_tokens,
            temperature=template.temperature,
            routed_experts=template.routed_experts,
            sampling_mask_ids=template.sampling_mask_ids,
            sampling_mask_offsets=template.sampling_mask_offsets,
            sampling_mask_logprobs=template.sampling_mask_logprobs,
            sampling_logprobs=template.sampling_logprobs,
            sampler_draws=template.sampler_draws,
        )
        t0 = time.perf_counter()
        try:
            stats = await store.write_sequence(
                seq, target_chunk_bytes=chunk_bytes, compression=compression
            )
            latencies.append(time.perf_counter() - t0)
            pack_times.append(stats.pack_seconds)
            write_times.append(stats.write_seconds)
            total_bytes += stats.total_chunk_bytes
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    wall_start = time.perf_counter()
    await asyncio.gather(*(write_one(seq_id) for seq_id in sequence_ids))
    wall_s = time.perf_counter() - wall_start
    return {
        "container": os.environ.get("MODAL_TASK_ID", "unknown"),
        "count": len(sequence_ids),
        "ok": len(latencies),
        "errors": errors,
        "total_mib": round(total_bytes / (1024 * 1024), 2),
        "wall_s": round(wall_s, 3),
        "throughput_mib_s": round((total_bytes / (1024 * 1024)) / max(wall_s, 1e-6), 2),
        "total_latency": _percentiles(latencies),
        "pack_latency": _percentiles(pack_times),
        "io_latency": _percentiles(write_times),
    }


@app.function(
    image=bench_image,
    volumes={DEFAULT_CAPTURE_MOUNT_ROOT: bench_volume},
    cpu=4,
    memory=16384,
    timeout=300,
)
async def read_batch_worker(
    variant: StoreVariant,
    app_id: str,
    session_id: str,
    sequence_ids: list[str],
    max_concurrency: int = 32,
) -> dict[str, Any]:
    """Read a batch of sequences concurrently from a fresh reader container (simulating trainer)."""
    store = _make_store(variant, app_id, max_concurrency=max_concurrency)
    sem = asyncio.Semaphore(max_concurrency)
    latencies: list[float] = []
    missing = 0
    errors: list[str] = []
    total_bytes = 0

    async def read_one(seq_id: str) -> None:
        nonlocal missing, total_bytes
        async with sem:
            t0 = time.perf_counter()
            try:
                got = await store.read_sequence(session_id, seq_id)
                elapsed = time.perf_counter() - t0
                if got is None:
                    missing += 1
                    return
                latencies.append(elapsed)
                if got.routed_experts is not None:
                    total_bytes += int(got.routed_experts.size * 9 // 8)
                elif got.sampler_draws is not None:
                    total_bytes += int(got.sampler_draws.size * 4)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

    wall_start = time.perf_counter()
    await asyncio.gather(*(read_one(seq_id) for seq_id in sequence_ids))
    wall_s = time.perf_counter() - wall_start
    return {
        "container": os.environ.get("MODAL_TASK_ID", "unknown"),
        "count": len(sequence_ids),
        "ok": len(latencies),
        "missing": missing,
        "errors": errors,
        "est_mib": round(total_bytes / (1024 * 1024), 2),
        "wall_s": round(wall_s, 3),
        "throughput_mib_s": round((total_bytes / (1024 * 1024)) / max(wall_s, 1e-6), 2),
        "latency": _percentiles(latencies),
    }


@app.function(
    image=bench_image,
    volumes={DEFAULT_CAPTURE_MOUNT_ROOT: bench_volume},
    timeout=300,
)
async def cleanup_bench_state(app_id: str, session_ids: list[str]) -> None:
    dict_store = ModalDictCaptureStore(app_id=app_id)
    vol_store = ModalVolumeCaptureStore(bench_volume, app_id=app_id, mode="api")
    await asyncio.gather(
        *(dict_store.delete_session(sid) for sid in session_ids),
        *(vol_store.delete_session(sid) for sid in session_ids),
    )


@app.local_entrypoint()
async def main(
    output_json: str = "/tmp/capture_store_benchmark_results.json",
    quick: bool = False,
) -> None:
    run_id = f"bench-{uuid.uuid4().hex[:8]}"
    print(f"=== Spindle Stability Capture Store Benchmark ({run_id}) ===")

    print("\n[1/4] Running local CPU codec benchmarks (GLM-Flash shapes)...")
    codec_results = run_codec_benchmark()
    for row in codec_results:
        print(
            f"  {row['workload']:<18} {row['compression']:<4} "
            f"raw={row['raw_mib']:6.2f} MiB -> packed={row['packed_mib']:6.2f} MiB "
            f"({row['ratio']:4.2f}x, {row['num_chunks_4m']:2d} chunks) "
            f"pack={row['pack_ms']:6.1f} ms unpack={row['unpack_ms']:6.1f} ms"
        )

    variants: tuple[StoreVariant, ...] = ("dict", "volume_mount", "volume_api")
    sessions_created: list[str] = []

    # Matrix of (payload_mib, writers_per_container, num_writer_containers)
    scenarios: list[tuple[str, float, int, int]] = (
        [
            ("1mib_1w", 1.0, 1, 1),
            ("4mib_16w_1c", 4.0, 16, 1),
            ("25mib_1w", 25.0, 1, 1),
        ]
        if quick
        else [
            ("0.25mib_1w", 0.25, 1, 1),
            ("1mib_1w", 1.0, 1, 1),
            ("4mib_1w", 4.0, 1, 1),
            ("25mib_1w", 25.0, 1, 1),
            ("1mib_16w_1c", 1.0, 16, 1),
            ("4mib_16w_1c", 4.0, 16, 1),
            ("25mib_16w_1c", 25.0, 16, 1),
            ("4mib_16w_4c", 4.0, 4, 4),
            ("1mib_128w_8c", 1.0, 16, 8),
            ("4mib_128w_8c", 4.0, 16, 8),
            ("25mib_128w_8c", 25.0, 16, 8),
        ]
    )

    store_results: list[dict[str, Any]] = []
    visibility_results: dict[str, dict[str, Any]] = {}

    try:
        print(
            "\n[2/4] Verifying cross-container immediate read-after-write visibility..."
        )
        for variant in variants:
            sid = f"{run_id}-vis-{variant}"
            sessions_created.append(sid)
            seq_id = f"{sid}:0"
            w = await write_batch_worker.remote.aio(
                variant, run_id, sid, [seq_id], 4.0, "none"
            )
            r = await read_batch_worker.remote.aio(variant, run_id, sid, [seq_id], 1)
            ok = w["ok"] == 1 and r["ok"] == 1 and r["missing"] == 0
            visibility_results[variant] = {
                "visible_immediately": ok,
                "writer_container": w["container"],
                "reader_container": r["container"],
                "different_containers": w["container"] != r["container"],
                "write_ms": w["io_latency"]["p50_ms"],
                "read_ms": r["latency"]["p50_ms"],
            }
            print(
                f"  {variant:<13} visible={ok} diff_containers={w['container'] != r['container']} "
                f"write_io={w['io_latency']['p50_ms']}ms read={r['latency']['p50_ms']}ms"
            )

        print("\n[3/4] Running write + cross-container step-read matrix...")
        scenario_timeout_s = 90.0
        for label, payload_mib, per_container, num_containers in scenarios:
            total_seqs = per_container * num_containers
            for variant in variants:
                sid = f"{run_id}-{label}-{variant}"
                sessions_created.append(sid)
                batches = [
                    [f"{sid}:c{c}:s{i}" for i in range(per_container)]
                    for c in range(num_containers)
                ]
                all_ids = [seq_id for batch in batches for seq_id in batch]

                w_wall_t0 = time.perf_counter()
                try:
                    async with asyncio.timeout(scenario_timeout_s):
                        w_outputs = await asyncio.gather(
                            *(
                                write_batch_worker.remote.aio(
                                    variant, run_id, sid, batch, payload_mib, "none"
                                )
                                for batch in batches
                            )
                        )
                except TimeoutError:
                    w_wall_s = time.perf_counter() - w_wall_t0
                    row = {
                        "scenario": label,
                        "variant": variant,
                        "payload_mib": payload_mib,
                        "total_sequences": total_seqs,
                        "writer_containers": num_containers,
                        "writers_per_container": per_container,
                        "timed_out": "write",
                        "write_wall_s": round(w_wall_s, 3),
                    }
                    store_results.append(row)
                    print(
                        f"  {label:<15} {variant:<13} W: TIMED OUT after {w_wall_s:.1f}s"
                    )
                    continue
                w_wall_s = time.perf_counter() - w_wall_t0

                w_ok = sum(o["ok"] for o in w_outputs)
                w_errors = [e for o in w_outputs for e in o["errors"]]
                w_mib = sum(o["total_mib"] for o in w_outputs)
                w_io_p50 = round(
                    statistics.median(o["io_latency"]["p50_ms"] for o in w_outputs), 2
                )
                w_io_p95 = round(max(o["io_latency"]["p95_ms"] for o in w_outputs), 2)

                # Now read ALL written sequences concurrently from a trainer container.
                try:
                    async with asyncio.timeout(scenario_timeout_s):
                        r_out = await read_batch_worker.remote.aio(
                            variant, run_id, sid, all_ids, min(64, max(1, total_seqs))
                        )
                except TimeoutError:
                    row = {
                        "scenario": label,
                        "variant": variant,
                        "payload_mib": payload_mib,
                        "total_sequences": total_seqs,
                        "writer_containers": num_containers,
                        "writers_per_container": per_container,
                        "timed_out": "read",
                        "write_ok": w_ok,
                        "write_errors": len(w_errors),
                        "write_total_mib": round(w_mib, 2),
                        "write_wall_s": round(w_wall_s, 3),
                        "write_agg_mib_s": round(w_mib / max(w_wall_s, 1e-6), 2),
                        "write_io_p50_ms": w_io_p50,
                        "write_io_p95_ms": w_io_p95,
                    }
                    store_results.append(row)
                    print(
                        f"  {label:<15} {variant:<13} "
                        f"W: {w_ok}/{total_seqs} wall={w_wall_s:5.2f}s | R: TIMED OUT after {scenario_timeout_s:.0f}s"
                    )
                    continue

                row = {
                    "scenario": label,
                    "variant": variant,
                    "payload_mib": payload_mib,
                    "total_sequences": total_seqs,
                    "writer_containers": num_containers,
                    "writers_per_container": per_container,
                    "write_ok": w_ok,
                    "write_errors": len(w_errors),
                    "write_error_sample": w_errors[:2],
                    "write_total_mib": round(w_mib, 2),
                    "write_wall_s": round(w_wall_s, 3),
                    "write_agg_mib_s": round(w_mib / max(w_wall_s, 1e-6), 2),
                    "write_io_p50_ms": w_io_p50,
                    "write_io_p95_ms": w_io_p95,
                    "read_ok": r_out["ok"],
                    "read_missing": r_out["missing"],
                    "read_errors": len(r_out["errors"]),
                    "read_error_sample": r_out["errors"][:2],
                    "read_wall_s": r_out["wall_s"],
                    "read_agg_mib_s": r_out["throughput_mib_s"],
                    "read_p50_ms": r_out["latency"]["p50_ms"],
                    "read_p95_ms": r_out["latency"]["p95_ms"],
                }
                store_results.append(row)
                print(
                    f"  {label:<15} {variant:<13} "
                    f"W: {w_ok}/{total_seqs} err={len(w_errors)} wall={w_wall_s:5.2f}s "
                    f"({row['write_agg_mib_s']:6.1f} MiB/s, p50={w_io_p50:6.0f}ms, p95={w_io_p95:6.0f}ms) | "
                    f"R: {r_out['ok']}/{total_seqs} miss={r_out['missing']} wall={r_out['wall_s']:5.2f}s "
                    f"({r_out['throughput_mib_s']:6.1f} MiB/s, p50={r_out['latency']['p50_ms']:6.0f}ms)"
                )
    finally:
        print("\n[4/4] Cleaning up benchmark Dicts and Volume paths...")
        if sessions_created:
            await cleanup_bench_state.remote.aio(run_id, sessions_created)
        try:
            await modal.Volume.objects.delete.aio(BENCH_VOLUME_NAME, allow_missing=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: could not delete volume {BENCH_VOLUME_NAME}: {exc}")

    report = {
        "run_id": run_id,
        "codec": codec_results,
        "visibility": visibility_results,
        "store_matrix": store_results,
    }
    await asyncio.to_thread(
        Path(output_json).write_text, json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nSaved full benchmark results to {output_json}")
