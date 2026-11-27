from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterable

import numpy as np
import pytest
import tinker
from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.types.forward_backward_input import ForwardBackwardInput
from tinker.types.forward_backward_request import ForwardBackwardRequest

from spindle.engine import Engine, FutureStatus
from spindle.engine.api import Command
from spindle.engine.ingress import decode_forward_backward
from spindle.stability import (
    MEGATRON_CAPTURES,
    CapturedSequence,
    Compression,
    InMemoryCaptureStore,
    WriteStats,
    assemble_payload,
)
from tests.support import EchoExecutor


def _sample_sequence(
    session_id: str,
    sequence_id: str,
    *,
    prompt: list[int],
    response: list[int],
    temperature: float = 0.8,
    with_routes: bool = True,
    with_mask: bool = True,
    with_probs: bool = True,
    with_draws: bool = True,
) -> CapturedSequence:
    n_in = len(prompt) + len(response) - 1
    routes = (
        np.arange(n_in * 2 * 4, dtype=np.int32).reshape(n_in, 2, 4) % 16
        if with_routes
        else None
    )
    mask_ids = []
    offsets = [0]
    mask_lps = []
    chosen_lps = []
    for tok in response:
        mask_ids.extend([tok, tok + 100])
        offsets.append(len(mask_ids))
        mask_lps.extend([math.log(0.75), math.log(0.25)])
        chosen_lps.append(math.log(0.75))
    return CapturedSequence(
        sequence_id=sequence_id,
        session_id=session_id,
        created_at=time.time(),
        prompt_tokens=np.asarray(prompt, dtype=np.int64),
        response_tokens=np.asarray(response, dtype=np.int64),
        temperature=temperature,
        routed_experts=routes,
        sampling_mask_ids=np.asarray(mask_ids, dtype=np.int64) if with_mask else None,
        sampling_mask_offsets=np.asarray(offsets, dtype=np.int64)
        if with_mask
        else None,
        sampling_mask_logprobs=np.asarray(mask_lps, dtype=np.float32)
        if with_probs
        else None,
        sampling_logprobs=np.asarray(chosen_lps, dtype=np.float32)
        if with_mask
        else None,
        sampler_draws=np.tile(np.asarray(response, dtype=np.int64)[:, None], (1, 8))
        if with_draws
        else None,
    )


def test_protobuf_provenance_spans_survive_ingress_decode() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 10]),
        loss_fn_inputs={"target_tokens": [2, 10, 11]},
    ).with_provenance(
        model_input=[
            tinker.PromptProvenanceSpan(sequence_id="req-1:0", offset=0, length=2),
            tinker.SampledProvenanceSpan(sequence_id="req-1:0", offset=0, length=1),
        ],
        loss_fn_inputs=[
            tinker.PromptProvenanceSpan(sequence_id="req-1:0", offset=1, length=1),
            tinker.SampledProvenanceSpan(sequence_id="req-1:0", offset=0, length=2),
        ],
    )
    req = ForwardBackwardRequest(
        model_id="m1",
        seq_id=1,
        forward_backward_input=ForwardBackwardInput(
            data=[datum], loss_fn="importance_sampling"
        ),
    )
    raw = forward_backward_request_to_proto(req).SerializeToString()
    _, _, _, decoded = decode_forward_backward(raw, "application/x-protobuf")
    assert decoded.data[0].model_input_spans == datum.model_input_spans
    assert decoded.data[0].loss_fn_input_spans == datum.loss_fn_input_spans


def test_assemble_validates_tokens_session_and_missing_required_fields() -> None:
    seq = _sample_sequence(
        "sess-1",
        "s1",
        prompt=[1, 2, 3],
        response=[10, 11],
        with_probs=False,
    )
    tokens = [1, 2, 3, 10, 11]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "advantages": [0.0, 0.0, 1.0, 1.0],
        },
    ).with_provenance(
        model_input=[
            tinker.PromptProvenanceSpan(sequence_id="s1", offset=0, length=3),
            tinker.SampledProvenanceSpan(sequence_id="s1", offset=0, length=1),
        ],
        loss_fn_inputs=[
            tinker.PromptProvenanceSpan(sequence_id="s1", offset=1, length=2),
            tinker.SampledProvenanceSpan(sequence_id="s1", offset=0, length=2),
        ],
    )
    with pytest.raises(ValueError, match=r"s1.*sampling_probabilities"):
        assemble_payload(
            ForwardBackwardInput(data=[datum], loss_fn="score_centered"),
            {"s1": seq},
            session_id="sess-1",
        )

    with pytest.raises(ValueError, match="does not belong to session"):
        assemble_payload(
            ForwardBackwardInput(data=[datum], loss_fn="mis"),
            {"s1": seq},
            session_id="sess-other",
        )

    bad_datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 999, 3, 10]),
        loss_fn_inputs={"target_tokens": tokens[1:]},
    ).with_provenance(
        model_input=datum.model_input_spans,  # type: ignore[arg-type]
        loss_fn_inputs=datum.loss_fn_input_spans,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="tokens do not match"):
        assemble_payload(
            ForwardBackwardInput(data=[bad_datum], loss_fn="mis"),
            {"s1": seq},
            session_id="sess-1",
        )

    megatron_payload = assemble_payload(
        ForwardBackwardInput(data=[datum], loss_fn="mis"),
        {"s1": seq},
        session_id="sess-1",
        supported_captures=MEGATRON_CAPTURES,
        auto_apply_mask=False,
    )
    assert "routed_experts" not in megatron_payload.data[0].loss_fn_inputs
    assert "sampling_mask_ids" in megatron_payload.data[0].loss_fn_inputs
    assert megatron_payload.loss_fn_config == {"sampling_temperature": 0.8}

    # On Megatron (auto_apply_mask=False), stock losses do not inject mask or sampling_temperature.
    megatron_stock = assemble_payload(
        ForwardBackwardInput(data=[datum], loss_fn="importance_sampling"),
        {"s1": seq},
        session_id="sess-1",
        supported_captures=MEGATRON_CAPTURES,
        auto_apply_mask=False,
    )
    assert "sampling_mask_ids" not in megatron_stock.data[0].loss_fn_inputs
    assert megatron_stock.loss_fn_config is None

    # Missing capture record is tolerated on non-replay losses, but fails on replay losses.
    missing_ce = assemble_payload(
        ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
        {"s1": None},
        session_id="sess-1",
    )
    assert "sampling_mask_ids" not in missing_ce.data[0].loss_fn_inputs
    with pytest.raises(ValueError, match="missing stability info"):
        assemble_payload(
            ForwardBackwardInput(data=[datum], loss_fn="mis"),
            {"s1": None},
            session_id="sess-1",
        )


def test_engine_background_fetch_does_not_block_other_models_and_updates_batch_key() -> (
    None
):
    async def scenario() -> None:
        fetch_started = asyncio.Event()
        release_fetch = asyncio.Event()
        base_store = InMemoryCaptureStore()
        seq = _sample_sequence("sess-a", "seq-a:0", prompt=[1, 2], response=[10, 11])
        await base_store.write_sequence(seq)

        class _SlowStore:
            async def write_sequence(
                self,
                sequence: CapturedSequence,
                *,
                target_chunk_bytes: int | None = None,
                compression: Compression = "none",
            ) -> WriteStats:
                return await base_store.write_sequence(
                    sequence,
                    target_chunk_bytes=target_chunk_bytes,
                    compression=compression,
                )

            async def read_sequence(
                self,
                session_id: str,
                sequence_id: str,
                *,
                fields: Iterable[str] | None = None,
            ) -> CapturedSequence | None:
                fetch_started.set()
                await release_fetch.wait()
                return await base_store.read_sequence(
                    session_id, sequence_id, fields=fields
                )

            async def delete_session(self, session_id: str) -> None:
                await base_store.delete_session(session_id)

            async def expire_before(self, cutoff: float) -> int:
                return await base_store.expire_before(cutoff)

        executed_batches: list[list[tuple[str, bool]]] = []

        class _RecordingExecutor(EchoExecutor):
            async def execute_forward_backward_batch(
                self, executions: tuple[Command, ...]
            ) -> tuple[object, ...]:
                executed_batches.append(
                    [
                        (
                            cmd.model_id,
                            "routed_experts" in cmd.payload.data[0].loss_fn_inputs,
                        )
                        for cmd in executions
                    ]
                )
                return await super().execute_forward_backward_batch(executions)

        engine = Engine(_RecordingExecutor(), capture_store=_SlowStore())
        await engine.accept_model("model-a", {"session_id": "sess-a"})
        await engine.accept_model("model-b", {"session_id": "sess-b"})

        datum_a = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2, 10]),
            loss_fn_inputs={
                "target_tokens": [2, 10, 11],
                "advantages": [0.0, 1.0, 1.0],
            },
        ).with_provenance(
            model_input=[
                tinker.PromptProvenanceSpan(sequence_id="seq-a:0", offset=0, length=2),
                tinker.SampledProvenanceSpan(sequence_id="seq-a:0", offset=0, length=1),
            ],
            loss_fn_inputs=[
                tinker.PromptProvenanceSpan(sequence_id="seq-a:0", offset=1, length=1),
                tinker.SampledProvenanceSpan(sequence_id="seq-a:0", offset=0, length=2),
            ],
        )
        req_a = ForwardBackwardRequest(
            model_id="model-a",
            seq_id=1,
            forward_backward_input=ForwardBackwardInput(
                data=[datum_a], loss_fn="score_centered_mis"
            ),
        )
        req_b = ForwardBackwardRequest(
            model_id="model-b",
            seq_id=1,
            forward_backward_input=ForwardBackwardInput(
                data=[
                    types.Datum(
                        model_input=types.ModelInput.from_ints([5, 6]),
                        loss_fn_inputs={"target_tokens": [6, 7]},
                    )
                ],
                loss_fn="cross_entropy",
            ),
        )

        fut_a = await engine.forward_backward(
            forward_backward_request_to_proto(req_a).SerializeToString(),
            "application/x-protobuf",
        )
        await asyncio.wait_for(fetch_started.wait(), 1)

        fut_b = await engine.forward_backward(
            forward_backward_request_to_proto(req_b).SerializeToString(),
            "application/x-protobuf",
        )
        state_b = await engine.retrieve_future(fut_b, timeout=1)
        assert state_b is not None and state_b.status == FutureStatus.COMPLETE

        release_fetch.set()
        state_a = await engine.retrieve_future(fut_a, timeout=2)
        assert state_a is not None and state_a.status == FutureStatus.COMPLETE
        assert executed_batches == [[("model-b", False)], [("model-a", True)]]
        await engine.close()

    asyncio.run(scenario())


def test_multiturn_stitched_spans_and_unload_cancels_inflight_fetch() -> None:
    seq1 = _sample_sequence("sess-m", "turn-1:0", prompt=[1, 2], response=[10, 11])
    seq2 = _sample_sequence(
        "sess-m", "turn-2:0", prompt=[1, 2, 10, 11, 3], response=[20, 21]
    )
    tokens = [1, 2, 10, 11, 3, 20, 21]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "advantages": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        },
    ).with_provenance(
        model_input=[
            tinker.PromptProvenanceSpan(sequence_id="turn-1:0", offset=0, length=2),
            tinker.SampledProvenanceSpan(sequence_id="turn-1:0", offset=0, length=1),
            tinker.PromptProvenanceSpan(sequence_id="turn-2:0", offset=3, length=2),
            tinker.SampledProvenanceSpan(sequence_id="turn-2:0", offset=0, length=1),
        ],
        loss_fn_inputs=[
            tinker.PromptProvenanceSpan(sequence_id="turn-1:0", offset=1, length=1),
            tinker.SampledProvenanceSpan(sequence_id="turn-1:0", offset=0, length=2),
            tinker.PromptProvenanceSpan(sequence_id="turn-2:0", offset=4, length=1),
            tinker.SampledProvenanceSpan(sequence_id="turn-2:0", offset=0, length=2),
        ],
    )
    assembled = assemble_payload(
        ForwardBackwardInput(data=[datum], loss_fn="score_centered_mis"),
        {"turn-1:0": seq1, "turn-2:0": seq2},
        session_id="sess-m",
    )
    inputs = assembled.data[0].loss_fn_inputs
    assert inputs["routed_experts"].shape == [6, 2, 4]
    assert list(inputs["sampling_mask_offsets"].data) == [0, 0, 2, 4, 4, 6, 8]

    async def unload_scenario() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class _BlockingStore:
            async def write_sequence(
                self,
                sequence: CapturedSequence,
                *,
                target_chunk_bytes: int | None = None,
                compression: Compression = "none",
            ) -> WriteStats:
                raise NotImplementedError

            async def read_sequence(
                self,
                session_id: str,
                sequence_id: str,
                *,
                fields: Iterable[str] | None = None,
            ) -> CapturedSequence | None:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            async def delete_session(self, session_id: str) -> None:
                pass

            async def expire_before(self, cutoff: float) -> int:
                return 0

        engine = Engine(EchoExecutor(), capture_store=_BlockingStore())
        await engine.accept_model("m-cancel", {"session_id": "sess-m"})
        req = ForwardBackwardRequest(
            model_id="m-cancel",
            seq_id=1,
            forward_backward_input=ForwardBackwardInput(
                data=[datum], loss_fn="score_centered_mis"
            ),
        )
        await engine.forward_backward(
            forward_backward_request_to_proto(req).SerializeToString(),
            "application/x-protobuf",
        )
        await asyncio.wait_for(started.wait(), 1)
        await engine.unload_model("m-cancel")
        await asyncio.wait_for(cancelled.wait(), 1)
        await engine.close()

    asyncio.run(unload_scenario())
