"""Resolve provenance spans on training Datums into replay loss_fn_inputs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from tinker.types._pydantic_types.datum import Datum as DatumModel
from tinker.types._pydantic_types.model_input import ModelInput
from tinker.types._pydantic_types.tensor_data import TensorData as TensorDataModel
from tinker.types.forward_backward_input import ForwardBackwardInput
from tinker.types.provenance_spans import PromptProvenanceSpan, SampledProvenanceSpan

from spindle.tensor_data import ArrayTensorData

from .codec import RECORDED_FIELDS, CapturedSequence
from .store import CaptureStore

ProvenanceSpan = PromptProvenanceSpan | SampledProvenanceSpan


class _TensorLike(Protocol):
    @property
    def data(self) -> object: ...


class _DatumLike(Protocol):
    @property
    def model_input(self) -> ModelInput: ...

    @property
    def loss_fn_inputs(self) -> Mapping[str, _TensorLike]: ...


MASK_REQUIRED_LOSSES = frozenset({"mis", "score_centered", "score_centered_mis"})
PROBS_REQUIRED_LOSSES = frozenset({"score_centered", "score_centered_mis"})
DRAWS_REQUIRED_LOSSES = frozenset({"stable_reinforce"})

ALL_CAPTURES = RECORDED_FIELDS
MEGATRON_CAPTURES = frozenset(
    {"sampling_mask", "sampling_probabilities", "sampler_draws"}
)


def datum_provenance_spans(
    datum: object,
) -> tuple[Sequence[ProvenanceSpan] | None, Sequence[ProvenanceSpan] | None]:
    model_spans = getattr(datum, "model_input_spans", None)
    loss_spans = getattr(datum, "loss_fn_input_spans", None)
    if (model_spans is None) != (loss_spans is None):
        raise ValueError(
            "provenance requires both model_input_spans and loss_fn_input_spans"
        )
    return model_spans, loss_spans


def payload_sequence_ids(payload: ForwardBackwardInput) -> tuple[str, ...]:
    """Unique sequence_ids referenced by provenance spans in a ForwardBackwardInput."""
    seen: dict[str, None] = {}
    for datum in payload.data:
        model_spans, loss_spans = datum_provenance_spans(datum)
        if model_spans is None or loss_spans is None:
            continue
        for span in (*model_spans, *loss_spans):
            seen.setdefault(span.sequence_id, None)
    return tuple(seen)


def required_fields_for_loss(loss_fn: str) -> frozenset[str]:
    if loss_fn in PROBS_REQUIRED_LOSSES:
        return frozenset({"sampling_mask", "sampling_probabilities"})
    if loss_fn in MASK_REQUIRED_LOSSES:
        return frozenset({"sampling_mask"})
    if loss_fn in DRAWS_REQUIRED_LOSSES:
        return frozenset({"sampler_draws"})
    return frozenset()


def fields_to_fetch(loss_fn: str, supported_captures: frozenset[str]) -> frozenset[str]:
    required = required_fields_for_loss(loss_fn)
    unsupported = required - supported_captures
    if unsupported:
        raise ValueError(
            f"loss {loss_fn!r} requires stability info {sorted(unsupported)}, "
            "which this trainer backend does not support"
        )
    wanted = set(supported_captures)
    if loss_fn not in PROBS_REQUIRED_LOSSES:
        wanted.discard("sampling_probabilities")
    if loss_fn not in DRAWS_REQUIRED_LOSSES:
        wanted.discard("sampler_draws")
    return frozenset(wanted)


async def fetch_and_assemble_payload(
    payload: ForwardBackwardInput,
    *,
    store: CaptureStore,
    session_id: str | None,
    supported_captures: frozenset[str] = ALL_CAPTURES,
    auto_apply_mask: bool = True,
) -> ForwardBackwardInput:
    """Fetch captured sequences referenced by provenance spans and inject tensors."""
    sequence_ids = payload_sequence_ids(payload)
    if not sequence_ids:
        return payload
    if not session_id:
        raise ValueError(
            "cannot resolve stability info without a session_id on the training model"
        )
    wanted = fields_to_fetch(str(payload.loss_fn), supported_captures)
    fetched = await asyncio.gather(
        *(
            store.read_sequence(session_id, seq_id, fields=wanted)
            for seq_id in sequence_ids
        )
    )
    sequences = dict(zip(sequence_ids, fetched, strict=True))
    return await asyncio.to_thread(
        assemble_payload,
        payload,
        sequences,
        session_id=session_id,
        supported_captures=supported_captures,
        auto_apply_mask=auto_apply_mask,
    )


def assemble_payload(
    payload: ForwardBackwardInput,
    sequences: Mapping[str, CapturedSequence | None],
    *,
    session_id: str | None = None,
    supported_captures: frozenset[str] = ALL_CAPTURES,
    auto_apply_mask: bool = True,
) -> ForwardBackwardInput:
    """Pure CPU assembly of fetched CapturedSequences into a ForwardBackwardInput."""
    loss_fn = str(payload.loss_fn)
    required = required_fields_for_loss(loss_fn)
    unsupported = required - supported_captures
    if unsupported:
        raise ValueError(
            f"loss {loss_fn!r} requires stability info {sorted(unsupported)}, "
            "which this trainer backend does not support"
        )

    new_data: list[object] = []
    temperatures: set[float] = set()
    for datum in payload.data:
        model_spans, loss_spans = datum_provenance_spans(datum)
        if model_spans is None or loss_spans is None:
            new_data.append(datum)
            continue
        assembled_datum, datum_temps = _assemble_datum(
            datum,
            model_spans,
            loss_spans,
            sequences,
            loss_fn=loss_fn,
            session_id=session_id,
            supported_captures=supported_captures,
            auto_apply_mask=auto_apply_mask,
        )
        new_data.append(assembled_datum)
        temperatures.update(datum_temps)

    loss_fn_config = (
        dict(payload.loss_fn_config) if payload.loss_fn_config is not None else None
    )
    if temperatures and (
        loss_fn_config is None or "sampling_temperature" not in loss_fn_config
    ):
        if len(temperatures) > 1:
            raise ValueError(
                "all sequences in one forward_backward call must share a sampling_temperature, "
                f"got {sorted(temperatures)}"
            )
        loss_fn_config = dict(loss_fn_config or {})
        loss_fn_config["sampling_temperature"] = next(iter(temperatures))

    return ForwardBackwardInput(
        data=new_data,  # ty: ignore[invalid-argument-type]
        loss_fn=payload.loss_fn,
        loss_fn_config=loss_fn_config,
    )


def _model_input_tokens(datum: _DatumLike) -> np.ndarray:
    parts: list[np.ndarray] = []
    for chunk in datum.model_input.chunks:
        if getattr(chunk, "type", None) != "encoded_text":
            raise ValueError(
                "stability provenance spans require encoded_text model_input chunks"
            )
        parts.append(np.asarray(chunk.tokens, dtype=np.int64).reshape(-1))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _tensor_to_numpy(tensor: _TensorLike) -> np.ndarray:
    if hasattr(tensor, "to_numpy"):
        return np.asarray(tensor.to_numpy()).reshape(-1)
    return np.asarray(tensor.data).reshape(-1)


def _int_tensor(values: np.ndarray, shape: list[int] | None = None) -> TensorDataModel:
    arr = np.asarray(values)
    if arr.dtype.kind not in "iu":
        arr = arr.astype(np.int64)
    return ArrayTensorData.from_array(
        arr,
        dtype="int64",
        shape=list(arr.shape) if shape is None else shape,
    )


def _float_tensor(
    values: np.ndarray, shape: list[int] | None = None
) -> TensorDataModel:
    arr = np.asarray(values, dtype=np.float32)
    return ArrayTensorData.from_array(
        arr,
        dtype="float32",
        shape=list(arr.shape) if shape is None else shape,
    )


def _require_sequence(
    sequences: Mapping[str, CapturedSequence | None],
    sequence_id: str,
    session_id: str | None,
) -> CapturedSequence:
    seq = sequences.get(sequence_id)
    if seq is None:
        raise ValueError(f"missing stability info for sequence {sequence_id!r}")
    if session_id is not None and seq.session_id != session_id:
        raise ValueError(
            f"sequence {sequence_id!r} does not belong to session {session_id!r}"
        )
    return seq


def _verify_spans_against_tokens(
    spans: Sequence[ProvenanceSpan],
    tokens: np.ndarray,
    sequences: Mapping[str, CapturedSequence | None],
    *,
    session_id: str | None,
    field_name: str,
) -> list[CapturedSequence]:
    total = sum(span.length for span in spans)
    if total != tokens.size:
        raise ValueError(
            f"{field_name} provenance spans cover {total} positions, "
            f"but datum has {tokens.size}"
        )
    resolved: list[CapturedSequence] = []
    pos = 0
    for span in spans:
        seq = _require_sequence(sequences, span.sequence_id, session_id)
        source = (
            seq.prompt_tokens
            if isinstance(span, PromptProvenanceSpan)
            else seq.response_tokens
        )
        end = span.offset + span.length
        if span.offset < 0 or end > source.size:
            raise ValueError(
                f"{field_name} span [{span.offset}:{end}) exceeds recorded length "
                f"{source.size} for sequence {span.sequence_id!r}"
            )
        expected = source[span.offset : end]
        actual = tokens[pos : pos + span.length]
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{field_name} tokens do not match recorded sequence {span.sequence_id!r}"
            )
        resolved.append(seq)
        pos += span.length
    return resolved


def _assemble_datum(
    datum: _DatumLike,
    model_spans: Sequence[ProvenanceSpan],
    loss_spans: Sequence[ProvenanceSpan],
    sequences: Mapping[str, CapturedSequence | None],
    *,
    loss_fn: str,
    session_id: str | None,
    supported_captures: frozenset[str],
    auto_apply_mask: bool = True,
) -> tuple[DatumModel, set[float]]:
    if not required_fields_for_loss(loss_fn) and any(
        sequences.get(span.sequence_id) is None for span in (*model_spans, *loss_spans)
    ):
        return (
            DatumModel.model_construct(
                model_input=datum.model_input,
                loss_fn_inputs=dict(datum.loss_fn_inputs),
            ),
            set(),
        )

    input_tokens = _model_input_tokens(datum)
    target_tensor = datum.loss_fn_inputs.get("target_tokens")
    if target_tensor is None:
        raise ValueError("stability provenance spans require target_tokens")
    target_tokens = _tensor_to_numpy(target_tensor).astype(np.int64, copy=False)

    model_seqs = _verify_spans_against_tokens(
        model_spans,
        input_tokens,
        sequences,
        session_id=session_id,
        field_name="model_input",
    )
    loss_seqs = _verify_spans_against_tokens(
        loss_spans,
        target_tokens,
        sequences,
        session_id=session_id,
        field_name="loss_fn_inputs",
    )

    inputs = dict(datum.loss_fn_inputs)
    temperatures: set[float] = set()

    if "routed_experts" in supported_captures and "routed_experts" not in inputs:
        has_any_routes = any(
            "routed_experts" in seq.recorded_fields for seq in model_seqs
        )
        if has_any_routes:
            route_parts: list[np.ndarray] = []
            for span, seq in zip(model_spans, model_seqs, strict=True):
                if seq.routed_experts is None:
                    raise ValueError(
                        f"sequence {seq.sequence_id!r} is missing recorded 'routed_experts'"
                    )
                start = (
                    span.offset
                    if isinstance(span, PromptProvenanceSpan)
                    else int(seq.prompt_tokens.size) + span.offset
                )
                end = start + span.length
                if end > seq.routed_experts.shape[0]:
                    raise ValueError(
                        f"routed_experts span [{start}:{end}) exceeds recorded length "
                        f"{seq.routed_experts.shape[0]} for sequence {seq.sequence_id!r}"
                    )
                route_parts.append(seq.routed_experts[start:end])
            routes = np.concatenate(route_parts, axis=0)
            inputs["routed_experts"] = _int_tensor(routes)

    sampled_loss_seqs = [
        seq
        for span, seq in zip(loss_spans, loss_seqs, strict=True)
        if isinstance(span, SampledProvenanceSpan)
    ]
    need_mask = loss_fn in MASK_REQUIRED_LOSSES
    need_probs = loss_fn in PROBS_REQUIRED_LOSSES
    should_apply_mask = (
        "sampling_mask" in supported_captures
        and "sampling_mask_ids" not in inputs
        and (
            need_mask
            or (
                auto_apply_mask
                and bool(sampled_loss_seqs)
                and any(
                    "sampling_mask" in seq.recorded_fields for seq in sampled_loss_seqs
                )
            )
        )
    )
    if should_apply_mask:
        id_parts: list[np.ndarray] = []
        offset_parts: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        prob_parts: list[np.ndarray] = []
        existing_logprobs = (
            _tensor_to_numpy(inputs["logprobs"]).astype(np.float32, copy=True)
            if "logprobs" in inputs
            else np.zeros(target_tokens.size, dtype=np.float32)
        )
        running = 0
        pos = 0
        for span, seq in zip(loss_spans, loss_seqs, strict=True):
            if isinstance(span, PromptProvenanceSpan):
                offset_parts.append(np.full(span.length, running, dtype=np.int64))
                pos += span.length
                continue
            if (
                seq.sampling_mask_ids is None
                or seq.sampling_mask_offsets is None
                or seq.sampling_logprobs is None
            ):
                raise ValueError(
                    f"sequence {seq.sequence_id!r} is missing recorded 'sampling_mask'"
                )
            if need_probs and seq.sampling_mask_logprobs is None:
                raise ValueError(
                    f"sequence {seq.sequence_id!r} is missing recorded 'sampling_probabilities'"
                )
            start = span.offset
            end = span.offset + span.length
            id_start = int(seq.sampling_mask_offsets[start])
            id_end = int(seq.sampling_mask_offsets[end])
            span_ids = seq.sampling_mask_ids[id_start:id_end]
            span_offsets = (
                seq.sampling_mask_offsets[start + 1 : end + 1] - id_start + running
            )
            id_parts.append(span_ids)
            offset_parts.append(span_offsets)
            if need_probs and seq.sampling_mask_logprobs is not None:
                prob_parts.append(seq.sampling_mask_logprobs[id_start:id_end])
            existing_logprobs[pos : pos + span.length] = seq.sampling_logprobs[
                start:end
            ]
            temperatures.add(seq.temperature or 1.0)
            running += int(span_ids.size)
            pos += span.length

        all_ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.int64)
        all_offsets = np.concatenate(offset_parts)
        inputs["sampling_mask_ids"] = _int_tensor(all_ids)
        inputs["sampling_mask_offsets"] = _int_tensor(all_offsets)
        inputs["logprobs"] = _float_tensor(existing_logprobs)
        if need_probs:
            all_probs = (
                np.concatenate(prob_parts)
                if prob_parts
                else np.empty(0, dtype=np.float32)
            )
            inputs["sampling_mask_logprobs"] = _float_tensor(all_probs)

    if loss_fn in DRAWS_REQUIRED_LOSSES and "sampler_draws" not in inputs:
        num_draws: int | None = None
        for seq in sampled_loss_seqs:
            if seq.sampler_draws is None:
                raise ValueError(
                    f"sequence {seq.sequence_id!r} is missing recorded 'sampler_draws'"
                )
            k = int(seq.sampler_draws.shape[1])
            if num_draws is not None and k != num_draws:
                raise ValueError(
                    f"inconsistent sampler_draws width ({num_draws} vs {k}) "
                    f"for sequence {seq.sequence_id!r}"
                )
            num_draws = k
        if num_draws is None:
            raise ValueError(
                "stable_reinforce requires at least one SampledProvenanceSpan"
            )
        draws = np.full((target_tokens.size, num_draws), -1, dtype=np.int64)
        pos = 0
        for span, seq in zip(loss_spans, loss_seqs, strict=True):
            if isinstance(span, SampledProvenanceSpan):
                assert seq.sampler_draws is not None
                draws[pos : pos + span.length] = seq.sampler_draws[
                    span.offset : span.offset + span.length
                ]
                temperatures.add(seq.temperature or 1.0)
            pos += span.length
        inputs["sampler_draws"] = _int_tensor(draws)

    return (
        DatumModel.model_construct(
            model_input=datum.model_input,
            loss_fn_inputs=inputs,
        ),
        temperatures,
    )
