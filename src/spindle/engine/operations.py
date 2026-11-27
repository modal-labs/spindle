from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tinker import AdamParams, LoraConfig
from tinker.types._pydantic_types.datum import Datum as DatumModel
from tinker.types._pydantic_types.forward_backward_input import (
    ForwardBackwardInput as TinkerForwardBackwardInputModel,
)
from tinker.types._pydantic_types.tensor_data import TensorData as TensorDataModel
from tinker.types.forward_backward_input import ForwardBackwardInput
from tinker.types.provenance_spans import PromptProvenanceSpan, SampledProvenanceSpan

from spindle.backends.contract import LossFn, ModelSpec

from .api import OperationKind

log = logging.getLogger(__name__)

ProvenanceSpan = PromptProvenanceSpan | SampledProvenanceSpan


def _coerce_span(item: object) -> ProvenanceSpan:
    if isinstance(item, (PromptProvenanceSpan, SampledProvenanceSpan)):
        return item
    if isinstance(item, Mapping):
        if "prompt_tokens" in item and isinstance(item["prompt_tokens"], Mapping):
            inner = item["prompt_tokens"]
            return PromptProvenanceSpan(
                sequence_id=str(inner["sequence_id"]),
                offset=int(inner.get("offset", 0)),
                length=int(inner["length"]),
            )
        if "sampled_tokens" in item and isinstance(item["sampled_tokens"], Mapping):
            inner = item["sampled_tokens"]
            return SampledProvenanceSpan(
                sequence_id=str(inner["sequence_id"]),
                offset=int(inner.get("offset", 0)),
                length=int(inner["length"]),
            )
        kind = item.get("type") or item.get("kind")
        if kind in {"prompt", "prompt_tokens"}:
            return PromptProvenanceSpan(
                sequence_id=str(item["sequence_id"]),
                offset=int(item.get("offset", 0)),
                length=int(item["length"]),
            )
        if kind in {"sampled", "sampled_tokens"}:
            return SampledProvenanceSpan(
                sequence_id=str(item["sequence_id"]),
                offset=int(item.get("offset", 0)),
                length=int(item["length"]),
            )
    raise ValueError(f"invalid provenance span: {item!r}")


def _serialize_span(span: ProvenanceSpan) -> dict[str, object]:
    key = (
        "prompt_tokens" if isinstance(span, PromptProvenanceSpan) else "sampled_tokens"
    )
    return {
        key: {
            "sequence_id": span.sequence_id,
            "offset": span.offset,
            "length": span.length,
        }
    }


class DatumWithProvenance(DatumModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_input_spans: list[ProvenanceSpan] | None = None
    loss_fn_input_spans: list[ProvenanceSpan] | None = None

    @field_validator("model_input_spans", "loss_fn_input_spans", mode="before")
    @classmethod
    def _parse_spans(cls, value: object) -> list[ProvenanceSpan] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise ValueError("provenance spans must be a list")
        return [_coerce_span(item) for item in value]

    @model_validator(mode="after")
    def _paired_spans(self) -> DatumWithProvenance:
        if (self.model_input_spans is None) != (self.loss_fn_input_spans is None):
            raise ValueError(
                "provenance requires both model_input_spans and loss_fn_input_spans"
            )
        return self


class ForwardBackwardInputModel(TinkerForwardBackwardInputModel):
    data: list[DatumWithProvenance]  # type: ignore[assignment]
    loss_fn: LossFn  # type: ignore[assignment]


class OperationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extensions: dict[str, Any] = Field(default_factory=dict)


class LoadCheckpointPayload(OperationPayloadModel):
    uri: str
    restore_optimizer: bool = False


class SaveCheckpointPayload(OperationPayloadModel):
    destination: str = "latest"
    include_optimizer: bool = True
    overwrite: bool = False


class SaveWeightsForSamplerPayload(OperationPayloadModel):
    publish_version: int = Field(gt=0)


class SkipPayload(OperationPayloadModel):
    error: str


type OperationPayload = (
    ForwardBackwardInput
    | AdamParams
    | LoadCheckpointPayload
    | SaveCheckpointPayload
    | SaveWeightsForSamplerPayload
    | SkipPayload
)

_PAYLOAD_TYPES = {
    OperationKind.FORWARD: ForwardBackwardInput,
    OperationKind.FORWARD_BACKWARD: ForwardBackwardInput,
    OperationKind.OPTIM_STEP: AdamParams,
    OperationKind.LOAD_WEIGHTS: LoadCheckpointPayload,
    OperationKind.SAVE_WEIGHTS: SaveCheckpointPayload,
    OperationKind.SAVE_WEIGHTS_FOR_SAMPLER: SaveWeightsForSamplerPayload,
}
_LORA_FIELDS = {
    "rank",
    "seed",
    "train_unembed",
    "train_mlp",
    "train_attn",
}


def parse_operation_payload(
    kind: OperationKind,
    value: object,
) -> OperationPayload:
    expected = _PAYLOAD_TYPES[kind]
    if isinstance(value, expected):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind.value} payload must be an object")

    raw = dict(value)
    if kind in {OperationKind.FORWARD, OperationKind.FORWARD_BACKWARD}:
        parsed = ForwardBackwardInputModel.model_validate(
            {
                "data": raw.get("data", ()),
                "loss_fn": raw.get("loss_fn"),
                "loss_fn_config": raw.get("loss_fn_config"),
            }
        )
        return ForwardBackwardInput(
            data=list(parsed.data),  # ty: ignore[invalid-argument-type]
            loss_fn=parsed.loss_fn,  # ty: ignore[invalid-argument-type]
            loss_fn_config=parsed.loss_fn_config,
        )
    if kind == OperationKind.OPTIM_STEP:
        _reject_non_adam_params(raw)
        return AdamParams.model_validate(raw.get("adam_params") or {})
    if kind == OperationKind.LOAD_WEIGHTS:
        known = {
            "path",
            "uri",
            "optimizer",
            "restore_optimizer",
            "extensions",
        }
        restore_optimizer = bool(
            raw.get("restore_optimizer", raw.get("optimizer", False))
        )
        return LoadCheckpointPayload(
            uri=raw.get("path") or raw.get("uri"),  # ty: ignore[invalid-argument-type]
            restore_optimizer=restore_optimizer,
            extensions=_extensions(raw, known),
        )
    if kind == OperationKind.SAVE_WEIGHTS:
        known = {
            "checkpoint_id",
            "name",
            "path",
            "destination",
            "include_optimizer",
            "overwrite",
            "extensions",
        }
        destination = next(
            (
                raw[key]
                for key in ("path", "destination", "checkpoint_id", "name")
                if key in raw and raw[key] is not None
            ),
            "latest",
        )
        if (
            not isinstance(destination, str)
            or not destination
            or destination != destination.strip()
            or destination in {".", ".."}
            or "/" in destination
            or "\\" in destination
            or "\0" in destination
        ):
            raise ValueError("checkpoint name must be a single path component")
        return SaveCheckpointPayload(
            destination=destination,
            include_optimizer=bool(raw.get("include_optimizer", True)),
            overwrite=bool(raw.get("overwrite", False)),
            extensions=_extensions(raw, known),
        )
    if kind == OperationKind.SAVE_WEIGHTS_FOR_SAMPLER:
        known = {"publish_version", "extensions"}
        return SaveWeightsForSamplerPayload(
            publish_version=raw.get("publish_version"),  # ty: ignore[invalid-argument-type]
            extensions=_extensions(raw, known),
        )
    raise ValueError(f"unsupported operation: {kind.value}")


def _reject_non_adam_params(raw: Mapping[str, Any]) -> None:
    optimizer_params = raw.get("optimizer_params")
    if optimizer_params is not None:
        family = (
            optimizer_params.get("type")
            if isinstance(optimizer_params, Mapping)
            else None
        )
        log.warning("rejected unsupported optimizer: %s", family or "unknown")
        raise ValueError(
            f"optimizer family {family or 'unknown'} is not supported; "
            "optim_step accepts only adam_params"
        )
    if raw.get("adam_params") is None:
        other = sorted(
            key
            for key, value in raw.items()
            if key.endswith("_params") and value is not None
        )
        if other:
            log.warning("rejected unsupported optimizer: %s", ",".join(other))
            raise ValueError(
                f"optim_step parameters {other} are not supported; "
                "optim_step accepts only adam_params"
            )


def serialize_operation_payload(payload: OperationPayload) -> dict[str, Any]:
    if isinstance(payload, ForwardBackwardInput):
        dumped = ForwardBackwardInputModel.model_construct(
            data=[
                DatumModel.model_construct(
                    model_input=datum.model_input,
                    loss_fn_inputs={
                        key: TensorDataModel(
                            data=tensor.data,
                            dtype=tensor.dtype,
                            shape=tensor.shape,
                            sparse_crow_indices=tensor.sparse_crow_indices,
                            sparse_col_indices=tensor.sparse_col_indices,
                        )
                        for key, tensor in datum.loss_fn_inputs.items()
                    },
                )
                for datum in payload.data
            ],
            loss_fn=payload.loss_fn,
            loss_fn_config=payload.loss_fn_config,
        ).model_dump(
            mode="json",
            exclude_defaults=True,
        )
        for serialized_datum, datum in zip(
            dumped.get("data", ()), payload.data, strict=True
        ):
            model_spans = getattr(datum, "model_input_spans", None)
            loss_spans = getattr(datum, "loss_fn_input_spans", None)
            if model_spans is not None:
                serialized_datum["model_input_spans"] = [
                    _serialize_span(span) for span in model_spans
                ]
            if loss_spans is not None:
                serialized_datum["loss_fn_input_spans"] = [
                    _serialize_span(span) for span in loss_spans
                ]
        return dumped
    if isinstance(payload, AdamParams):
        return {"adam_params": payload.model_dump(mode="json", exclude_defaults=True)}
    return payload.model_dump(mode="json", exclude_defaults=True)


def parse_model_spec(value: object) -> ModelSpec:
    if isinstance(value, ModelSpec):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("model spec must be an object")

    raw_parameterization = value.get("parameterization")
    parameterization = (
        raw_parameterization.get("type")
        if isinstance(raw_parameterization, Mapping)
        else raw_parameterization
    )
    raw_lora = value.get("lora_config")
    if parameterization is None and raw_lora is not None:
        parameterization = "lora"
    if parameterization not in {"lora", "full"}:
        raise ValueError("parameterization must be lora or full")

    base_model = value.get("base_model")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("base_model is required")

    if isinstance(raw_lora, LoraConfig):
        lora = raw_lora
    elif isinstance(raw_lora, Mapping):
        lora = LoraConfig.model_validate(
            {key: raw_lora[key] for key in _LORA_FIELDS if key in raw_lora}
        )
    elif raw_lora is None:
        lora = None
    else:
        raise ValueError("lora_config must be an object")
    return ModelSpec(
        base_model=base_model,
        parameterization=parameterization,
        lora_config=lora,
    )


def _extensions(raw: dict[str, Any], known: set[str]) -> dict[str, Any]:
    extensions = raw.get("extensions") or {}
    if not isinstance(extensions, Mapping):
        raise ValueError("extensions must be an object")
    return {
        **dict(extensions),
        **{key: value for key, value in raw.items() if key not in known},
    }
