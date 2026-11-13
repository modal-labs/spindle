"""Assemble Miles native adapter directories before exposing a capture."""

from __future__ import annotations

import json
import os
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata


def validate_adapter(path: Path, *, checkpoint: bool) -> None:
    if checkpoint:
        fields = json.loads((path / "adapter_parametrization.json").read_text())
        if fields.get("format_version") != 1:
            raise ValueError("unsupported Miles adapter parametrization format")
        state = path / "adapter_state"
        metadata = FileSystemReader(state).read_metadata()
        if not metadata.storage_data:
            raise ValueError("Miles checkpoint contains no stored tensors")
        for key, tensor in metadata.state_dict_metadata.items():
            indices = (
                [MetadataIndex(key, chunk.offsets) for chunk in tensor.chunks]
                if isinstance(tensor, TensorStorageMetadata)
                else [MetadataIndex(key)]
            )
            if not indices or any(
                index not in metadata.storage_data for index in indices
            ):
                raise ValueError(f"Miles checkpoint is missing storage for {key}")
        for info in metadata.storage_data.values():
            if info.offset < 0 or info.length <= 0:
                raise ValueError("Miles checkpoint has an invalid shard byte range")
            relative = Path(info.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "Miles checkpoint references a shard outside adapter_state"
                )
            shard = state / relative
            if not shard.is_file() or shard.stat().st_size < info.offset + info.length:
                raise ValueError(
                    f"Miles checkpoint shard is missing or truncated: {relative}"
                )
    else:
        json.loads((path / "adapter_config.json").read_text())
        weights = path / "adapter_model.safetensors"
        if not weights.is_file() or weights.stat().st_size == 0:
            raise ValueError(
                "Miles sampler export is missing adapter_model.safetensors"
            )


def materialize_adapter(source: Path, target: Path, *, checkpoint: bool) -> None:
    source, target = Path(source), Path(target)
    if source.parent != target.parent or not source.name.startswith(f".{target.name}."):
        raise ValueError("Miles returned a capture outside its operation directory")
    validate_adapter(source, checkpoint=checkpoint)
    if target.exists():
        raise FileExistsError(f"capture already exists: {target}")
    os.rename(source, target)
