"""Assemble Miles native adapter directories before exposing a capture."""

from __future__ import annotations


import json
import os
import shutil
import uuid
from pathlib import Path

import modal
from modal.types import FileEntryType
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata


def validate_adapter(
    path: Path,
    *,
    checkpoint: bool,
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
) -> None:
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
        for key, shape in (expected_shapes or {}).items():
            for suffix in ("weight", "exp_avg", "exp_avg_sq"):
                tensor = metadata.state_dict_metadata.get(f"{key}.{suffix}")
                if tensor is None and suffix != "weight":
                    continue
                if not isinstance(tensor, TensorStorageMetadata) or tuple(
                    tensor.size
                ) != tuple(shape):
                    raise ValueError(
                        f"Miles checkpoint tensor shape does not match the slot: {key}.{suffix}"
                    )
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
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    root = Path(os.environ.get("SPINDLE_CHECKPOINT_ROOT", "/checkpoints"))
    name = os.environ.get("SPINDLE_CHECKPOINT_VOLUME")
    try:
        temporary.mkdir(parents=True)
        if name is None:
            shutil.copytree(source, temporary, dirs_exist_ok=True)
        else:
            relative = source.relative_to(root)
            volume = modal.Volume.from_name(name)
            for entry in volume.listdir(str(relative), recursive=True):
                if entry.type != FileEntryType.FILE:
                    continue
                child = Path(entry.path.lstrip("/")).relative_to(relative)
                destination = temporary / child
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as output:
                    for data in volume.read_file(entry.path):
                        output.write(data)
        validate_adapter(temporary, checkpoint=checkpoint)
        if target.exists():
            raise FileExistsError(f"capture already exists: {target}")
        os.rename(temporary, target)
        if name is not None:
            volume.commit()
        shutil.rmtree(source, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
