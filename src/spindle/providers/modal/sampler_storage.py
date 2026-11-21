"""Durable named sampler checkpoints on the configured bulletin volume.

``tinker://<run>/sampler_weights/<name>`` resolves to
``/bulletin/sampler_weights/<run>/<name>/metadata.json``. The metadata pins
the existing bulletin snapshot; adapter tensors are not copied.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile

import modal

from spindle.control_plane.records import SamplerArtifactRecord
from spindle.control_plane.service import path_component
from spindle.errors import SequenceConflict

BULLETIN_ROOT = "/bulletin"


def sampler_checkpoint_directory(root: str, model_path: str) -> Path:
    parts = model_path.removeprefix("tinker://").split("/")
    if (
        not model_path.startswith("tinker://")
        or len(parts) != 3
        or parts[1] != "sampler_weights"
    ):
        raise FileNotFoundError(model_path)
    run = path_component(parts[0], "training_run_id")
    name = path_component(parts[2], "sampler checkpoint name")
    directory = Path(root) / "sampler_weights" / run / name
    if not directory.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("sampler checkpoint is outside configured storage")
    return directory


def _read(directory: Path) -> SamplerArtifactRecord:
    return SamplerArtifactRecord.model_validate_json(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )


def _install(directory: Path, artifact: SamplerArtifactRecord) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=directory.parent))
    try:
        (temporary / "metadata.json").write_text(
            json.dumps(artifact.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        try:
            os.rename(temporary, directory)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if _read(directory).model_dump(exclude={"telemetry_tags"}) != (
                artifact.model_dump(exclude={"telemetry_tags"})
            ):
                raise SequenceConflict(
                    artifact.model_id, artifact.export_seq_id
                ) from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class ModalSamplerStorage:
    def __init__(
        self,
        volume: modal.Volume,
        root: str = BULLETIN_ROOT,
        *,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self.volume = volume
        self.root = root
        self.lock = lock if lock is not None else asyncio.Lock()

    async def read(self, model_path: str) -> SamplerArtifactRecord:
        directory = sampler_checkpoint_directory(self.root, model_path)
        async with self.lock:
            await asyncio.to_thread(self.volume.reload)
            artifact = await asyncio.to_thread(_read, directory)
        if artifact.model_path != model_path:
            raise ValueError("sampler checkpoint metadata does not match its path")
        return artifact

    async def write(self, artifact: SamplerArtifactRecord) -> None:
        directory = sampler_checkpoint_directory(self.root, artifact.model_path)
        async with self.lock:
            await asyncio.to_thread(self.volume.reload)
            await asyncio.to_thread(_install, directory, artifact)
            await asyncio.to_thread(self.volume.commit)
