import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from spindle.control_plane.records import SamplerArtifactRecord
from spindle.errors import SequenceConflict
from spindle.providers.modal.sampler_storage import ModalSamplerStorage


def artifact(**updates: object) -> SamplerArtifactRecord:
    return SamplerArtifactRecord(
        model_path="tinker://run:train:0/sampler_weights/eval",
        model_id="run",
        base_model="Qwen/Qwen3-8B",
        engine_definition_id="qwen3_8b",
        export_seq_id=1,
        publish_version=7,
        created_at=100,
    ).model_copy(update=updates)


def test_named_checkpoint_round_trip_and_immutable_retry(tmp_path: Path) -> None:
    async def run() -> None:
        volume = Mock()
        saved = artifact()
        storage = ModalSamplerStorage(volume, str(tmp_path))
        await storage.write(saved)
        assert (tmp_path / "sampler_weights/run:train:0/eval/metadata.json").is_file()
        # A different storage instance needs no original control-plane state.
        other = ModalSamplerStorage(volume, str(tmp_path))
        assert await other.read(saved.model_path) == saved
        await other.write(saved)
        with pytest.raises(SequenceConflict):
            await other.write(artifact(publish_version=8))
        assert await other.read(saved.model_path) == saved
        assert volume.commit.call_count == 2
        assert volume.reload.call_count == 5

    asyncio.run(run())


@pytest.mark.parametrize(
    "path",
    [
        "tinker://../sampler_weights/eval",
        "tinker://run/sampler_weights/..",
        "tinker://run/sampler_weights/bad\\name",
    ],
)
def test_checkpoint_path_rejects_traversal(tmp_path: Path, path: str) -> None:
    storage = ModalSamplerStorage(Mock(), str(tmp_path))
    with pytest.raises(ValueError):
        asyncio.run(storage.read(path))
    storage.volume.reload.assert_not_called()


def test_checkpoint_metadata_must_match_path_and_schema(tmp_path: Path) -> None:
    async def run() -> None:
        storage = ModalSamplerStorage(Mock(), str(tmp_path))
        saved = artifact()
        await storage.write(saved)
        metadata = tmp_path / "sampler_weights/run:train:0/eval/metadata.json"
        metadata.write_text(
            artifact(model_path="tinker://other/sampler_weights/eval").model_dump_json()
        )
        with pytest.raises(ValueError, match="does not match"):
            await storage.read(saved.model_path)
        metadata.write_text('{"schema_version": 2}')
        with pytest.raises(ValueError):
            await storage.read(saved.model_path)

    asyncio.run(run())


def test_checkpoint_commit_failure_can_be_retried(tmp_path: Path) -> None:
    async def run() -> None:
        volume = Mock()
        volume.commit.side_effect = [OSError("commit failed"), None]
        storage = ModalSamplerStorage(volume, str(tmp_path))
        saved = artifact()
        with pytest.raises(OSError, match="commit failed"):
            await storage.write(saved)
        await storage.write(saved)
        assert await storage.read(saved.model_path) == saved
        assert volume.commit.call_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("version", [7, 8])
def test_concurrent_publishers_cannot_replace_a_checkpoint(
    tmp_path: Path, version: int
) -> None:
    async def run() -> None:
        first, second = artifact(), artifact(publish_version=version)
        publishers = [ModalSamplerStorage(Mock(), str(tmp_path)) for _ in range(2)]
        results = await asyncio.gather(
            publishers[0].write(first),
            publishers[1].write(second),
            return_exceptions=True,
        )
        saved = await publishers[0].read(first.model_path)
        if version == 7:
            assert results == [None, None]
            assert saved == first
        else:
            assert sum(isinstance(result, SequenceConflict) for result in results) == 1
            assert sum(result is None for result in results) == 1
            assert saved in (first, second)

    asyncio.run(run())
