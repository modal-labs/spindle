from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

import json
import pickle
from dataclasses import replace
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
from modal.types import FileEntryType
import torch
import torch.distributed.checkpoint as dcp

from spindle.backends.miles_runtime import native_checkpoint
from spindle.backends.miles_runtime.native_checkpoint import (
    materialize_adapter,
    validate_adapter,
)


def _checkpoint(path: Path, value: float = 1.0) -> dict[str, torch.Tensor]:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_parametrization.json").write_text(
        json.dumps({"format_version": 1})
    )
    state = {
        "adapter.weight": torch.tensor([value, value + 1]),
        "adapter.exp_avg": torch.tensor([0.125, 0.25]),
        "adapter.exp_avg_sq": torch.tensor([0.015625, 0.0625]),
        "optimizer.step": torch.tensor(2),
    }
    dcp.save(state, checkpoint_id=path / "adapter_state", no_dist=True)
    return state


@pytest.mark.parametrize("volume_configured", [False, True])
def test_native_capture_preserves_weights_moments_and_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, volume_configured: bool
) -> None:
    if volume_configured:
        monkeypatch.setenv("SPINDLE_CHECKPOINT_VOLUME", "spindle-checkpoints")
        monkeypatch.setenv("SPINDLE_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
        monkeypatch.setattr(native_checkpoint.modal.Volume, "from_name", None)
    source = tmp_path / ".capture.1234"
    expected = _checkpoint(source)
    target = tmp_path / "capture"
    materialize_adapter(source, target, checkpoint=True)
    restored = {key: torch.zeros_like(value) for key, value in expected.items()}
    dcp.load(restored, checkpoint_id=target / "adapter_state", no_dist=True)
    assert all(torch.equal(restored[key], value) for key, value in expected.items())
    assert not source.exists()


@pytest.mark.parametrize("damage", ["missing", "truncated", "metadata"])
def test_native_capture_rejects_incomplete_checkpoint(
    tmp_path: Path, damage: str
) -> None:
    source = tmp_path / ".capture.1234"
    _checkpoint(source)
    shard = next((source / "adapter_state").glob("*.distcp"))
    if damage == "missing":
        shard.unlink()
    elif damage == "truncated":
        shard.write_bytes(b"partial")
    else:
        (source / "adapter_state" / ".metadata").unlink()
    (source / "adapter_state" / "unrelated.distcp").write_bytes(b"unrelated")
    target = tmp_path / "capture"
    with pytest.raises((ValueError, FileNotFoundError)):
        materialize_adapter(source, target, checkpoint=True)
    assert not target.exists()
    assert source.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "missing", ["adapter_config.json", "adapter_model.safetensors"]
)
def test_sampler_requires_final_weights_and_config(
    tmp_path: Path, missing: str
) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / missing).unlink()
    (tmp_path / (missing + ".tmp")).write_bytes(b"incomplete")
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_adapter(tmp_path, checkpoint=False)


def test_concurrent_captures_keep_their_own_files(tmp_path: Path) -> None:
    sources = [tmp_path / f".capture{i}.1234" for i in range(2)]
    expected = [_checkpoint(source, float(i)) for i, source in enumerate(sources)]
    targets = [tmp_path / f"capture{i}" for i in range(2)]
    with ThreadPoolExecutor(2) as workers:
        futures = [
            workers.submit(materialize_adapter, source, target, checkpoint=True)
            for source, target in zip(sources, targets, strict=True)
        ]
        for future in futures:
            future.result()
    for target, state in zip(targets, expected, strict=True):
        restored = {key: torch.zeros_like(value) for key, value in state.items()}
        dcp.load(restored, checkpoint_id=target / "adapter_state", no_dist=True)
        assert all(torch.equal(restored[key], value) for key, value in state.items())


@pytest.mark.parametrize("missing_peer", [False, True])
@pytest.mark.parametrize("partial_local", [False, True])
def test_volume_assembly_requires_metadata_referenced_peer_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_peer: bool,
    partial_local: bool,
) -> None:
    source = tmp_path / ".capture.1234"
    _checkpoint(source)
    state_dir = source / "adapter_state"
    metadata = dcp.FileSystemReader(state_dir).read_metadata()
    original = next(state_dir.glob("*.distcp"))
    peer = state_dir / "peer.distcp"
    peer.write_bytes(original.read_bytes())
    key = next(iter(metadata.storage_data))
    metadata.storage_data[key] = replace(
        metadata.storage_data[key], relative_path=peer.name
    )
    (state_dir / ".metadata").write_bytes(pickle.dumps(metadata))
    files = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    if missing_peer:
        files.pop(str(peer.relative_to(tmp_path)))
    files[f"{source.name}/adapter_state/unrelated.distcp"] = b"unrelated"
    peer.unlink()
    if partial_local:
        original.write_bytes(b"incomplete")

    class Volume:
        def listdir(
            self, path: str, *, recursive: bool = False
        ) -> list[SimpleNamespace]:
            assert recursive
            return [
                SimpleNamespace(path=name, type=FileEntryType.FILE, size=len(data))
                for name, data in files.items()
            ]

        def read_file(self, path: str) -> Iterator[bytes]:
            local = tmp_path / path
            assert not local.is_file() or local.stat().st_size != len(files[path])
            yield files[path]

        def commit(self) -> None:
            pass

    monkeypatch.setenv("SPINDLE_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setenv("SPINDLE_CHECKPOINT_VOLUME", "checkpoints")
    monkeypatch.setattr(
        native_checkpoint.modal.Volume, "from_name", lambda name: Volume()
    )
    target = tmp_path / "capture"
    if missing_peer:
        with pytest.raises(ValueError, match=r"peer\.distcp"):
            materialize_adapter(source, target, checkpoint=True)
        assert not target.exists()
    else:
        materialize_adapter(source, target, checkpoint=True)
        validate_adapter(target, checkpoint=True)
        assert (target / "adapter_state" / "peer.distcp").is_file()


def test_native_metadata_rejects_incomplete_storage(tmp_path: Path) -> None:
    _checkpoint(tmp_path)
    state_dir = tmp_path / "adapter_state"
    metadata = dcp.FileSystemReader(state_dir).read_metadata()
    metadata.storage_data.pop(next(iter(metadata.storage_data)))
    (state_dir / ".metadata").write_bytes(pickle.dumps(metadata))
    with pytest.raises(ValueError, match="missing storage"):
        validate_adapter(tmp_path, checkpoint=True)
