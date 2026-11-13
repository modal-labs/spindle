from __future__ import annotations

from pathlib import Path

import json
import pickle

import pytest
import torch
import torch.distributed.checkpoint as dcp

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


def test_native_capture_preserves_weights_moments_and_step(tmp_path: Path) -> None:
    source = tmp_path / ".capture.1234"
    expected = _checkpoint(source)
    target = tmp_path / "capture"
    materialize_adapter(source, target, checkpoint=True)
    restored = {key: torch.zeros_like(value) for key, value in expected.items()}
    dcp.load(restored, checkpoint_id=target / "adapter_state", no_dist=True)
    assert all(torch.equal(restored[key], value) for key, value in expected.items())
    assert not source.exists()


@pytest.mark.parametrize("damage", ["missing", "truncated", "metadata", "index"])
def test_native_capture_rejects_incomplete_checkpoint(
    tmp_path: Path, damage: str
) -> None:
    source = tmp_path / ".capture.1234"
    _checkpoint(source)
    state_dir = source / "adapter_state"
    shard = next(state_dir.glob("*.distcp"))
    if damage == "missing":
        shard.unlink()
    elif damage == "truncated":
        shard.write_bytes(b"partial")
    elif damage == "metadata":
        (state_dir / ".metadata").unlink()
    else:
        metadata = dcp.FileSystemReader(state_dir).read_metadata()
        metadata.storage_data.pop(next(iter(metadata.storage_data)))
        (state_dir / ".metadata").write_bytes(pickle.dumps(metadata))
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
