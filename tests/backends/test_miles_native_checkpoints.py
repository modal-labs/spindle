"""Native lifecycle boundaries: real CPU DCP, relocation, missing shards and Modal visibility."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from spindle.backends.miles_runtime import native_checkpoints as native
from spindle.backends.miles_runtime import volumes


class Commit:
    @staticmethod
    def lock_checkpoint_root(root):
        return (root / ".lock").open("w")

    @staticmethod
    def commit_adapter_dir(root, name, written, *, overwrite):
        assert not overwrite
        assert written.path == root / f".{name}.{written.commit_id}"
        (root / name).symlink_to(written.path.name, target_is_directory=True)


class Bridge:
    def __init__(self):
        self.weight = torch.tensor([2.0, 3.0])
        self.moment = torch.tensor([0.1, 0.2])
        self.step = torch.tensor(7)
        self.restored = False

    async def save_slot(self, slot, root, name, tenant_digest, parametrization_fields):
        assert tenant_digest is None
        path = Path(root) / f".{name}.abcd"
        path.mkdir()
        (path / "adapter_parametrization.json").write_text(
            json.dumps(parametrization_fields)
        )
        dcp.save(
            {"weight": self.weight, "adam": self.moment, "step": self.step},
            checkpoint_id=path / "adapter_state",
        )
        return SimpleNamespace(path=path, commit_id="abcd")

    async def export_slot(
        self, slot, root, name, tenant_digest, parametrization_fields
    ):
        path = Path(root) / f".{name}.abcd"
        path.mkdir()
        (path / "adapter_parametrization.json").write_text(
            json.dumps(parametrization_fields)
        )
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_bytes(b"weights-only")
        return SimpleNamespace(path=path, commit_id="abcd")

    async def load_slot(self, slot, rank, alpha, source=None):
        if source is None:
            return None
        fields = json.loads(
            (Path(source["path"]) / "adapter_parametrization.json").read_text()
        )
        if fields != source["parametrization_fields"]:
            return {"error": "parametrization mismatch"}
        if source["load_optimizer"]:
            state = {"weight": self.weight, "adam": self.moment, "step": self.step}
            dcp.load(state, checkpoint_id=Path(source["path"]) / "adapter_state")
        self.restored = True


def lifecycle(bridge):
    io = native.NativeCheckpoints.__new__(native.NativeCheckpoints)
    io.bridge = bridge
    io.commit = Commit
    io.args = SimpleNamespace(
        hf_checkpoint="base", lora_type="lora", experts_shared_outer_loras=False
    )
    io.read_parametrization = lambda path: SimpleNamespace(
        **json.loads((path / "adapter_parametrization.json").read_text())
    )
    io.base_fingerprint = "base-fingerprint"
    io.slots = {0: (2, 4.0)}
    return io


@pytest.mark.parametrize("optimizer", [False, True])
def test_capture_is_owned_and_relocatable(tmp_path, monkeypatch, optimizer):
    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    bridge = Bridge()
    io = lifecycle(bridge)
    target = tmp_path / "capture"
    asyncio.run(io.capture(0, str(target), include_optimizer=optimizer))
    assert target.is_dir() and not target.is_symlink()
    assert not (tmp_path / ".capture.abcd").exists()
    assert (target / "adapter_state").exists() == optimizer
    relocated = tmp_path / "persisted"
    target.rename(relocated)
    if optimizer:
        bridge.weight.zero_()
        bridge.moment.zero_()
        bridge.step.zero_()
    asyncio.run(
        io.load_slot(0, 2, 4.0, checkpoint=str(relocated), restore_optimizer=optimizer)
    )
    assert bridge.restored
    if optimizer:
        torch.testing.assert_close(
            bridge.weight, torch.tensor([2.0, 3.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            bridge.moment, torch.tensor([0.1, 0.2]), rtol=0, atol=0
        )
        assert bridge.step.item() == 7
    assert (relocated / "adapter_model.safetensors").exists() != optimizer


def test_missing_dcp_shard_prevents_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    io = lifecycle(Bridge())
    original = io.bridge.save_slot

    async def incomplete(*args, **kwargs):
        written = await original(*args, **kwargs)
        next(written.path.rglob("*.distcp")).unlink()
        return written

    io.bridge.save_slot = incomplete
    with pytest.raises(RuntimeError, match="missing referenced shards"):
        asyncio.run(io.capture(0, str(tmp_path / "capture"), include_optimizer=True))
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / ".capture.abcd").exists()


def test_source_rejection_keeps_slot_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    io = lifecycle(Bridge())
    asyncio.run(io.capture(0, str(tmp_path / "capture"), include_optimizer=True))
    with pytest.raises(ValueError, match="lora_rank"):
        asyncio.run(io.load_slot(0, 3, 6.0, checkpoint=str(tmp_path / "capture")))
    assert io.slots == {0: (2, 4.0)}
    assert not io.bridge.restored


@pytest.mark.parametrize("missing", [False, True])
def test_native_publication_checks_all_peer_files(tmp_path, monkeypatch, missing):
    path = tmp_path / ".capture.abcd"
    path.mkdir()
    (path / "local").write_bytes(b"local")
    monkeypatch.setenv("SPINDLE_CHECKPOINT_VOLUME", "ckpt")
    monkeypatch.setenv("SPINDLE_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setattr(volumes, "_native_file_inventory", lambda _: {"local", "peer"})
    events = []
    remote = set()

    class Volume:
        def listdir(self, relative, *, recursive):
            assert recursive and events == ["commit"]
            return [SimpleNamespace(path=f"{relative}/{name}") for name in remote]

    def commit(action):
        assert action == "commit"
        events.append(action)
        remote.update({"local"} if missing else {"local", "peer"})

    monkeypatch.setattr(volumes, "sync_checkpoint_volume", commit)
    monkeypatch.setattr(volumes.modal.Volume, "from_name", lambda _: Volume())
    if missing:
        with pytest.raises(RuntimeError, match="missing peer files"):
            volumes.publish_native_directory(path)
    else:
        volumes.publish_native_directory(path)
    assert not (
        path / "peer"
    ).exists()  # rank0's mount never stood in for the remote inventory


@pytest.mark.parametrize(
    ("peer_inventory", "remote_weights", "error"),
    [
        ("adapter_model.safetensors.partial", True, None),
        ("adapter_model.safetensors.partial", False, "adapter_model.safetensors"),
        ("peer.distcp", True, "peer.distcp"),
        ("peer.distcp.partial", True, "peer.distcp.partial"),
    ],
)
def test_native_publication_requires_final_atomic_export(
    tmp_path, monkeypatch, peer_inventory, remote_weights, error
):
    path = tmp_path / ".capture.abcd"
    path.mkdir()
    partial = path / "adapter_model.safetensors.partial"
    partial.write_bytes(b"complete weights")
    partial.replace(path / "adapter_model.safetensors")
    monkeypatch.setenv("SPINDLE_CHECKPOINT_VOLUME", "ckpt")
    monkeypatch.setenv("SPINDLE_CHECKPOINT_ROOT", str(tmp_path))
    # A peer's cached directory can retain the staging name after rank zero's rename.
    monkeypatch.setattr(
        volumes,
        "_native_file_inventory",
        lambda _: {"adapter_model.safetensors", peer_inventory},
    )
    durable = {}

    def commit(action):
        assert action == "commit"
        if remote_weights:
            durable["adapter_model.safetensors"] = (
                path / "adapter_model.safetensors"
            ).read_bytes()

    class Volume:
        def listdir(self, relative, *, recursive):
            assert recursive
            return [SimpleNamespace(path=f"{relative}/{name}") for name in durable]

    monkeypatch.setattr(volumes, "sync_checkpoint_volume", commit)
    monkeypatch.setattr(volumes.modal.Volume, "from_name", lambda _: Volume())
    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            volumes.publish_native_directory(path)
    else:
        volumes.publish_native_directory(path)
        assert durable == {"adapter_model.safetensors": b"complete weights"}
        assert not partial.exists()


def test_native_writer_error_never_publishes(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def writer(*args, **kwargs):
        yield SimpleNamespace(path=tmp_path)

    adapter_dir = SimpleNamespace(write_adapter_dir=writer)
    monkeypatch.setattr(
        volumes, "publish_native_directory", lambda _: events.append("published")
    )
    volumes.install_native_checkpoint_publish(adapter_dir)
    with pytest.raises(RuntimeError, match="failed write"):
        with adapter_dir.write_adapter_dir():
            raise RuntimeError("failed write")
    assert events == []


def test_detach_refuses_an_unrelated_directory(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    link = tmp_path / "capture"
    link.symlink_to(other.name, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unexpected physical"):
        native.detach_native_capture(link, other)
    assert link.is_symlink() and other.is_dir()


def test_actual_native_commit_can_detach_and_restore_dcp(tmp_path, monkeypatch):
    commit = pytest.importorskip(
        "miles.backends.training_utils.checkpoint.adapter.commit"
    )
    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    io = lifecycle(Bridge())
    io.commit = commit
    target = tmp_path / "capture"
    # Use a real native commit id (the native library validates ownership names).
    original = io.bridge.save_slot

    async def save(*args, **kwargs):
        written = await original(*args, **kwargs)
        token = "0123456789abcdef"
        physical = written.path.with_name(f".capture.{token}")
        written.path.rename(physical)
        return commit.UncommittedAdapterDir(physical, token)

    io.bridge.save_slot = save
    asyncio.run(io.capture(0, str(target), include_optimizer=True))
    io.bridge.weight.zero_()
    asyncio.run(io.load_slot(0, 2, 4.0, checkpoint=str(target)))
    assert not target.is_symlink()
    assert io.bridge.weight.tolist() == [2.0, 3.0]


@pytest.mark.parametrize("optimizer", [False, True])
def test_actual_native_writer_reader_round_trip(tmp_path, monkeypatch, optimizer):
    adapter = pytest.importorskip(
        "miles.backends.training_utils.checkpoint.adapter.adapter_dir"
    )
    from miles.backends.training_utils.checkpoint.adapter.adapter_state import (
        AdapterState,
        LocalShardSpec,
    )
    from miles.backends.training_utils.checkpoint.adapter.parametrization import (
        AdapterParametrization,
    )
    from miles.utils.lora.hf_lora_targets import HfLoraTargets

    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    target_module = "model.layers.0.self_attn.q_proj"
    targets = HfLoraTargets(attention=(target_module,), mlp=(), unembed=())
    params = AdapterParametrization(
        base_model="base",
        lora_rank=2,
        lora_alpha=4.0,
        lora_type="lora",
        target_modules=(target_module,),
        experts_shared_outer_loras=False,
        base_model_fingerprint="base-fingerprint",
    )

    class Members:
        bf16_adapter_key_style = "peft"
        head_rows = None

        def __init__(self):
            self.weights = [
                torch.arange(6).reshape(2, 3).float(),
                torch.arange(8).reshape(4, 2).float(),
            ]
            self.means = [torch.full_like(t, 0.125) for t in self.weights]
            self.squares = [torch.full_like(t, 0.25) for t in self.weights]
            self.step = 9
            self.finished = False

        def get_local_shard_specs(self, slot):
            return [
                LocalShardSpec(
                    f"{target_module}.{factor}", tuple(t.shape), (0, 0), tuple(t.shape)
                )
                for factor, t in zip(("lora_A", "lora_B"), self.weights, strict=True)
            ]

        def export_adapter_state(self, slot):
            return AdapterState(
                self.get_local_shard_specs(slot),
                self.weights,
                self.means,
                self.squares,
                self.step,
            )

        def export_bf16_adapter(self, slot):
            return [
                (f"{target_module}.{factor}.weight", t.bfloat16())
                for factor, t in zip(("lora_A", "lora_B"), self.weights, strict=True)
            ]

        def import_adapter_state(self, slot, state):
            self.weights = [t.clone() for t in state.weights]
            self.means = [t.clone() for t in state.exp_avgs]
            self.squares = [t.clone() for t in state.exp_avg_sqs]
            self.step = state.step

        def finish_import_adapter_state(self, slot):
            self.finished = True

    members = Members()
    expected = [
        [t.clone() for t in tensors]
        for tensors in (members.weights, members.means, members.squares)
    ]

    class NativeBridge:
        async def save_slot(
            self, slot, root, name, tenant_digest, parametrization_fields
        ):
            return adapter.save_adapter_dir(
                Path(root), name, members, slot, params, tenant_digest=tenant_digest
            )

        async def export_slot(
            self, slot, root, name, tenant_digest, parametrization_fields
        ):
            return adapter.save_sampler_export(
                Path(root),
                name,
                members,
                slot,
                params,
                targets,
                tenant_digest=tenant_digest,
                lora_dropout=0.0,
            )

        async def load_slot(self, slot, rank, alpha, source=None):
            adapter.load_adapter_source(
                Path(source["path"]),
                members,
                slot,
                params,
                targets,
                load_optimizer=source["load_optimizer"],
            )

    io = lifecycle(NativeBridge())
    io.commit = pytest.importorskip(
        "miles.backends.training_utils.checkpoint.adapter.commit"
    )
    monkeypatch.setattr(adapter, "write_adapter_dir", adapter.write_adapter_dir)
    volumes.install_native_checkpoint_publish(adapter)
    capture = tmp_path / "capture"
    asyncio.run(io.capture(0, str(capture), include_optimizer=optimizer))
    persisted = tmp_path / "persisted"
    capture.rename(persisted)
    for tensors in (members.weights, members.means, members.squares):
        for tensor in tensors:
            tensor.zero_()
    members.step = 0
    asyncio.run(
        io.load_slot(0, 2, 4.0, checkpoint=str(persisted), restore_optimizer=optimizer)
    )
    assert members.finished
    for actual, wanted in zip(members.weights, expected[0], strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert members.step == (9 if optimizer else 0)
    for actual_list, wanted_list in zip(
        (members.means, members.squares), expected[1:], strict=True
    ):
        for actual, wanted in zip(actual_list, wanted_list, strict=True):
            torch.testing.assert_close(
                actual,
                wanted if optimizer else torch.zeros_like(wanted),
                rtol=0,
                atol=0,
            )
    assert not persisted.is_symlink()
    assert (persisted / "adapter_state").exists() == optimizer


def test_failed_duplicate_capture_keeps_the_previous_capture(tmp_path, monkeypatch):
    monkeypatch.delenv("SPINDLE_CHECKPOINT_VOLUME", raising=False)
    bridge = Bridge()
    io = lifecycle(bridge)
    target = tmp_path / "capture"
    asyncio.run(io.capture(0, str(target), include_optimizer=True))
    bridge.weight.add_(100)
    with pytest.raises(FileExistsError):
        asyncio.run(io.capture(0, str(target), include_optimizer=True))
    bridge.weight.zero_()
    asyncio.run(io.load_slot(0, 2, 4.0, checkpoint=str(target)))
    assert bridge.weight.tolist() == [2.0, 3.0]
    assert not (tmp_path / ".capture.abcd").exists()


@pytest.mark.parametrize(
    "field,value", [("target_modules", None), ("lora_rank", "invalid")]
)
def test_native_malformed_fields_are_preflight_errors(
    tmp_path, monkeypatch, field, value
):
    parametrization = pytest.importorskip(
        "miles.backends.training_utils.checkpoint.adapter.parametrization"
    )
    io = lifecycle(Bridge())
    source = tmp_path / "source"
    source.mkdir()
    fields = io.fields(2, 4.0) | {
        "format_version": 1,
        "target_modules": ["q_proj"],
        "shared_lora_a_groups": [],
    }
    (source / "adapter_parametrization.json").write_text(
        json.dumps(fields | {field: value})
    )
    io.read_parametrization = parametrization.read_parametrization
    with pytest.raises(native.NativeSourceError) as error:
        asyncio.run(io.load_slot(0, 2, 4.0, checkpoint=str(source)))
    assert isinstance(error.value.__cause__, TypeError)
    assert io.slots == {0: (2, 4.0)}
    assert not io.bridge.restored
