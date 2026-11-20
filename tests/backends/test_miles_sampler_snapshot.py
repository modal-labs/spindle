from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from spindle.backends.miles_runtime.sampler_snapshot import (
    AdapterShard,
    CpuSamplerSnapshot,
    SamplerExport,
    SamplerLayout,
)


def test_snapshot_survives_slot_mutation_and_reuse() -> None:
    parameter = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).requires_grad_()
    indices = torch.tensor([[0, 2], [8, 10]], dtype=torch.int32)
    spec = SimpleNamespace(
        key="layer.lora_A",
        global_shape=(2, 2),
        offsets=(0, 0),
        local_shape=(2, 2),
        is_primary=True,
    )
    slots = SimpleNamespace(
        get_local_shard_specs=lambda slot: [
            spec,
            replace_shard(spec, is_primary=False),
        ],
        piece_parameter=lambda slot, holds: [(0, indices), (1, indices)],
    )
    layout = SamplerLayout.from_slots(slots, 0)
    metadata = {"rank": 2, "groups": ["q", "k"]}
    first = CpuSamplerSnapshot.capture(
        layout, [parameter, parameter], metadata, {}, 0.0
    )
    with torch.no_grad():
        parameter.add_(100)
    second = CpuSamplerSnapshot.capture(
        layout, [parameter, parameter], metadata, {}, 0.0
    )
    metadata["groups"].clear()
    expected = torch.tensor([[0, 2], [8, 10]], dtype=torch.bfloat16)
    torch.testing.assert_close(
        first.export().weights()[spec.key], expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        second.export().weights()[spec.key], expected + 100, rtol=0, atol=0
    )
    assert first.parametrization["groups"] == ["q", "k"]
    assert set(first.parameters) == {0}
    assert not first.parameters[0].requires_grad


def replace_shard(spec: SimpleNamespace, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(**(vars(spec) | fields))


def test_rank_shards_assemble_without_transporting_layout_indices() -> None:
    first = AdapterShard(
        "expert.lora_B", (2, 4), (0, 0), torch.tensor([[1, 2], [5, 6]])
    )
    second = AdapterShard(
        "expert.lora_B", (2, 4), (0, 2), torch.tensor([[3, 4], [7, 8]])
    )
    parts = [SamplerExport((shard,), {"rank": 2}, {}, 0.0) for shard in (first, second)]
    merged = SamplerExport.combine(parts)
    torch.testing.assert_close(
        merged.weights()[first.key], torch.arange(1, 9).reshape(2, 4)
    )
    with pytest.raises(ValueError, match="metadata differs"):
        SamplerExport.combine(
            [parts[0], replace(parts[1], parametrization={"rank": 4})]
        )


@pytest.mark.parametrize("damage", ["overlap", "gap", "bounds", "dtype", "shape"])
def test_invalid_shards_fail_before_an_adapter_can_be_written(damage: str) -> None:
    first = AdapterShard("layer.lora_B", (4, 2), (0, 0), torch.ones(2, 2))
    second = replace(first, offsets=(2, 0))
    if damage == "overlap":
        second = replace(second, offsets=(0, 0))
    elif damage == "gap":
        second = replace(second, tensor=torch.ones(1, 2))
    elif damage == "bounds":
        second = replace(second, offsets=(3, 0))
    elif damage == "dtype":
        second = replace(second, tensor=torch.ones(2, 2, dtype=torch.bfloat16))
    else:
        second = replace(second, global_shape=(5, 2))
    with pytest.raises(ValueError, match="sampler shard"):
        SamplerExport((first, second), {}, {}, 0.0).weights()
