"""Owned CPU adapter tensors for publication outside the training lane."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from miles.backends.training_utils.checkpoint.adapter.parametrization import (
        AdapterParametrization,
    )
    from miles.backends.training_utils.checkpoint.adapter.probed_slots import (
        ProbedAdapterSlots,
    )
    from miles.utils.lora.hf_lora_targets import HfLoraTargets


@dataclass(frozen=True)
class AdapterShard:
    key: str
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    tensor: torch.Tensor


@dataclass(frozen=True)
class SamplerLayout:
    shards: tuple[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]
    sources: tuple[tuple[int, torch.Tensor], ...]

    @classmethod
    def from_slots(cls, slots: ProbedAdapterSlots, slot: int) -> SamplerLayout:
        specs = slots.get_local_shard_specs(slot)
        sources = slots.piece_parameter(slot, None)
        shards, selected = [], []
        for spec, source in zip(specs, sources, strict=True):
            # Padded vocabulary rows belong to checkpoints, not sampler exports.
            if spec.is_primary and not spec.key.endswith(".vocab_padding"):
                shards.append(
                    (spec.key, spec.global_shape, spec.offsets, spec.local_shape)
                )
                selected.append(source)
        return cls(tuple(shards), tuple(selected))

    def capture(self, parameters: Sequence[torch.Tensor]) -> dict[int, torch.Tensor]:
        with torch.profiler.record_function("spindle/capture_sampler_snapshot"):
            return {
                index: parameters[index].detach().to(device="cpu", copy=True)
                for index in sorted({index for index, _ in self.sources})
            }

    def extract(self, parameters: dict[int, torch.Tensor]) -> tuple[AdapterShard, ...]:
        return tuple(
            AdapterShard(
                key,
                shape,
                offsets,
                parameters[index].reshape(-1)[elements.flatten()].reshape(local_shape),
            )
            for (key, shape, offsets, local_shape), (index, elements) in zip(
                self.shards, self.sources, strict=True
            )
        )


@dataclass(frozen=True)
class SamplerExport:
    shards: tuple[AdapterShard, ...]
    parametrization: AdapterParametrization
    targets: HfLoraTargets
    lora_dropout: float

    @classmethod
    def combine(cls, parts: Sequence[SamplerExport]) -> SamplerExport:
        if not parts:
            raise ValueError("sampler capture has no ranks")
        first = parts[0]
        if any(
            part.parametrization != first.parametrization
            or part.targets != first.targets
            or part.lora_dropout != first.lora_dropout
            for part in parts[1:]
        ):
            raise ValueError("sampler capture metadata differs across ranks")
        return cls(
            tuple(shard for part in parts for shard in part.shards),
            first.parametrization,
            first.targets,
            first.lora_dropout,
        )

    def weights(self) -> dict[str, torch.Tensor]:
        by_key: dict[str, list[AdapterShard]] = defaultdict(list)
        for shard in self.shards:
            by_key[shard.key].append(shard)
        if not by_key:
            raise ValueError("sampler capture contains no adapter tensors")
        weights = {}
        for key, shards in by_key.items():
            shape, dtype = shards[0].global_shape, shards[0].tensor.dtype
            boxes = []
            count = 0
            for shard in shards:
                tensor = shard.tensor
                if (
                    tensor.device.type != "cpu"
                    or tensor.dtype != dtype
                    or shard.global_shape != shape
                    or len(shard.offsets) != len(shape)
                    or tensor.ndim != len(shape)
                    or any(
                        offset < 0 or size <= 0 or offset + size > full
                        for offset, size, full in zip(
                            shard.offsets, tensor.shape, shape, strict=True
                        )
                    )
                ):
                    raise ValueError(f"invalid sampler shard for {key}")
                box = tuple(
                    (offset, offset + size)
                    for offset, size in zip(shard.offsets, tensor.shape, strict=True)
                )
                if any(
                    all(
                        a < d and c < b
                        for (a, b), (c, d) in zip(box, other, strict=True)
                    )
                    for other in boxes
                ):
                    raise ValueError(f"overlapping sampler shards for {key}")
                boxes.append(box)
                count += tensor.numel()
            if count != math.prod(shape):
                raise ValueError(f"incomplete sampler shards for {key}")
            whole = torch.empty(shape, dtype=dtype, device="cpu")
            for shard, box in zip(shards, boxes, strict=True):
                whole[tuple(slice(start, end) for start, end in box)].copy_(
                    shard.tensor
                )
            weights[key] = whole
        return weights

    def write(self, path: Path) -> None:
        # Miles is installed only in trainer images.
        from miles.backends.training_utils.checkpoint.adapter.adapter_exports import (
            write_sampler_adapter,
        )

        write_sampler_adapter(
            path,
            self.weights(),
            self.parametrization,
            self.targets,
            lora_dropout=self.lora_dropout,
        )


@dataclass(frozen=True)
class CpuSamplerSnapshot:
    layout: SamplerLayout
    parameters: dict[int, torch.Tensor]
    parametrization: AdapterParametrization
    targets: HfLoraTargets
    lora_dropout: float

    @classmethod
    def capture(
        cls,
        layout: SamplerLayout,
        parameters: Sequence[torch.Tensor],
        parametrization: AdapterParametrization,
        targets: HfLoraTargets,
        lora_dropout: float,
    ) -> CpuSamplerSnapshot:
        return cls(
            layout,
            layout.capture(parameters),
            deepcopy(parametrization),
            deepcopy(targets),
            lora_dropout,
        )

    def export(self) -> SamplerExport:
        return SamplerExport(
            self.layout.extract(self.parameters),
            self.parametrization,
            self.targets,
            self.lora_dropout,
        )
