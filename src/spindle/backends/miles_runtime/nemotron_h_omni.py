"""Train Nemotron-H Omni checkpoints as their Nemotron-H language model.

The checkpoint nests that model's config under ``llm_config`` and its tensors
under ``language_model.``, and Megatron-Bridge has no bridge for the architecture.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
PREFIX = "language_model."


class _Mapping(Protocol):
    hf_param: str | dict[str, str]


def install(hf_checkpoint: str) -> None:
    config = Path(hf_checkpoint, "config.json")
    if not config.is_file():
        return
    if ARCHITECTURE not in (json.loads(config.read_text()).get("architectures") or ()):
        return
    # Megatron-Bridge and Miles exist only in the trainer image.
    import miles_plugins.megatron_bridge  # noqa: F401
    from megatron.bridge.models.conversion import auto_bridge
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge
    from megatron.core.models.mamba import MambaModel

    if ARCHITECTURE in auto_bridge.SUPPORTED_HF_ARCHITECTURES:
        return
    (miles_bridge,) = [
        bridge
        for bridge in NemotronHBridge.__subclasses__()
        if bridge.__name__ == "MilesNemotronHBridge"
    ]
    # Bridge annotates this decorator as taking a bridge instance.
    register = cast(
        "Callable[[type], object]",
        MegatronModelBridge.register_bridge(source=ARCHITECTURE, target=MambaModel),
    )
    register(language_model_bridge(miles_bridge, MegatronMappingRegistry))
    # AutoBridge accepts only architectures whose names end in a listed suffix.
    auto_bridge.SUPPORTED_HF_ARCHITECTURES += (ARCHITECTURE,)


def language_model_bridge(
    nemotron_h_bridge: type, registry: Callable[..., object]
) -> type:
    class NemotronHOmniBridge(nemotron_h_bridge):  # ty: ignore[unsupported-base]
        def provider_bridge(self, hf_pretrained: object) -> object:
            language_model = SimpleNamespace(config=hf_pretrained.config.llm_config)
            return super().provider_bridge(language_model)

        def mapping_registry(self) -> object:
            omni_config = self.hf_config
            self.hf_config = omni_config.llm_config
            try:
                mappings = super().mapping_registry().mappings
            finally:
                self.hf_config = omni_config
            return registry(*map(_under_language_model, mappings))

    return NemotronHOmniBridge


def _under_language_model(mapping: _Mapping) -> _Mapping:
    mapping = copy.copy(mapping)
    names = mapping.hf_param
    mapping.hf_param = (
        PREFIX + names
        if isinstance(names, str)
        else {part: PREFIX + name for part, name in names.items()}
    )
    return mapping
