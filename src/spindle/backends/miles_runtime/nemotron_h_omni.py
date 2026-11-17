"""Train Nemotron-H Omni checkpoints as their Nemotron-H language model.

Nemotron 3.5 Super VL nests the Nemotron 3 Super config under ``llm_config`` and
stores its tensors under ``language_model.``. Megatron-Bridge releases without a
bridge for the architecture cannot build or export the trainer model.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"
PREFIX = "language_model."


def install(hf_checkpoint: str) -> None:
    config = Path(hf_checkpoint, "config.json")
    if not config.is_file():
        return
    if ARCHITECTURE not in json.loads(config.read_text()).get("architectures", ()):
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
    MegatronModelBridge.register_bridge(source=ARCHITECTURE, target=MambaModel)(
        language_model_bridge(miles_bridge, MegatronMappingRegistry)
    )
    # AutoBridge accepts only architectures whose names end in a listed suffix.
    auto_bridge.SUPPORTED_HF_ARCHITECTURES += (ARCHITECTURE,)
    auto_bridge.SUPPORTED_HF_ARCHITECTURES_DISPLAY += f" or '{ARCHITECTURE}'"


def language_model_bridge(
    nemotron_h_bridge: Any, mapping_registry: Callable[..., Any]
) -> Any:
    class NemotronHOmniBridge(nemotron_h_bridge):
        def provider_bridge(self, hf_pretrained: Any) -> Any:
            config = getattr(hf_pretrained, "config", hf_pretrained)
            return super().provider_bridge(SimpleNamespace(config=config.llm_config))

        def mapping_registry(self) -> Any:
            omni_config = self.hf_config
            self.hf_config = omni_config.llm_config
            try:
                mappings = super().mapping_registry().mappings
            finally:
                self.hf_config = omni_config
            return mapping_registry(*map(_under_language_model, mappings))

    return NemotronHOmniBridge


def _under_language_model(mapping: Any) -> Any:
    mapping = copy.copy(mapping)
    names = mapping.hf_param
    mapping.hf_param = (
        PREFIX + names
        if isinstance(names, str)
        else {part: PREFIX + name for part, name in names.items()}
    )
    return mapping
