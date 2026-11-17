import json
from pathlib import Path
from types import SimpleNamespace

from spindle.backends.miles_runtime import nemotron_h_omni


class NemotronHBridge:
    hf_config: object

    def provider_bridge(self, hf_pretrained: SimpleNamespace) -> object:
        return hf_pretrained.config

    def mapping_registry(self) -> SimpleNamespace:
        assert self.hf_config == "language config"
        return SimpleNamespace(
            mappings=[
                SimpleNamespace(hf_param="backbone.embeddings.weight"),
                SimpleNamespace(
                    hf_param={"q": "backbone.layers.*.mixer.q_proj.weight"}
                ),
            ]
        )


def test_omni_checkpoints_convert_as_their_language_model() -> None:
    bridge = nemotron_h_omni.language_model_bridge(
        NemotronHBridge, lambda *mappings: list(mappings)
    )()
    omni_config = SimpleNamespace(llm_config="language config")

    assert bridge.provider_bridge(SimpleNamespace(config=omni_config)) == (
        "language config"
    )
    bridge.hf_config = omni_config
    embeddings, attention = bridge.mapping_registry()
    assert bridge.hf_config is omni_config
    assert embeddings.hf_param == "language_model.backbone.embeddings.weight"
    assert attention.hf_param == {
        "q": "language_model.backbone.layers.*.mixer.q_proj.weight"
    }


def test_other_checkpoints_are_left_alone(tmp_path: Path) -> None:
    nemotron_h_omni.install(str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["NemotronHForCausalLM"]})
    )
    nemotron_h_omni.install(str(tmp_path))
