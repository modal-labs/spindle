from types import SimpleNamespace

import pytest

from spindle.backends.miles_runtime import fused_squared_relu


class Bridge:
    def to_megatron_provider(
        self, load_weights: bool = True, hf_path: str | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            use_fused_weighted_squared_relu=False, load_weights=load_weights
        )


def test_provider_gets_the_fused_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fused_squared_relu.ENV_FLAG, "1")

    fused_squared_relu.install(Bridge)
    provider = Bridge().to_megatron_provider(load_weights=False)

    assert provider.use_fused_weighted_squared_relu
    assert not provider.load_weights
