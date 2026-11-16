from types import SimpleNamespace

from spindle.backends.miles_runtime import fused_squared_relu


class Bridge:
    @staticmethod
    def to_megatron_provider(bridge, load_weights=False):
        return SimpleNamespace(use_fused_weighted_squared_relu=False, load_weights=load_weights)


def test_provider_gets_the_fused_activation(monkeypatch) -> None:
    monkeypatch.setenv(fused_squared_relu.ENV_FLAG, "1")
    bridge = type("PatchedBridge", (Bridge,), {})

    assert fused_squared_relu.install(bridge)
    provider = bridge.to_megatron_provider(None, load_weights=True)

    assert provider.use_fused_weighted_squared_relu
    assert provider.load_weights


def test_disabled_without_the_recipe_flag(monkeypatch) -> None:
    monkeypatch.delenv(fused_squared_relu.ENV_FLAG, raising=False)
    bridge = type("UntouchedBridge", (Bridge,), {})

    assert not fused_squared_relu.install(bridge)
    assert not bridge.to_megatron_provider(None).use_fused_weighted_squared_relu
