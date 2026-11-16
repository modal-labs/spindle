from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spindle.backends.miles_runtime import expert_grouped_gemm  # noqa: E402


class GroupedLinear(torch.nn.Module):
    def __init__(self, experts: int, out_features: int, in_features: int) -> None:
        super().__init__()
        self.num_gemms = experts
        self.use_bias = False
        self.config = SimpleNamespace(fp8=None, fp4=None)
        for index in range(experts):
            weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
            self.register_parameter(f"weight{index}", torch.nn.Parameter(weight, requires_grad=False))

    def forward(self, x, m_splits):
        rows = x.split(list(m_splits))
        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        return torch.cat([row @ weight.T for row, weight in zip(rows, weights)]), None


def test_stacking_keeps_each_expert_parameter_as_a_view() -> None:
    module = GroupedLinear(experts=3, out_features=8, in_features=16)
    before = [module.weight0.clone(), module.weight1.clone(), module.weight2.clone()]

    stacked = expert_grouped_gemm.stacked_weight(module)

    assert stacked.shape == (3, 8, 16)
    for index, weight in enumerate(before):
        parameter = getattr(module, f"weight{index}")
        assert torch.equal(parameter, weight)
        assert parameter.data_ptr() == stacked[index].data_ptr()
    assert expert_grouped_gemm.stacked_weight(module) is stacked


def test_trainable_or_cpu_inputs_keep_the_transformer_engine_path(monkeypatch) -> None:
    monkeypatch.setenv(expert_grouped_gemm.ENV_FLAG, "1")

    class Patched(GroupedLinear):
        pass

    assert expert_grouped_gemm.install(Patched)
    module = Patched(experts=2, out_features=8, in_features=16)
    x = torch.randn(5, 16, dtype=torch.bfloat16)

    out, bias = module(x, [2, 3])

    assert bias is None
    assert torch.equal(out, GroupedLinear.forward(module, x, [2, 3])[0])
    assert not hasattr(module, "_spindle_stacked_weight")


def test_disabled_without_the_recipe_flag(monkeypatch) -> None:
    monkeypatch.delenv(expert_grouped_gemm.ENV_FLAG, raising=False)

    class Untouched(GroupedLinear):
        pass

    forward = Untouched.forward
    assert not expert_grouped_gemm.install(Untouched)
    assert Untouched.forward is forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMM needs CUDA")
def test_grouped_gemm_matches_one_gemm_per_expert(monkeypatch) -> None:
    monkeypatch.setenv(expert_grouped_gemm.ENV_FLAG, "1")

    class Patched(GroupedLinear):
        pass

    expert_grouped_gemm.install(Patched)
    module = Patched(experts=4, out_features=64, in_features=128).cuda()
    x = torch.randn(40, 128, dtype=torch.bfloat16, device="cuda")
    splits = [16, 0, 8, 16]

    out, _ = module(x, splits)

    torch.testing.assert_close(out, GroupedLinear.forward(module, x, splits)[0])
