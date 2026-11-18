from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spindle.backends.miles_runtime import expert_grouped_gemm  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM needs CUDA"
)


class GroupedLinear(torch.nn.Module):
    def __init__(self, experts: int, out_features: int, in_features: int) -> None:
        super().__init__()
        self.num_gemms = experts
        self.use_bias = False
        self.config = SimpleNamespace(fp8=None, fp4=None)
        for index in range(experts):
            weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
            self.register_parameter(
                f"weight{index}", torch.nn.Parameter(weight, requires_grad=False)
            )

    def forward(self, x: torch.Tensor, m_splits: list[int]) -> tuple:
        rows = x.split(list(m_splits))
        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        return torch.cat([row @ weight.T for row, weight in zip(rows, weights)]), None


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> type:
    monkeypatch.setenv(expert_grouped_gemm.ENV_FLAG, "1")
    cls = type("Patched", (GroupedLinear,), {})
    assert expert_grouped_gemm.install(cls)
    return cls


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


@needs_cuda
def test_grouped_gemm_matches_one_gemm_per_expert(patched: type) -> None:
    module = patched(experts=4, out_features=64, in_features=128).cuda()
    x = torch.randn(37, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    splits = [13, 0, 7, 17]

    out, _ = module(x, splits)
    reference = GroupedLinear.forward(module, x, splits)[0]
    grad = torch.randn_like(out)

    assert "_spindle_stacked_weight" in module.__dict__
    torch.testing.assert_close(out, reference)
    torch.testing.assert_close(
        torch.autograd.grad(out, x, grad)[0], torch.autograd.grad(reference, x, grad)[0]
    )


@needs_cuda
def test_trainable_experts_keep_the_transformer_engine_path(patched: type) -> None:
    module = patched(experts=2, out_features=64, in_features=128).cuda()
    module.weight0.requires_grad_(True)

    module(torch.randn(8, 128, dtype=torch.bfloat16, device="cuda"), [4, 4])

    assert "_spindle_stacked_weight" not in module.__dict__
