import sys
import types
from collections.abc import Callable

import pytest

from spindle.backends import fla_kernels


class FakeKernel:
    def __init__(self, arg_names: list[str]) -> None:
        self.arg_names = arg_names
        self.launches: list[dict] = []

    def __getitem__(self, grid: object) -> Callable[..., None]:
        return lambda **kwargs: self.launches.append(kwargs)


def test_pin_fixes_block_count_once_and_skips_kernels_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("fake_fla_ops")
    with_block_count = FakeKernel(["x", "NB"])
    without_block_count = FakeKernel(["x"])
    module.with_block_count = with_block_count
    module.without_block_count = without_block_count
    monkeypatch.setitem(sys.modules, "fake_fla_ops", module)
    monkeypatch.setattr(
        fla_kernels,
        "FLA_KERNELS",
        {"fake_fla_ops": ("with_block_count", "without_block_count", "missing")},
    )

    fla_kernels.pin_fla_block_counts()
    fla_kernels.pin_fla_block_counts()

    module.with_block_count[(4,)](x=3, NB=37)
    assert with_block_count.launches == [{"x": 3, "NB": 1}]
    assert module.with_block_count.kernel is with_block_count
    assert module.without_block_count is without_block_count
