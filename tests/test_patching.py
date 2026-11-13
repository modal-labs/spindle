"""patch() re-points dependent modules only while they still refer to the
original."""

import sys
from types import ModuleType

from spindle.patching import patch


def _module(name: str, source: str) -> ModuleType:
    module = ModuleType(name)
    exec(source, module.__dict__)
    sys.modules[name] = module
    return module


def test_dependents_repointed_only_while_bound_to_original() -> None:
    upstream = _module("fake_upstream_deps", "def f():\n    return 1\n")
    stale = _module("fake_dep_stale", "")
    stale.f = upstream.f
    drifted = _module("fake_dep_drifted", "")
    drifted.f = lambda: "local"
    empty = _module("fake_dep_empty", "")

    def new():
        return 2

    patch(upstream.f, dependents=(stale, drifted, empty))(new)  # type: ignore[attr-defined]

    assert stale.f is new
    assert drifted.f() == "local"
    assert not hasattr(empty, "f")
