from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "determinism_report", Path(__file__).parents[1] / "scripts/determinism_report.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compare = module.compare


def trace():
    return {
        "mode": "repeat",
        "deterministic": True,
        "initial": [1],
        "restored": [1],
        "midpoint": [2],
        "final_nll": 3.0,
        "final_evaluation": [3],
        "steps": [
            {
                "step": i,
                "forward": [i],
                "before_optimizer": [i],
                "optimizer": [i],
                "after_optimizer": [i],
            }
            for i in range(1, 13)
        ],
    }


def test_exact_repeat_and_checkpoint_suffix():
    reference = trace()
    assert compare(reference, deepcopy(reference))["equal"]
    restored = deepcopy(reference)
    restored.update(mode="restore", restored_midpoint=[2], steps=restored["steps"][6:])
    assert compare(reference, restored)["equal"]


@pytest.mark.parametrize(
    "field", ["forward", "before_optimizer", "optimizer", "after_optimizer"]
)
def test_records_first_divergent_step_and_field(field):
    reference, candidate = trace(), trace()
    candidate["steps"][2][field] = [99]
    result = compare(reference, candidate)
    assert not result["equal"]
    assert result["first_divergence"] == {"step": 3, "field": field}


@pytest.mark.parametrize(
    "change",
    ["empty", "missing", "reordered", "unexpected", "initial", "restored", "final"],
)
def test_incomplete_or_mismatched_traces_never_pass(change):
    reference, candidate = trace(), trace()
    if change == "empty":
        candidate["steps"] = []
    elif change == "missing":
        candidate["steps"].pop(3)
    elif change == "reordered":
        candidate["steps"].reverse()
    elif change == "unexpected":
        candidate["steps"][0]["step"] = 100
    elif change in ("initial", "restored"):
        candidate[change] = [99]
    else:
        candidate.update(final_evaluation=[99], final_nll=3.000001)
    result = compare(reference, candidate)
    assert not result["equal"]
    if change == "final":
        assert result["replay_nll_difference"] == pytest.approx(1e-6)
