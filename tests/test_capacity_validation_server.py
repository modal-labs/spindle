from pathlib import Path

import pytest

from spindle.deployments import load

RECIPES = sorted(
    (Path(__file__).parents[1] / "examples/capacity-validation-server/recipes").glob(
        "*.py"
    )
)


@pytest.mark.parametrize("path", RECIPES, ids=lambda path: path.stem)
def test_capacity_validation_recipes_run_on_one_b300_node(path):
    recipe = load(path)
    assert recipe.trainer_gpu == "B300"
    assert recipe.inference_gpu == "B300"
    assert recipe.trainer_nodes == 1
    assert recipe.trainer_gpus_per_node <= 8
    assert recipe.inference_gpus_per_node <= 8
    assert recipe.platform["modal"]["environment"] == "capacity-validation"
    assert recipe.platform["frontend"] == "cv"
