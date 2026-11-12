import runpy
from pathlib import Path

from spindle.deployments import DeploymentConfig, load


def test_reference_hardware_matches_resolved_deployment() -> None:
    directory = Path(__file__).resolve().parents[1] / "reference_runs"
    client = runpy.run_path(str(directory / "dapo_math.py"))
    deployment = DeploymentConfig.create(load(directory / "dapo_math_config.py"))
    recipe = deployment.recipe
    trainer = deployment.trainer_settings["miles"]
    hardware = client["HARDWARE"]
    assert client["RECIPE"] == recipe.name
    assert hardware["trainer"] == {
        "gpu": recipe.trainer_gpu,
        "gpus": trainer["actor_num_gpus_per_node"] * trainer["actor_num_nodes"],
    }
    assert trainer["tensor_model_parallel_size"] == 1
    assert hardware["sampler"] == {
        "gpu": recipe.inference_gpu,
        "gpus_per_replica": deployment.inference_settings["tp_size"],
        "replicas": recipe.inference_min_replicas,
    }
    assert recipe.inference_min_replicas == recipe.inference_max_replicas
    assert recipe.trainer_max_instances == recipe.trainer_max_clients_per_instance == 1
