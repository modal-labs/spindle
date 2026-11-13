"""Deployment settings used for the recorded Miles v0.2.0 run.

The reference recipe in reference_runs/dapo_math_config.py with the new trainer
image, deployed with the frontend, trainer and samplers in us-east.
"""

from copy import deepcopy

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "dapo-math-qwen35-4b-h100"
    trainer_image = "miles-v0.2.0"
    trainer_gpu = "H100!"
    trainer_gpus_per_node = 4
    trainer_nodes = 1
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 1
    inference_gpu = "H100!"
    inference_gpus_per_node = 1
    inference_min_replicas = 2
    inference_max_replicas = 2
    platform = deepcopy(Parent.platform)
    platform["modal"]["region"] = "us-east"


config = Config()
