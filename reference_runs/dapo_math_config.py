"""Fixed hardware for the DAPO-Math reference run."""

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "dapo-math-qwen35-4b-h100"
    trainer_gpu = "H100!"
    trainer_gpus_per_node = 4
    trainer_nodes = 1
    trainer_max_instances = 1
    trainer_max_clients_per_instance = 1
    inference_gpu = "H100!"
    inference_gpus_per_node = 1
    inference_min_replicas = 2
    inference_max_replicas = 2


config = Config()
