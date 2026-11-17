"""Experimental opt-in GLM Flash shared-prefix LoRA; one trainer and adapter."""

from spindle.configs.glm53_flash_lora_64k import Config as Parent


class Config(Parent):
    name = "glm53-flash-lora-shared-prefix-b300"
    trainer_image = "spindle.providers.modal.shared_prefix_image:glm53_trainer_image"
    trainer_gpu = "B300"
    trainer_gpus_per_node = 8
    trainer_nodes = 1
    trainer_memory_mib = 786432
    trainer_cpu = 64
    trainer_max_clients_per_instance = 1
    trainer_env = {
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    inference_gpu = "B300"
    inference_memory_mib = 786432
    max_context_length = 32768
    overrides = {
        "platform.modal.exclude_reserved_capacity": True,
        "miles_cfg.shared_prefix_training": True,
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.expert_model_parallel_size": 8,
        "miles_cfg.max_lora_slots": 1,
        "miles_cfg.max_tokens_per_gpu": 49152,
        "miles_cfg.cli_options.deterministic_mode": True,
        "miles_cfg.cli_options.use_rollout_routing_replay": False,
        "miles_cfg.cli_options.enable_mtp_training": False,
        "sglang_cfg.enable_return_routed_experts": False,
    }


config = Config()
