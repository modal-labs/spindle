from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-instruct-lora-32k"
    trainer_image = "miles-v0.2.0-e1b172f"
    inference_image = "sglang-v0.5.23-e003cf9"
    max_context_length = 32768
    trainer_cpu = 32
    trainer_memory_mib = 196608
    trainer_max_clients_per_instance = 1
    inference_cpu = 24
    inference_memory_mib = 131072
    inference_target_concurrency = 64
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "trainer_env.TORCH_COMPILE_DISABLE": "1",
        "miles_cfg.max_lora_slots": 1,
        "miles_cfg.max_tokens_per_gpu": 18432,
        "miles_cfg.cli_options": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
        "sglang_cfg.max_loaded_loras": 32,
        "sglang_cfg.max_loras_per_batch": 2,
    }


config = Config()
