from spindle.configs.qwen35_9b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-lora-64k"
    trainer_image = "miles-v0.2.0-e814245"
    inference_image = "sglang-v0.5.23-e003cf9"
    max_context_length = 65536
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "trainer_gpu": "H200",
        "inference_gpu": "H200",
        "trainer_gpus_per_node": 8,
        "miles_cfg.tensor_model_parallel_size": 4,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "sglang_cfg.max_running_requests": 32,
        "sglang_cfg.max_queued_requests": 8,
    }


config = Config()
