from spindle.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-64k"
    trainer_image = "miles-v0.2.0-1c27e6d"
    inference_image = "sglang-v0.5.23-e003cf9"
    max_context_length = 65536
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 2,
        "inference_target_concurrency": 8,
        "sglang_cfg.max_running_requests": 16,
    }


config = Config()
