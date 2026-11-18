from spindle.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-128k"
    trainer_image = "miles-v0.2.0-82e2671"
    inference_image = "sglang-v0.5.23-e003cf9"
    max_context_length = 131072
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.max_tokens_per_gpu": 32768,
        "miles_cfg.context_parallel_size": 4,
        "inference_gpus_per_node": 2,
        "inference_max_replicas": 4,
        "inference_target_concurrency": 4,
        "sglang_cfg.tp_size": 2,
        "sglang_cfg.max_running_requests": 8,
    }


config = Config()
