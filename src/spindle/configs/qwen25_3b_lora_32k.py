from spindle.configs.qwen3_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "qwen25-3b-lora-32k"
    trainer_image = "miles-v0.2.0-c82136c"
    inference_image = "sglang-v0.5.23-e003cf9"
    model = "Qwen/Qwen2.5-3B"
    inference_max_replicas = 8
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "miles_cfg.model_type": "qwen2.5-3B",
    }


config = Config()
