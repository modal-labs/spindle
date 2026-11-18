from spindle.configs.qwen38_27b_lora_16k import Config as Parent


class Config(Parent):
    name = "qwen38-27b-lora-16k-b200"
    trainer_image = "miles-v0.2.0-1c27e6d"
    inference_image = "sglang-v0.5.23-e003cf9"
    inference_gpu = "B200"
    overrides = {
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        "sglang_cfg.mamba_full_memory_ratio": 0.4,
        "sglang_cfg.attention_backend": "trtllm_mha",
        "sglang_cfg.page_size": 64,
    }


config = Config()
