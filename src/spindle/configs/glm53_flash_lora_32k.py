from spindle.configs.glm53_flash_lora_64k import Config as Parent


class Config(Parent):
    name = "glm53-flash-lora-32k"
    trainer_image = "miles-v0.1.2-464efc9-glm53"
    inference_image = "sglang-v0.5.23-c4182b2-glm53"
    max_context_length = 32768
    inference_draft_model = "modal-labs/GLM-5.3-Flash-DFlash"
    inference_draft_model_revision = "dae6d319510889a7f1ac5242d5b7fb2d5d95eb05"
    overrides = {
        "miles_cfg.max_tokens_per_gpu": 32768,
        "sglang_cfg.max_running_requests": 64,
        "sglang_cfg.max_queued_requests": 128,
        "sglang_cfg.max_loaded_loras": 4,
        "sglang_cfg.max_loras_per_batch": 2,
        "sglang_cfg.dtype": "bfloat16",
        "sglang_cfg.page_size": 256,
        "sglang_cfg.language_only": True,
        "sglang_cfg.mamba_ssm_dtype": "float32",
        "sglang_cfg.speculative_algorithm": "DFLASH",
        "sglang_cfg.speculative_dflash_block_size": 8,
        "sglang_cfg.speculative_draft_attention_backend": "fa4",
        "sglang_cfg.speculative_draft_model_quantization": "unquant",
        "sglang_cfg.speculative_draft_kv_cache_dtype": "bfloat16",
    }


config = Config()
