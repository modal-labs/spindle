"""256K single-node GLM Flash. Create clients with train_unembed=True; routes from
sample_with_replay(return_routed_experts=True) are optional per request.
"""

from spindle.configs.glm53_flash_lora_64k import Config as LegacyConfig
from spindle.configs.glm53_flash_lora_64k_b300 import Config as Parent


class Config(Parent):
    name = "glm53-flash-lora-256k"
    trainer_image = "miles-v0.1.2-99e1ebe-kpool-glm53"
    inference_image = "sglang-v0.5.23-c4182b2-glm53"
    max_context_length = 262144
    inference_target_concurrency = 2
    overrides = {
        # The 256K indexer patch currently lives only in the Bridge image.
        "miles_cfg.model_type": "",
        "miles_cfg.megatron_to_hf_mode": "bridge",
        "miles_cfg.lora_type": "lora",
        "miles_cfg.target_modules": [
            *LegacyConfig.miles_cfg["target_modules"],
            "lm_head",
        ],
        "trainer_env.PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "miles_cfg.tensor_model_parallel_size": 8,
        "miles_cfg.max_tokens_per_gpu": 262144,
        "sglang_cfg.max_total_tokens": 1048576,
    }


config = Config()
