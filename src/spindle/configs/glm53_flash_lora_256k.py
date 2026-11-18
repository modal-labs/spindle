"""256K single-node GLM Flash. Create clients with train_unembed=True; routes from
sample_with_replay(return_routed_experts=True) are optional per request.
"""

from spindle.configs.glm53_flash_lora_64k_b300 import Config as Parent


class Config(Parent):
    name = "glm53-flash-lora-256k"
    trainer_image = "miles-v0.2.0-7524da0"
    inference_image = "sglang-v0.5.23-c4182b2-glm53"
    max_context_length = 262144
    inference_target_concurrency = 2
    overrides = {
        "trainer_env.PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "miles_cfg.tensor_model_parallel_size": 8,
        "miles_cfg.max_tokens_per_gpu": 262144,
        "sglang_cfg.max_total_tokens": 1048576,
    }


config = Config()
