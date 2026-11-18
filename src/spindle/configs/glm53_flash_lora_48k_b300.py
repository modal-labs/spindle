"""Single-node B300 GLM Flash: TP2, EP8, and one 48K row per DP rank.

The 9.31K input tokens/s measurement used the previous Miles image, rank 32,
24 sequences, and masked cross-entropy. Revalidate throughput on this image;
that measurement does not cover rank 16, lm_head, or rollout routing replay.
"""

from spindle.configs.glm53_flash_lora_64k import Config as GLMConfig


class Config(GLMConfig):
    name = "glm53-flash-lora-48k-b300"
    trainer_image = "miles-v0.2.0-ac54685"
    inference_image = "sglang-v0.5.23-c4182b2-glm53"
    max_context_length = 49152
    trainer_gpu = "B300"
    trainer_nodes = 1
    trainer_rdma = True
    inference_gpu = "B300"
    overrides = {
        "miles_cfg.model_type": "glm5.3-flash",
        "miles_cfg.megatron_to_hf_mode": "raw",
        "miles_cfg.lora_type": "canonical_lora",
        # Native expert slot routing needs the dispatcher's unfused permutation map.
        "miles_cfg.cli_options.moe_permute_fusion": False,
        "miles_cfg.target_modules": [
            f"model.language_model.layers.*.{target.removeprefix('*.')}"
            for target in GLMConfig.miles_cfg["target_modules"]
        ],
        "miles_cfg.tensor_model_parallel_size": 2,
        "miles_cfg.expert_model_parallel_size": 8,
        "miles_cfg.max_tokens_per_gpu": 49152,
    }


config = Config()
