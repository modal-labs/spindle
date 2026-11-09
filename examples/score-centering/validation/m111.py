from spindle.configs.qwen3_4b_lora_32k import Config as Parent

from validation_config import OVERRIDES


class Config(Parent):
    name = "m111"
    overrides = OVERRIDES | {
        "weights_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "trainer_max_clients_per_instance": 2,
        "miles_cfg.max_lora_slots": 2,
        "miles_cfg.max_lora_rank": 16,
        "miles_cfg.default_lora_alpha": 16,
        "miles_cfg.max_tokens_per_gpu": 2048,
        "miles_cfg.context_parallel_size": 1,
        "miles_cfg.align_sequences_to_parallel_layout": True,
        "sglang_cfg.max_loaded_loras": 8,
        "sglang_cfg.max_loras_per_batch": 2,
    }


config = Config()
