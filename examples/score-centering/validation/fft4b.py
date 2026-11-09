from spindle.configs.qwen35_4b_fft_64k import Config as Parent

from validation_config import OVERRIDES


class Config(Parent):
    name = "fft4b"
    overrides = OVERRIDES | {
        "weights_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "megatron_cfg.tensor_model_parallel_size": 1,
        "megatron_cfg.context_parallel_size": 1,
        "megatron_cfg.sequence_parallel": False,
        "megatron_cfg.max_tokens_per_microbatch": 2048,
        "megatron_cfg.optimizer.lr": 1e-06,
        "megatron_cfg.optimizer.min_lr": 1e-06,
    }


config = Config()
