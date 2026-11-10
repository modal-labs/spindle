from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    """Nemotron 3.5 Lightning LoRA for rows up to 32K tokens.

    Each GPU holds the whole model and trains its own rows, which takes less
    time per step than the 64K recipe. A 64K row does not fit on one GPU.
    """

    name = "nemotron35-lightning-30b-a3b-lora-32k"
    max_context_length = 32768
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_tokens_per_gpu": 24576,
    }


config = Config()
