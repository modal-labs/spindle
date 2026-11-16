from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    """Nemotron 3.5 Lightning LoRA with 64K context on two B300s.

    A B300 holds the whole model and a 64K row, so each GPU trains its own rows
    and no layer is split between them. The spare memory also lets the last 16
    layers keep their activations instead of recomputing them in backward.
    """

    name = "nemotron35-lightning-30b-a3b-lora-64k-b300"
    trainer_gpu = "B300"
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.cli_options.recompute_method": "block",
        "miles_cfg.cli_options.recompute_num_layers": 36,
    }


config = Config()
