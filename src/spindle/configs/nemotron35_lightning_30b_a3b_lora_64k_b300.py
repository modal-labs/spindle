from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    """Nemotron 3.5 Lightning LoRA with 64K context on two B300s."""

    name = "nemotron35-lightning-30b-a3b-lora-64k-b300"
    trainer_image = "miles-v0.1.1-fdd808c"
    inference_image = "sglang-v0.5.23-c4182b2"
    trainer_gpu = "B300"
    overrides = {
        "miles_cfg.tensor_model_parallel_size": 1,
        "miles_cfg.expert_model_parallel_size": 1,
        "miles_cfg.max_tokens_per_gpu": 65536,
        "miles_cfg.cli_options.recompute_method": "block",
        "miles_cfg.cli_options.recompute_num_layers": 36,
        "trainer_env.EXPERT_GROUPED_GEMM": "1",
        "trainer_env.FUSED_WEIGHTED_SQUARED_RELU": "1",
    }


config = Config()
