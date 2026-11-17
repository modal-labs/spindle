from spindle.configs.qwen35_35b_a3b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen36-35b-a3b-fft-64k"
    trainer_image = "megatron-2d1fa8d"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = "Qwen/Qwen3.6-35B-A3B"
    overrides = {"sglang_cfg.dp_size": 1, "sglang_cfg.enable_dp_attention": False}


config = Config()
