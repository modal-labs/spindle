from spindle.configs.qwen35_4b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-fft-64k"
    trainer_image = "megatron-2d1fa8d"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = "Qwen/Qwen3.5-9B"
    overrides = {
        "trainer_gpu": "H200",
        "trainer_env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        },
        "inference_gpu": "H200",
        "sglang_cfg.ep_size": 1,
    }


config = Config()
