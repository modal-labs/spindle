from spindle.configs.qwen35_4b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-fft-64k"
    trainer_image = "megatron-2d1fa8d"
    inference_image = "sglang-v0.5.23-c4182b2"
    model = "Qwen/Qwen3.5-9B"
    trainer_gpu = ["B300", "B200"]
    trainer_gpus_per_node = 4
    megatron_cfg = {
        **Parent.megatron_cfg,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "logprob_chunk_tokens": 8192,
    }
    trainer_env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        # The fp32 LM head takes bf16 inputs, which TF32 represents exactly.
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
    }
    overrides = {
        "inference_gpu": "H200",
        "sglang_cfg.ep_size": 1,
    }


config = Config()
