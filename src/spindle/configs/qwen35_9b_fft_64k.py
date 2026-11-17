from spindle.configs.qwen35_4b_fft_64k import Config as Parent


class Config(Parent):
    name = "qwen35-9b-fft-64k"
    model = "Qwen/Qwen3.5-9B"
    megatron_cfg = {
        **Parent.megatron_cfg,
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "logprob_chunk_tokens": None,
    }
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
